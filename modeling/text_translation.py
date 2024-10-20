from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from .scheduler import CustomDDIMScheduler, CustomDDPMScheduler

SCHEDULER_FUNC = {
    "ddim": CustomDDIMScheduler,
    "ddpm": CustomDDPMScheduler,
}


def tensor_dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode="reflect")
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def tensor_erode(bin_img, ksize=5):
    out = 1 - tensor_dilate(1 - bin_img, ksize)
    return out


class TextTranslationDiffusion:
    NUM_TO_COMP = [9, 13, 4, 8, 1, 6, 2, 7]  # order matters
    UNCON_TOKEN = "no or light makeup"

    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.eta = cfg.EVAL.ETA
        self.transform = T.Compose(
            [
                T.Resize((cfg.TRAIN.IMAGE_SIZE, cfg.TRAIN.IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.img_size = (cfg.TRAIN.IMAGE_SIZE, cfg.TRAIN.IMAGE_SIZE)
        self.sample_steps = cfg.EVAL.SAMPLE_STEPS
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
        self.scheduler: Union[CustomDDIMScheduler, CustomDDPMScheduler] = SCHEDULER_FUNC[
            cfg.EVAL.SCHEDULER
        ](
            num_train_timesteps=cfg.TRAIN.TIME_STEPS,
            prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
            beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
            beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
            beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
        )

    def process_image(self, image: Union[str, torch.Tensor]):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            image = self.transform(image)
            image = image.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image

    def process_mask(
        self,
        mask: Optional[Union[str, torch.Tensor]],
        size: Tuple[int, int],
        return_mask_value: bool = False,
        dilate_eye: bool = True,
    ):
        if mask is None:
            return (None, None) if return_mask_value else None
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        elif isinstance(mask, str):
            mask = Image.open(mask)
        if isinstance(mask, Image.Image):
            mask = np.array(mask.convert("L").resize(size, Image.Resampling.NEAREST))

        if not isinstance(mask, torch.Tensor):
            mask_value = torch.as_tensor(mask, device=self.device).unsqueeze(0)

        empty_mask = torch.zeros(*mask_value.shape, device=self.device, dtype=torch.float32)
        for val in self.NUM_TO_COMP:
            if dilate_eye and val in [1, 6]:
                target_index = (mask_value == val).float()
                keep_index = tensor_dilate(target_index, ksize=5)
                keep_index[target_index > 0] = 0.0
                empty_mask[keep_index > 0] = 1.0
            else:
                empty_mask[mask_value == val] = 1.0
        mask = 1 - empty_mask

        if return_mask_value:
            return mask_value, mask
        return mask

    def process_text(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embedding = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return tokens, embedding

    def sample_xt(
        self, input_: torch.Tensor, timesteps: torch.LongTensor, mask: Optional[torch.Tensor] = None
    ):
        noise = torch.randn_like(input_)
        add_noise_image = self.scheduler.add_noise(input_, noise, timesteps)
        if mask is not None:
            add_noise_image = mask * add_noise_image + (1 - mask) * input_
        return add_noise_image

    def modify_with_text(
        self,
        model: torch.nn.Module,
        source_label: int,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        original_prompt: Optional[Union[str, List[str]]] = None,
        mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        encode_scale: float = 1.0,
        dilate_eye: bool = True,
        start_from_step: int = -1,
    ):
        image = self.process_image(image)
        source_label = (
            torch.nn.functional.one_hot(torch.LongTensor([source_label]), 2).to(self.device).float()
        )

        height, width = image.shape[-2:]
        mask = self.process_mask(
            mask,
            (width, height),
            dilate_eye=dilate_eye,
        )

        with torch.no_grad():
            eps_list = self.encode(
                image,
                model,
                source_label=source_label,
                start_from_step=start_from_step,
                encode_scale=encode_scale,
                original_prompt=original_prompt,
            )
            modified_image = self.generate(
                model,
                source_label,
                prompt,
                eps_list,
                guidance_scale=guidance_scale,
                start_from_step=start_from_step,
                original_image=image,
                mask=mask,
            )

        del eps_list, image, mask
        torch.cuda.empty_cache()
        return modified_image

    @torch.no_grad()
    def encode(
        self,
        image: torch.Tensor,
        model: torch.nn.Module,
        source_label: torch.Tensor,
        start_from_step: int = -1,
        encode_scale: float = 1.0,
        original_prompt: Optional[Union[str, List[str]]] = None,
    ):
        encode_prompt = original_prompt or [self.UNCON_TOKEN] * image.shape[0]
        tokens_unconditional = (
            self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.repeat(image.shape[0], 1, 1)
            .to(self.device)
        )
        tokens_conditional = (
            self.tokenizer(
                encode_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.repeat(image.shape[0], 1, 1)
            .to(self.device)
        )

        initial_noise_steps = start_from_step if start_from_step != -1 else self.sample_steps
        x_T = self.sample_xt(image, torch.LongTensor([initial_noise_steps - 1]).to(self.device))
        return_latent_list = [x_T]
        noise_output = x_T
        start_from_step = -start_from_step if start_from_step != -1 else 0

        self.scheduler.set_timesteps(self.sample_steps, device=self.device)
        for timestep in tqdm(self.scheduler.timesteps[start_from_step:-1], desc="encode"):
            self.scheduler.config.prediction_type = "sample"
            noise_output_next = self.scheduler.step(
                image, timestep, noise_output, eta=self.eta
            ).prev_sample

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output_uncond = model(
                    noise_output,
                    timestep,
                    text=tokens_unconditional,
                    class_labels=source_label,
                )
                model_output_text = model(
                    noise_output,
                    timestep,
                    text=tokens_conditional,
                    class_labels=source_label,
                )
            model_output = model_output_uncond + encode_scale * (
                model_output_text - model_output_uncond
            )
            self.scheduler.config.prediction_type = "epsilon"
            eps = self.scheduler.compute_eps(
                model_output, timestep, noise_output, noise_output_next, eta=self.eta
            )
            return_latent_list.append(eps)
            noise_output = noise_output_next
        return torch.stack(return_latent_list, dim=1)

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        source_label: torch.Tensor,
        prompt: List[str],
        eps_list: Optional[List[torch.Tensor]] = None,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        original_image: Optional[torch.Tensor] = None,
        start_from_step: int = -1,
        mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ):
        assert len(prompt) == num_images

        tokens_unconditional = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        tokens_conditional = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        noise_output = torch.randn(
            (num_images, 3, *self.img_size),
            device=self.device,
        )
        if eps_list is not None:
            assert eps_list.shape[1] == (
                self.sample_steps if start_from_step == -1 else start_from_step
            )
            noise_output = eps_list[:, 0]
            eps_list = eps_list[:, 1:]
        self.scheduler.set_timesteps(self.sample_steps, device=self.device)
        start_from_step = -start_from_step if start_from_step != -1 else 0
        # StableDiffusion has `step_offset` 1
        for it, timestep in enumerate(
            tqdm(self.scheduler.timesteps[start_from_step:], desc="generate")
        ):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output_uncond = model(
                    noise_output,
                    timestep,
                    text=tokens_unconditional,
                    class_labels=source_label,
                )
                model_output_text = model(
                    noise_output,
                    timestep,
                    text=tokens_conditional,
                    class_labels=source_label,
                )
            model_output = model_output_uncond + guidance_scale * (
                model_output_text - model_output_uncond
            )
            noise_output = self.scheduler.step(
                model_output,
                timestep,
                noise_output,
                variance_noise=(
                    eps_list[:, it] if ((eps_list is not None) and timestep > 0) else None
                ),
                eta=self.eta,
                original_image=original_image,
                mask=mask,
            ).prev_sample

        noise_output = noise_output.clamp(-1, 1)
        if normalize:
            return (noise_output / 2 + 0.5).cpu()
        return noise_output
