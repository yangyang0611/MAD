from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from PIL import Image
from tqdm import tqdm

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

    def __init__(self, img_size, scheduler, device, model_path=None, sample_steps=1000):
        # Finetuned from "runwayml/stable-diffusion-v1-5"
        if model_path is not None:
            unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(f"{model_path}/unet")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", unet=unet
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.scheduler = SCHEDULER_FUNC[scheduler].from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)
        pipe.safety_checker = None
        self.pipe: StableDiffusionPipeline = pipe
        self.device = device
        self.img_size = img_size
        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.sample_steps = sample_steps

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
        tokens = self.pipe.tokenizer(
            text,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embedding = self.pipe.text_encoder(tokens.input_ids.to(self.device))[0]
        return tokens, embedding

    def sample_xt(
        self, input_: torch.Tensor, timesteps: torch.LongTensor, mask: Optional[torch.Tensor] = None
    ):
        noise = torch.randn_like(input_)
        add_noise_image = self.pipe.scheduler.add_noise(input_, noise, timesteps)
        if mask is not None:
            add_noise_image = mask * add_noise_image + (1 - mask) * input_
        return add_noise_image

    def modify_with_text(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        original_prompt: Optional[Union[str, List[str]]] = None,
        mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        encode_scale: float = 1.0,
        eta: float = 0.1,
        dilate_eye: bool = True,
        start_from_step: int = -1,
    ):
        image = self.process_image(image)
        height, width = image.shape[-2:]
        mask = self.process_mask(
            mask,
            (width // self.pipe.vae_scale_factor, height // self.pipe.vae_scale_factor),
            dilate_eye=dilate_eye,
        )

        with torch.no_grad():
            eps_list = self.encode(
                image,
                eta,
                start_from_step=start_from_step,
                encode_scale=encode_scale,
                original_prompt=original_prompt,
            )
            modified_image = self.generate(
                prompt,
                eps_list,
                guidance_scale=guidance_scale,
                eta=eta,
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
        eta: float,
        start_from_step: int = -1,
        encode_scale: float = 1.0,
        original_prompt: Optional[Union[str, List[str]]] = None,
    ):
        encode_prompt = original_prompt or [self.UNCON_TOKEN] * image.shape[0]
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            encode_prompt, self.device, 1, True
        )
        text_embedding = torch.cat([negative_prompt_embeds, prompt_embeds])
        image = self.pipe.vae.encode(image.to(text_embedding.dtype)).latent_dist.sample()
        image = image * self.pipe.vae.config.scaling_factor

        initial_noise_steps = start_from_step if start_from_step != -1 else self.sample_steps
        x_T = self.sample_xt(image, torch.LongTensor([initial_noise_steps - 1]).to(self.device))
        return_latent_list = [x_T]
        noise_output = x_T
        start_from_step = -start_from_step if start_from_step != -1 else 0

        # Stable diffusion has step_offset 1
        self.pipe.scheduler.set_timesteps(self.sample_steps, device=self.device)
        for timestep in tqdm(self.pipe.scheduler.timesteps[start_from_step:-1], desc="encode"):
            self.pipe.scheduler.config.prediction_type = "sample"
            noise_output_next = self.pipe.scheduler.step(
                image, timestep - 1, noise_output, eta=eta
            ).prev_sample

            latent_model_input = torch.cat([noise_output] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, timestep - 1
            )
            model_output = self.pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embedding,
                return_dict=False,
            )[0]
            model_output_uncond, model_output_text = model_output.chunk(2)
            model_output = model_output_uncond + encode_scale * (
                model_output_text - model_output_uncond
            )
            self.pipe.scheduler.config.prediction_type = "epsilon"
            eps = self.pipe.scheduler.compute_eps(
                model_output, timestep - 1, noise_output, noise_output_next, eta=eta
            )
            return_latent_list.append(eps)
            noise_output = noise_output_next
        return torch.stack(return_latent_list, dim=1)

    @torch.no_grad()
    def generate(
        self,
        prompt: List[str],
        eps_list: Optional[List[torch.Tensor]] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        num_images: int = 1,
        original_image: Optional[torch.Tensor] = None,
        start_from_step: int = -1,
        mask: Optional[torch.Tensor] = None,
    ):
        if original_image is not None:
            original_image = self.pipe.vae.encode(
                original_image.to(eps_list.dtype)
            ).latent_dist.sample()
            original_image = original_image * self.pipe.vae.config.scaling_factor

        assert len(prompt) == num_images

        tokens_unconditional = self.pipe.tokenizer(
            "",
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embedding_unconditional = self.pipe.text_encoder(
            tokens_unconditional.input_ids.to(self.device)
        )[0]
        tokens_conditional = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embedding_conditional = self.pipe.text_encoder(
            tokens_conditional.input_ids.to(self.device)
        )[0]

        noise_output = (
            eps_list[:, 0]
            if eps_list is not None
            else torch.randn(
                (
                    num_images,
                    self.pipe.unet.config.in_channels,
                    self.img_size // self.pipe.vae_scale_factor,
                    self.img_size // self.pipe.vae_scale_factor,
                ),
                device=self.device,
                dtype=embedding_conditional.dtype,
            )
        )
        if eps_list is not None:
            assert eps_list.shape[1] == (
                self.sample_steps if start_from_step == -1 else start_from_step
            )
            noise_output = eps_list[:, 0]
            eps_list = eps_list[:, 1:]
        self.pipe.scheduler.set_timesteps(self.sample_steps, device=self.device)
        start_from_step = -start_from_step if start_from_step != -1 else 0
        # StableDiffusion has `step_offset` 1
        for it, timestep in enumerate(
            tqdm(self.pipe.scheduler.timesteps[start_from_step:], desc="generate")
        ):
            latent_model_input = self.pipe.scheduler.scale_model_input(noise_output, timestep - 1)
            model_output_uncond = self.pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=embedding_unconditional,
                return_dict=False,
            )[0]
            model_output_text = self.pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=embedding_conditional,
                return_dict=False,
            )[0]
            model_output = model_output_uncond + guidance_scale * (
                model_output_text - model_output_uncond
            )
            noise_output = self.pipe.scheduler.step(
                model_output,
                timestep - 1,
                noise_output,
                variance_noise=(
                    eps_list[:, it] if ((eps_list is not None) and (timestep - 1) > 0) else None
                ),
                eta=eta,
                original_image=original_image,
                mask=mask,
            ).prev_sample

        image = self.pipe.vae.decode(
            noise_output / self.pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        do_denormalize = [True] * image.shape[0]
        image = self.pipe.image_processor.postprocess(
            image, output_type="pil", do_denormalize=do_denormalize
        )
        return image
