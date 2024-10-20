import copy
import warnings
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import dlib
import mediapipe as mp
import numpy as np
import skimage
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.exposure import match_histograms
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from transformers import CLIPTokenizer

from misc.constant import (
    DLIB_LANDMARKS,
    MEDIAPIPE_LANDMARKS,
    MEDIAPIPE_OVAL,
    NUM_TO_EYE_CONTOUR,
)
from misc.morphing import morph_seq

from .scheduler import CustomDDIMScheduler, CustomDDPMScheduler

SCHEDULER_FUNC = {
    "ddim": CustomDDIMScheduler,
    "ddpm": CustomDDPMScheduler,
}


COMP_INFO = namedtuple("COMP_INFO", "scale, step")

DEFAULT_COMP_SETUP = {
    1: COMP_INFO(scale=0.8, step=100),
    6: COMP_INFO(scale=0.8, step=100),
    9: COMP_INFO(scale=0.8, step=80),
    13: COMP_INFO(scale=0.8, step=80),
    10: COMP_INFO(scale=0.35, step=180),
    4: COMP_INFO(scale=0.4, step=180),
    8: COMP_INFO(scale=0.4, step=180),
    2: COMP_INFO(scale=0.8, step=80),
    7: COMP_INFO(scale=0.8, step=80),
}
MAX_STEP = max([comp_info.step for comp_info in DEFAULT_COMP_SETUP.values()])


COMP_PRIORITY = {4: 1, 8: 1, 10: 1, 9: 2, 13: 2, 2: 3, 7: 3, 1: 4, 6: 4}


def tensor_dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode="reflect")
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def tensor_erode(bin_img, ksize=5):
    out = 1 - tensor_dilate(1 - bin_img, ksize)
    return out


def component_histogram_matching(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    component_list: List[int],
    scale_list: List[float],
    dilate_kernel: int = 21,
):
    return_source_image = copy.deepcopy(source_image)
    hist_image = copy.deepcopy(source_image)
    for scale, comp in zip(scale_list, component_list):
        source_mask_comp = source_mask == comp
        if comp == 10:
            target_mask_comp = target_mask == 4
        else:
            target_mask_comp = target_mask == comp
        if comp in [1, 6]:
            source_mask_comp_new = cv2.dilate(
                source_mask_comp.astype("float"),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilate_kernel, dilate_kernel),
                ),
            )
            target_mask_comp_new = cv2.dilate(
                target_mask_comp.astype("float"),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilate_kernel, dilate_kernel),
                ),
            )
            source_mask_comp_new[source_mask_comp] = 0
            target_mask_comp_new[target_mask_comp] = 0
            source_mask_comp = source_mask_comp_new > 0
            target_mask_comp = target_mask_comp_new > 0
        elif comp in [4, 8]:
            source_mask_comp = cv2.dilate(
                source_mask_comp.astype("float"),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilate_kernel, dilate_kernel),
                ),
            )
            target_mask_comp = cv2.erode(
                target_mask_comp.astype("float"),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilate_kernel, dilate_kernel),
                ),
            )
            source_mask_comp = source_mask_comp > 0
            target_mask_comp = target_mask_comp > 0
            for num in [1, 6]:
                source_eye_mask_comp = source_mask == num
                source_new_eye_mask_comp = cv2.dilate(
                    source_eye_mask_comp.astype("float"),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (21, 21),
                    ),
                )
                source_new_eye_mask_comp[source_eye_mask_comp] = 0
                source_eye_mask_comp = source_new_eye_mask_comp > 0
                source_mask_comp[source_eye_mask_comp] = 0

        if source_mask_comp.sum() == 0 or target_mask_comp.sum() == 0:
            continue
        hist_image[source_mask_comp] = (1 - scale) * source_image[
            source_mask_comp
        ] + scale * match_histograms(
            source_image[source_mask_comp], target_image[target_mask_comp], channel_axis=-1
        )

        if comp not in [1, 6]:
            source_mask_comp = source_mask == comp
        return_source_image[source_mask_comp] = hist_image[source_mask_comp]

    return return_source_image


class TranslationDiffusion:
    BOARD_POINT = np.array(
        [[0, 0], [0, 255], [0, 127], [255, 0], [255, 255], [255, 127], [127, 0], [127, 255]]
    )
    BASIC_COMPONENT = [9, 13, 4, 8, 10, 2, 7, 1, 6]

    def __init__(self, cfg, device):
        self.device = device

        self.cfg = cfg
        self.num_classes = cfg.MODEL.LABEL_DIM
        self.sample_steps = cfg.EVAL.SAMPLE_STEPS
        self.refine_iterations = cfg.EVAL.REFINE_ITERATIONS
        self.refine_steps = cfg.EVAL.REFINE_STEPS
        self.eta = cfg.EVAL.ETA
        self.scheduler: Union[CustomDDIMScheduler, CustomDDPMScheduler] = SCHEDULER_FUNC[
            cfg.EVAL.SCHEDULER
        ](
            num_train_timesteps=cfg.TRAIN.TIME_STEPS,
            prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
            beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
            beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
            beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
        )
        self.transform = T.Compose(
            [
                T.Resize((cfg.TRAIN.IMAGE_SIZE, cfg.TRAIN.IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

        # Use face mesh for obtain a better eyes area and original landmarks for aligning
        self.eyes_landmark = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

        tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
        text_inputs = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        self.null_text = text_inputs.input_ids.unsqueeze(0).to(self.device)

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
        dilate: bool = True,
        component_list: List[int] = None,
        landmarks: Optional[np.ndarray] = None,
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
            mask_value = torch.as_tensor(mask, device=self.device)

        component_list = component_list or self.BASIC_COMPONENT
        component_list.sort(key=lambda x: COMP_PRIORITY[x])

        empty_mask = torch.zeros(*mask_value.shape, device=self.device, dtype=torch.float32)
        for val in component_list:
            if landmarks is not None and val in [1, 6]:
                eye_mask = mask_value == val
                new_eye_mask = torch.as_tensor(
                    skimage.draw.polygon2mask(
                        mask_value.shape, landmarks[NUM_TO_EYE_CONTOUR[val], ::-1]
                    ),
                    device=self.device,
                )
                new_eye_mask[eye_mask | (mask_value == 2) | (mask_value == 7)] = 0
                eye_mask[eye_mask] = 0
                eye_mask[new_eye_mask] = 1
                empty_mask[eye_mask] = 1.0
            elif dilate and val in [1, 6]:
                target_index = (mask_value == val).float()
                expand_index = tensor_dilate(target_index.unsqueeze(0), ksize=21)[0]
                expand_index[target_index > 0] = 0.0
                empty_mask[expand_index > 0] = 1.0
            elif val in [4, 8]:
                empty_mask[mask_value == val] = 1.0
                for num in [1, 6]:
                    eye_mask = (mask_value == num).float()
                    eye_mask = tensor_dilate(eye_mask.unsqueeze(0), ksize=11)[0]
                    empty_mask[eye_mask > 0] = 0.0
            else:
                empty_mask[mask_value == val] = 1.0

        always_hide = ~torch.isin(
            mask_value, torch.as_tensor(self.BASIC_COMPONENT, device=self.device)
        )
        empty_mask[always_hide] = 0.0
        mask = 1 - empty_mask

        if return_mask_value:
            return mask_value, mask.unsqueeze(0)
        return mask.unsqueeze(0)

    def process_label(self, label: Union[int, torch.Tensor], batch_size: int):
        if label is not None and isinstance(label, int):
            label = torch.tensor(label, device=self.device)
            label = torch.nn.functional.one_hot(label, self.num_classes).float()
            label = label.unsqueeze(0).repeat(batch_size, 1)
        return label

    def inverse_diffusion(
        self,
        model: torch.nn.Module,
        input_image: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        inverse_step: int = 300,
    ):
        noise_output = input_image.clone()
        self.scheduler.set_inverse_timesteps(self.sample_steps, device="cuda")
        for timestep in tqdm(self.scheduler.timesteps[:inverse_step], desc="inverse"):
            model_output = model(
                noise_output,
                timestep.reshape(-1),
                text=self.null_text,
                class_labels=class_label,
            )
            noise_output = self.scheduler.ddim_inverse_step(
                model_output, timestep, noise_output
            ).prev_sample
        return noise_output

    def compute_hist_matching(
        self,
        source_image: str,
        target_image: str,
        source_parsing_mask: str,
        target_parsing_mask: str,
        component_list: List[int] = None,
        scale_list: List[int] = None,
    ):
        if isinstance(source_image, str):
            source_image = np.array(Image.open(source_image))
        if isinstance(target_image, str):
            target_image = np.array(Image.open(target_image))

        if not isinstance(source_image, np.ndarray):
            source_image = np.array(source_image)
        if not isinstance(target_image, np.ndarray):
            target_image = np.array(target_image)

        if isinstance(source_parsing_mask, str):
            source_parsing_mask = Image.open(source_parsing_mask).convert("L")
        if isinstance(target_parsing_mask, str):
            target_parsing_mask = Image.open(target_parsing_mask).convert("L")

        source_mask_np = np.array(
            source_parsing_mask.resize(source_image.shape[:2][::-1], Image.Resampling.NEAREST)
        )
        target_mask_np = np.array(
            target_parsing_mask.resize(target_image.shape[:2][::-1], Image.Resampling.NEAREST)
        )
        his_matching_makeup_image = component_histogram_matching(
            source_image,
            target_image,
            source_mask_np,
            target_mask_np,
            component_list=(component_list or self.BASIC_COMPONENT),
            scale_list=scale_list,
        )
        return his_matching_makeup_image

    def morphing(
        self,
        source_img: str,
        target_img: str,
        source_mask: str,
        target_mask: str,
        comp_weight: Optional[Dict[int, float]] = None,
    ):
        def get_mesh_landmarks(img):
            results = self.eyes_landmark.process(img)
            face_landmarks = np.array(
                [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
            )
            face_landmarks = face_landmarks * np.array([256, 256])
            face_landmarks = face_landmarks.astype(np.int32)
            return face_landmarks

        def shape_to_np(shape, dtype="int"):
            coords = np.zeros((68, 2), dtype=dtype)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            return coords

        def get_dlib_landmarks(img):
            rects = self.face_detector(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 0)
            shape = self.shape_predictor(img, rects[0])
            coords = shape_to_np(shape, dtype="int")
            return coords

        source_img = np.array(Image.open(source_img).convert("RGB").resize((256, 256)))
        target_img = np.array(Image.open(target_img).convert("RGB").resize((256, 256)))
        source_mask = np.array(
            (Image.open(source_mask).convert("L").resize((256, 256), Image.Resampling.NEAREST))
        )
        target_mask = np.array(
            (Image.open(target_mask).convert("L").resize((256, 256), Image.Resampling.NEAREST))
        )

        # Expand the hair mask a little bit to avoid the hairline
        hair_mask = target_mask == 12
        hair_mask = cv2.dilate(
            hair_mask.astype("float"),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        target_mask[hair_mask > 0] = 12

        assist_source_landmark = get_mesh_landmarks(source_img)
        assist_target_landmark = get_mesh_landmarks(target_img)

        # Do histogram matching for face first
        HIST_COMP = list(comp_weight.keys())
        HIST_COMP.sort(key=lambda x: COMP_PRIORITY[x])

        pseudo_source_img = component_histogram_matching(
            source_img,
            target_img,
            source_mask,
            target_mask,
            component_list=HIST_COMP,
            scale_list=[comp_weight[comp] for comp in HIST_COMP],
        )
        source_always_hide = ~np.isin(source_mask, self.BASIC_COMPONENT)
        target_always_hide = ~np.isin(target_mask, self.BASIC_COMPONENT)

        if 1 in comp_weight or 6 in comp_weight or 4 in comp_weight or 8 in comp_weight:
            for comp in [1, 6]:
                if 4 not in comp_weight and 8 not in comp_weight and comp not in comp_weight:
                    continue
                source_eye_mask = skimage.draw.polygon2mask(
                    source_mask.shape[:2], assist_source_landmark[NUM_TO_EYE_CONTOUR[comp], ::-1]
                )
                source_eye_mask[
                    (source_mask == comp)
                    | (source_mask == 2)
                    | (source_mask == 7)
                    | source_always_hide
                ] = 0
                source_mask[source_mask == comp] = 0
                source_mask[source_eye_mask] = comp
                target_eye_mask = skimage.draw.polygon2mask(
                    target_mask.shape[:2], assist_target_landmark[NUM_TO_EYE_CONTOUR[comp], ::-1]
                )
                target_eye_mask[
                    (target_mask == comp)
                    | (target_mask == 2)
                    | (target_mask == 7)
                    | target_always_hide
                ] = 0
                target_mask[target_mask == comp] = 0
                target_mask[target_eye_mask] = comp

        try:
            source_landmark = get_dlib_landmarks(source_img)
            target_landmark = get_dlib_landmarks(target_img)
            source_landmark = np.concatenate(
                (
                    source_landmark[DLIB_LANDMARKS],
                    assist_source_landmark[MEDIAPIPE_LANDMARKS],
                ),
                axis=0,
            )
            target_landmark = np.concatenate(
                (
                    target_landmark[DLIB_LANDMARKS],
                    assist_target_landmark[MEDIAPIPE_LANDMARKS],
                ),
                axis=0,
            )
        except IndexError:
            source_landmark = np.delete(assist_source_landmark, MEDIAPIPE_OVAL, axis=0)
            target_landmark = np.delete(assist_target_landmark, MEDIAPIPE_OVAL, axis=0)

        source_landmark = np.concatenate(
            (source_landmark, self.BOARD_POINT),
            axis=0,
        )
        target_landmark = np.concatenate(
            (target_landmark, self.BOARD_POINT),
            axis=0,
        )
        source_landmark = np.minimum(np.maximum(source_landmark, 0), 255)
        target_landmark = np.minimum(np.maximum(target_landmark, 0), 255)

        if 10 in comp_weight:
            comp_weight.pop(10)
        result_img, non_cover_mask = morph_seq(
            target_img,
            source_img,
            target_landmark,
            source_landmark,
            target_mask,
            source_mask,
            comp_weight,
        )
        non_cover_mask[source_mask == 10] = 1
        result_img[non_cover_mask] = pseudo_source_img[non_cover_mask]

        # Smooth the boundary but keep the teeth
        source_always_hide[source_mask == 11] = 0
        source_always_hide = (
            cv2.dilate(
                source_always_hide.astype("float"),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            > 0
        )
        result_img[source_always_hide] = source_img[source_always_hide]
        return result_img, assist_target_landmark, assist_target_landmark

    def copy_paste(
        self,
        source_image: Union[str, torch.Tensor],
        target_image: Union[str, torch.Tensor],
        target_parsing_mask: Optional[Union[str, torch.Tensor]] = None,
    ):
        _, target_parsing_mask = self.process_mask(
            target_parsing_mask,
            target_image.shape[-2:][::-1],
            dilate=True,
            component_list=[9, 13, 1, 6],
            return_mask_value=True,
        )

        paste_image = source_image * target_parsing_mask + target_image * (1 - target_parsing_mask)
        paste_image = gaussian_blur(paste_image, kernel_size=7)
        return paste_image

    def image_translation(
        self,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        source_image: Union[str, torch.Tensor],
        target_image: Union[str, torch.Tensor],
        source_class_label: Optional[Union[int, torch.Tensor]],
        target_class_label: Optional[Union[int, torch.Tensor]],
        inverse_step: int = 0,
        use_encode_eps: bool = True,
        use_his_matching: bool = False,
        use_morphing: bool = True,
        use_copy_paste: bool = False,
        comp_weight_step: Dict[int, Union[COMP_INFO, Tuple[float, int]]] = DEFAULT_COMP_SETUP,
        source_parsing_mask: Optional[Union[str, torch.Tensor]] = None,
        target_parsing_mask: Optional[Union[str, torch.Tensor]] = None,
        use_cam: bool = True,
        inpainting: bool = True,
        return_res: bool = False,
    ):
        assert (
            use_morphing or use_his_matching or use_copy_paste
        ), "At least 'his_matching', 'copy_paste' or 'use_morphing' should be used"

        if use_morphing:
            assert (not use_his_matching) and (
                not use_copy_paste
            ), "Cannot use morphing with others"

        comp_weight_step = {
            comp: COMP_INFO(*comp_info) if isinstance(comp_info, tuple) else comp_info
            for comp, comp_info in comp_weight_step.items()
        }

        source_landmark = None
        if use_his_matching:
            inverse_image = self.compute_hist_matching(
                source_image,
                target_image,
                target_parsing_mask,
                component_list=[1, 6, 9, 13],
            )
            inverse_image = self.process_image(inverse_image)
        elif use_morphing:
            inverse_image, source_landmark, _ = self.morphing(
                source_image,
                target_image,
                source_parsing_mask,
                target_parsing_mask,
                {comp: comp_info.scale for comp, comp_info in comp_weight_step.items()},
            )
            inverse_image = self.process_image(inverse_image)
        source_image = self.process_image(source_image)
        target_image = self.process_image(target_image)
        if use_copy_paste:
            if use_his_matching:
                inverse_image = self.copy_paste(
                    inverse_image, target_image, source_parsing_mask, target_parsing_mask
                )
            else:
                inverse_image = self.copy_paste(
                    source_image, target_image, source_parsing_mask, target_parsing_mask
                )
        if return_res:
            inverse_image = inverse_image.clamp(-1, 1) / 2 + 0.5
            return inverse_image.cpu()
        source_mask_param = {
            "mask": source_parsing_mask,
            "size": source_image.shape[-2:][::-1],
            "landmarks": source_landmark,
        }
        source_class_label = self.process_label(source_class_label, source_image.shape[0])
        target_class_label = self.process_label(target_class_label, target_image.shape[0])
        source_eps_list = None

        start_from_step = -1
        if self.cfg.EVAL.SCHEDULER != "ddim":
            for _, comp_info in comp_weight_step.items():
                start_from_step = max(start_from_step, comp_info.step)

        with torch.no_grad():
            if use_encode_eps:
                source_eps_list = self.encode(
                    source_image, source_class_label, source_model, start_from_step=start_from_step
                )
            else:
                inverse_image = self.sample_xt(
                    inverse_image, torch.LongTensor([start_from_step - 1]).to(self.device)
                )

            if inverse_step > 0:
                inverse_image = self.inverse_diffusion(
                    target_model,
                    inverse_image,
                    target_class_label,
                    inverse_step=inverse_step,
                )

            output = self.generate(
                target_model,
                eps_list=source_eps_list,
                class_label=target_class_label,
                source_image=source_image,
                source_mask_param=source_mask_param if inpainting else None,
                noise_initialize=inverse_image,
                start_from_step=start_from_step,
                comp_step={
                    comp: (comp_info.step if use_cam else MAX_STEP)
                    for comp, comp_info in comp_weight_step.items()
                },
                extra_image=inverse_image,
            )

        del (
            source_eps_list,
            inverse_image,
            source_image,
            target_image,
            source_parsing_mask,
            source_class_label,
            target_class_label,
        )

        torch.cuda.empty_cache()
        return output

    def domain_translation(
        self,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        source_image: Union[str, torch.Tensor],
        source_class_label: Optional[Union[torch.Tensor, int]] = None,
        target_class_label: Optional[Union[torch.Tensor, int]] = None,
        parsing_mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
        use_eps: bool = True,
        use_inversion: bool = False,
        inverse_step: int = 300,
        dilate: bool = True,
        start_from_step: int = -1,
    ):
        assert use_eps or use_inversion, "At least 'eps' or 'inversion' should be used"
        source_image = self.process_image(source_image)
        parsing_mask = self.process_mask(parsing_mask, source_image.shape[-2:][::-1], dilate=dilate)
        source_class_label = self.process_label(source_class_label, source_image.shape[0])
        target_class_label = self.process_label(target_class_label, source_image.shape[0])

        eps_list, inverse_latent = None, None
        with torch.no_grad():
            if use_eps:
                eps_list = self.encode(
                    source_image,
                    source_class_label,
                    source_model,
                    start_from_step=start_from_step,
                )
            if use_inversion:
                inverse_latent = self.inverse_diffusion(
                    target_model,
                    source_image,
                    target_class_label,
                    inverse_step=inverse_step,
                )
            output = self.generate(
                target_model,
                eps_list=eps_list,
                class_label=target_class_label,
                source_image=source_image,
                source_mask=parsing_mask,
                noise_initialize=inverse_latent if use_inversion else None,
                start_from_step=start_from_step,
            )

        if return_latent:
            if use_eps:
                eps_list = eps_list.cpu()
            return eps_list, output

        del (
            eps_list,
            inverse_latent,
            source_image,
            parsing_mask,
            source_class_label,
            target_class_label,
        )
        torch.cuda.empty_cache()
        return output

    def sample_xt(
        self, input_: torch.Tensor, timesteps: torch.LongTensor, mask: Optional[torch.Tensor] = None
    ):
        noise = torch.randn_like(input_)
        add_noise_image = self.scheduler.add_noise(input_, noise, timesteps)
        if mask is not None:
            add_noise_image = mask * add_noise_image + (1 - mask) * input_
        return add_noise_image

    def encode(
        self,
        image: torch.Tensor,
        class_label: Optional[torch.Tensor],
        model: torch.nn.Module,
        start_from_step: int = -1,
        return_noise: bool = False,
    ):
        self.scheduler.set_timesteps(self.sample_steps, device=self.device)
        initial_noise_steps = start_from_step if start_from_step != -1 else self.sample_steps
        x_T = self.sample_xt(image, torch.LongTensor([initial_noise_steps - 1]).to(self.device))
        if return_noise:
            return_noise_list = []
        return_latent_list = [x_T]
        start_from_step = -start_from_step if start_from_step != -1 else 0
        noise_output = x_T
        for timestep in tqdm(
            self.scheduler.timesteps[start_from_step:-1], desc="encode"
        ):  # The final step has no noise inserted
            self.scheduler.config.prediction_type = "sample"  # hack implementation
            noise_output_next = self.scheduler.step(
                image, timestep, noise_output, eta=self.eta
            ).prev_sample
            with torch.autocast("cuda", dtype=torch.float16):
                model_output = model(
                    noise_output,
                    timestep.reshape(-1),
                    text=self.null_text,
                    class_labels=class_label,
                )
            model_output = model_output.float()
            if return_noise:
                return_noise_list.append(model_output)
            self.scheduler.config.prediction_type = "epsilon"  # hack implementation
            eps = self.scheduler.compute_eps(
                model_output,
                timestep,
                noise_output,
                noise_output_next,
                eta=self.eta,
            )
            return_latent_list.append(eps)
            noise_output = noise_output_next

        if return_noise:
            return torch.stack(return_noise_list, dim=1), torch.stack(return_latent_list, dim=1)
        return torch.stack(return_latent_list, dim=1)

    def generate(
        self,
        model: torch.nn.Module,
        eps_list: Optional[torch.Tensor] = None,
        class_label: Optional[torch.Tensor] = None,
        source_image: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        source_mask_param: Optional[Dict[str, Any]] = None,
        noise_initialize: Optional[torch.Tensor] = None,
        start_from_step: int = -1,
        comp_step: Dict[int, int] = None,
        extra_image: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ):
        if source_mask_param is not None and source_mask is not None:
            warnings.warn(
                "source_mask and source_mask_param are both provided, source_mask will be overwrite"
            )
        if eps_list is not None:
            assert eps_list.shape[1] == (
                self.sample_steps if start_from_step == -1 else start_from_step
            )
            if noise_initialize is None:
                noise_output = eps_list[:, 0]
            else:
                noise_output = self.scheduler.add_noise(
                    noise_initialize,
                    torch.randn_like(noise_initialize),
                    (
                        torch.tensor(start_from_step)
                        if start_from_step != -1
                        else torch.tensor(self.sample_steps)
                    ),
                )
            eps_list = eps_list[:, 1:]
        elif noise_initialize is not None:
            noise_output = noise_initialize

        added_component = set()
        self.scheduler.config.prediction_type = "epsilon"
        self.scheduler.set_timesteps(self.sample_steps, device=self.device)
        start_from_step = -start_from_step if start_from_step != -1 else 0
        total_step = len(self.scheduler.timesteps[start_from_step:])
        for it, timestep in enumerate(
            tqdm(self.scheduler.timesteps[start_from_step:], desc="generate")
        ):
            if comp_step is not None:
                changed_component = [
                    comp for comp, step in comp_step.items() if step >= (total_step - it)
                ]
            if (
                source_mask_param is not None
                and comp_step is not None
                and (len(changed_component) != len(added_component))
            ):
                source_mask = self.process_mask(
                    **{
                        **source_mask_param,
                        "component_list": changed_component,
                    }
                )
                for comp in changed_component:
                    if comp not in added_component:
                        added_component.add(comp)
                extra_to_source = (source_mask > 0).unsqueeze(1).repeat(1, 3, 1, 1)
                source_image[extra_to_source] = extra_image[extra_to_source]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = model(
                    noise_output,
                    timestep.reshape(-1),
                    text=self.null_text,
                    class_labels=class_label,
                )
            model_output = model_output.float()
            noise_output = self.scheduler.step(
                model_output,
                timestep,
                noise_output,
                variance_noise=(
                    eps_list[:, it] if ((eps_list is not None) and timestep > 0) else None
                ),
                eta=self.eta,
                original_image=source_image,
                mask=source_mask,
            ).prev_sample

        if self.refine_iterations != 0 and self.refine_steps != 0:
            for _ in tqdm(range(self.refine_iterations), desc="refine"):
                self.scheduler.set_timesteps(1000, device=self.device)
                start_time_step = self.scheduler.timesteps[-self.refine_steps :][0]
                start_time_step = torch.full(
                    (noise_output.shape[0],),
                    start_time_step,
                    dtype=torch.long,
                    device=self.device,
                )

                noise_output = self.sample_xt(noise_output, start_time_step, mask=source_mask)
                for timestep in self.scheduler.timesteps[-self.refine_steps :]:
                    model_output = model(
                        noise_output,
                        timestep.reshape(-1),
                        text=self.null_text,
                        class_labels=class_label,
                    )
                    noise_output = self.scheduler.step(
                        model_output,
                        timestep,
                        noise_output,
                        eta=1.0 if timestep > 0 else 0,
                        original_image=source_image,
                        mask=source_mask,
                    ).prev_sample

        noise_output = noise_output.clamp(-1, 1)
        if normalize:
            return (noise_output / 2 + 0.5).cpu()
        return noise_output
