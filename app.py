import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from config import create_cfg, merge_possible_with_base
from modeling.model import build_model
from modeling.text_translation import TextTranslationDiffusion
from modeling.translation import TranslationDiffusion

SOURCE_TO_NUM = {"makeup": 0, "non-makeup": 1}


def copy_parameters(from_parameters: torch.nn.Parameter, to_parameters: torch.nn.Parameter):
    to_parameters = list(to_parameters)
    assert len(from_parameters) == len(to_parameters)
    for s_param, param in zip(from_parameters, to_parameters):
        param.data.copy_(s_param.to(param.device).data)


def create_diffusion_model(cfg_path: str, diffusion_model_pth: str, device: str):
    cfg = create_cfg()
    merge_possible_with_base(cfg, cfg_path)
    cfg.EVAL.SCHEDULER = "ddpm"
    cfg.EVAL.SAMPLE_STEPS = 1000
    cfg.EVAL.ETA = 0.01
    cfg.EVAL.REFINE_ITERATIONS = 0
    cfg.EVAL.REFINE_STEPS = 10
    model = build_model(cfg)
    weight = torch.load(diffusion_model_pth, map_location="cpu")
    copy_parameters(weight["ema_state_dict"]["shadow_params"], model.parameters())
    del weight
    model = model.to(device)
    model.eval()

    cycle_diffuser = TranslationDiffusion(cfg, device)

    return cycle_diffuser, model


def create_translation_model(sd_model_path: str, device: str):
    stable_cycle_diffuser = TextTranslationDiffusion(
        img_size=512,
        scheduler="ddpm",
        device=device,
        model_path=sd_model_path,
    )

    return stable_cycle_diffuser


def domain_translation(
    config_file: str,
    diffusion_model_path: str,
    device: str,
    source_label: str,
    target_label: str,
    image_input: Image.Image,
    mask_input: Image.Image,
):
    cycle_diffuser, diffusion_model = create_diffusion_model(
        config_file, diffusion_model_path, device
    )
    transform_result = cycle_diffuser.domain_translation(
        source_model=diffusion_model,
        target_model=diffusion_model,
        source_image=image_input,
        source_class_label=SOURCE_TO_NUM[source_label],
        target_class_label=SOURCE_TO_NUM[target_label],
        parsing_mask=mask_input,
        use_inversion=False,
    )

    del cycle_diffuser, diffusion_model
    torch.cuda.empty_cache()
    return Image.fromarray(
        (transform_result[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    )


def reference_translation(
    config_file: str,
    diffusion_model_path: str,
    device: str,
    source_label: str,
    target_label: str,
    source_image_input: Image.Image,
    target_image_input: Image.Image,
    source_mask_input: Image.Image,
    target_mask_input: Image.Image,
):
    cycle_diffuser, diffusion_model = create_diffusion_model(
        config_file, diffusion_model_path, device
    )
    transform_result = cycle_diffuser.image_translation(
        source_model=diffusion_model,
        target_model=diffusion_model,
        source_image=source_image_input,
        target_image=target_image_input,
        source_class_label=SOURCE_TO_NUM[source_label],
        target_class_label=SOURCE_TO_NUM[target_label],
        source_parsing_mask=source_mask_input,
        target_parsing_mask=target_mask_input,
        use_his_matching=False,
    )
    del cycle_diffuser, diffusion_model
    torch.cuda.empty_cache()
    return Image.fromarray(
        (transform_result[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    )


def text_manipulate(
    stable_diffusion_path: str,
    device: str,
    source_image_with_brush_mask: Image.Image,
    source_mask: Image.Image,
    prompt: str,
    guidance_scale: float,
):
    text_translation = create_translation_model(stable_diffusion_path, device)
    source_image = source_image_with_brush_mask["background"].convert("RGB")
    source_brush_mask = source_image_with_brush_mask["layers"][0]

    # Trick for transform all the other component to non-change and leave only the brush
    source_brush_mask = np.array(source_brush_mask)[..., 3]
    if np.sum(source_brush_mask) != 0:
        source_brush_mask[source_brush_mask != 0] = 4  # Pretend it to be 1 and will not be filtered
        contours, _ = cv2.findContours(
            ((source_brush_mask > 0) * 255).astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros((*source_brush_mask.shape, 3), dtype="uint8")
        mask[source_brush_mask != 0] = np.array([255, 205, 235])
        for c in contours:
            cv2.drawContours(mask, [c], -1, (0, 255, 0), 2)

        result = cv2.addWeighted(
            mask, 0.5, cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR), 0.5, 0
        )
        cv2.imwrite("temp.png", result)
        source_brush_mask[source_brush_mask == 0] = 255

        if source_mask is not None:
            source_mask = np.array(source_mask.resize(source_brush_mask.shape[:2][::-1]))
            source_brush_mask[(source_mask == 1) | (source_mask == 6)] = 255

    transform_result = text_translation.modify_with_text(
        image=source_image,
        prompt=[prompt],
        mask=source_brush_mask,
        guidance_scale=guidance_scale,
        start_from_step=180,
    )
    del text_translation
    torch.cuda.empty_cache()
    return source_mask, transform_result[0]


def domain_to_text(domain_image_output, domain_mask_input):
    return domain_image_output, domain_mask_input


def reference_to_text(reference_image_output, reference_mask_input_source):
    return reference_image_output, reference_mask_input_source


with gr.Blocks(title="Makeup Transfer") as demo:
    with gr.Row():
        with gr.Column():
            config_file = gr.Dropdown(
                ["configs/model_256_256.yaml"],
                value="configs/model_256_256.yaml",
                label="Select config file",
            )
            diffusion_model_path = gr.Dropdown(
                ["makeup_checkpoint.pth"],
                value="makeup_checkpoint.pth",
                label="Select diffusion model path",
            )

        with gr.Column():
            stable_diffusion_path = gr.Dropdown(
                ["text_checkpoint.pth"],
                value="text_checkpoint.pth",
                label="Select stable diffusion model path",
            )
            device = gr.Dropdown(["cpu", "cuda"], value="cuda", label="Select Device")

    with gr.Tab("Domain Translation"):
        with gr.Row():
            domain_image_input = gr.Image(type="pil", label="Input Image")
            domain_mask_input = gr.Image(type="pil", label="Input Mask", image_mode="L")
            domain_image_output = gr.Image(type="pil", label="Output Image", interactive=False)
        with gr.Row():
            domain_source_label = gr.Dropdown(["makeup", "non-makeup"], label="Source Label")
            domain_target_label = gr.Dropdown(["makeup", "non-makeup"], label="Target Label")
        domain_to_text_button = gr.Button(value="Send to text manipulation")
        domain_submit = gr.Button(value="Transform")
        with gr.Row():
            gr.Examples(
                examples=["data/mtdataset/images/non-makeup/xfsy_0307.png"],
                inputs=domain_image_input,
                label="Image Example",
            )
            gr.Examples(
                examples=["data/mtdataset/parsing/non-makeup/xfsy_0307.png"],
                inputs=domain_mask_input,
                label="Mask Example (pair with image)",
            )

    with gr.Tab("Reference Translation"):
        with gr.Row():
            reference_image_input_source = gr.Image(type="pil", label="Source Image")
            reference_mask_input_source = gr.Image(type="pil", label="Source Mask", image_mode="L")
        with gr.Row():
            reference_image_input_target = gr.Image(type="pil", label="Reference Image")
            reference_mask_input_target = gr.Image(
                type="pil", label="Reference Mask", image_mode="L"
            )

        reference_image_output = gr.Image(
            type="pil", label="Output Image", interactive=False, width="50%"
        )
        with gr.Row():
            reference_source_label = gr.Dropdown(["makeup", "non-makeup"], label="Source Label")
            reference_target_label = gr.Dropdown(["makeup", "non-makeup"], label="Target Label")
        reference_to_text_button = gr.Button(value="Send to text manipulation")
        reference_submit = gr.Button(value="Transform")
        with gr.Row():
            gr.Examples(
                examples=["data/mtdataset/images/non-makeup/vSYYZ223.png"],
                inputs=reference_image_input_source,
                label="Source Image Example",
            )
            gr.Examples(
                examples=["data/mtdataset/parsing/non-makeup/vSYYZ223.png"],
                inputs=reference_mask_input_source,
                label="Reference Mask Example (pair with image)",
            )
        with gr.Row():
            gr.Examples(
                examples=["data/mtdataset/images/makeup/vFG66.png"],
                inputs=reference_image_input_target,
                label="Reference Image Example",
            )
            gr.Examples(
                examples=["data/mtdataset/parsing/makeup/vFG66.png"],
                inputs=reference_mask_input_target,
                label="Reference Mask Example (pair with image)",
            )

    with gr.Tab("Text Manipulation"):
        with gr.Row():
            text_image_input = gr.ImageMask(type="pil", label="Input Image")
            text_mask_input = gr.Image(type="pil", label="Input Mask", image_mode="L")

        with gr.Row():
            text_brush_mask = gr.Image(
                type="pil", label="Brush Mask", image_mode="L", interactive=False
            )
            text_image_output = gr.Image(type="pil", label="Output Image", interactive=False)

        with gr.Row():
            text_input = gr.Textbox(lines=1, label="Input Text")
            text_guidance_scale = gr.Slider(minimum=0, maximum=30, value=15, label="Guidance Scale")
        text_submit = gr.Button(value="Transform")
        with gr.Row():
            gr.Examples(
                examples=[
                    "data/mtdataset/images/non-makeup/xfsy_0327.png",
                    "data/mtdataset/images/makeup/vFG805.png",
                ],
                inputs=text_image_input,
                label="Image Example",
            )
            gr.Examples(
                examples=[
                    "data/mtdataset/parsing/non-makeup/xfsy_0327.png",
                    "data/mtdataset/parsing/makeup/vFG805.png",
                ],
                inputs=text_mask_input,
                label="Mask Example (pair with image)",
            )
            gr.Examples(
                examples=["makeup with smoky eyeshadow"],
                inputs=text_input,
                label="Text Example",
            )
    domain_submit.click(
        domain_translation,
        [
            config_file,
            diffusion_model_path,
            device,
            domain_source_label,
            domain_target_label,
            domain_image_input,
            domain_mask_input,
        ],
        [domain_image_output],
    )
    domain_to_text_button.click(
        domain_to_text,
        [domain_image_output, domain_mask_input],
        [text_image_input, text_mask_input],
    )

    reference_submit.click(
        reference_translation,
        [
            config_file,
            diffusion_model_path,
            device,
            reference_source_label,
            reference_target_label,
            reference_image_input_source,
            reference_image_input_target,
            reference_mask_input_source,
            reference_mask_input_target,
        ],
        [reference_image_output],
    )
    reference_to_text_button.click(
        reference_to_text,
        [reference_image_output, reference_mask_input_source],
        [text_image_input, text_mask_input],
    )

    text_submit.click(
        text_manipulate,
        [
            stable_diffusion_path,
            device,
            text_image_input,
            text_mask_input,
            text_input,
            text_guidance_scale,
        ],
        [text_brush_mask, text_image_output],
    )

# Launch the demo
demo.queue(max_size=1).launch(share=False, debug=True)
