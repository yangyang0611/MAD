import argparse
import json
import os
from typing import List, Tuple

import torch
from joblib import Parallel, delayed
from loguru import logger
from PIL import Image
from tqdm import tqdm

from config import create_cfg, merge_possible_with_base, show_config
from modeling import build_model
from modeling.text_translation import TextTranslationDiffusion


def copy_parameters(from_parameters, to_parameters):
    to_parameters = list(to_parameters)
    assert len(from_parameters) == len(to_parameters)
    for s_param, param in zip(from_parameters, to_parameters):
        param.data.copy_(s_param.to(param.device).data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--save-folder", default="batch_images", type=str)
    parser.add_argument("--source-root", required=True, type=str)
    parser.add_argument("--source-list", required=True, type=str)
    parser.add_argument("--source-label", required=True, type=int)
    parser.add_argument("--num-process", default=1, type=int)
    parser.add_argument("--num-of-step", default=180, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


def generate_image(
    cfg,
    save_folder: str,
    source_list: List[Tuple[str, str]],
    source_label: int,
    offset: int,
    device: str,
    num_of_step: int,
):
    model = build_model(cfg).to(device)
    if cfg.MODEL.PRETRAINED:
        logger.info(f"Loading pretrained model from {cfg.MODEL.PRETRAINED}")
        weight = torch.load(cfg.MODEL.PRETRAINED, map_location=device)
        copy_parameters(weight["ema_state_dict"]["shadow_params"], model.parameters())
        del weight
        torch.cuda.empty_cache()
    diffuser = TextTranslationDiffusion(cfg, device=device)
    os.makedirs(args.save_folder, exist_ok=True)

    progress_bar = tqdm(total=len(source_list), position=int(device.split(":")[-1]))
    count_error = 0

    for idx, (source_image, source_mask, editing_prompt) in enumerate(source_list):
        save_image_name = os.path.join(save_folder, f"pred_{idx + offset}.png")
        if os.path.exists(save_image_name):
            progress_bar.update(1)
            continue
        if source_mask.endswith("jpg"):
            source_mask = source_mask.replace("jpg", "png")
        try:
            editing_result = diffuser.modify_with_text(
                model=model,
                source_label=source_label,
                image=source_image,
                mask=source_mask,
                prompt=[editing_prompt],
                start_from_step=num_of_step,
                guidance_scale=15,
            )
        except Exception as e:
            logger.error(str(e))
            count_error += 1
            continue
        save_image = Image.fromarray(
            (editing_result[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        )
        save_image.save(save_image_name)
        progress_bar.update(1)
    progress_bar.close()

    if count_error != 0:
        print(f"Error in {device}: {count_error}")


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    if args.config:
        merge_possible_with_base(cfg, args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    show_config(cfg)

    with open(args.source_list, "r") as f:
        data = json.load(f)
    source_list = []
    for info in data.items():
        image_path = os.path.join(args.source_root, info["image"])
        mask_path = image_path.replace("images", "parsing")
        source_list.append((image_path, mask_path, info["style"]))

    task_per_process = len(source_list) // args.num_process
    Parallel(n_jobs=args.num_process)(
        delayed(generate_image)(
            args.img_size,
            args.save_folder,
            source_list=source_list[
                gpu_idx
                * task_per_process : (
                    ((gpu_idx + 1) * task_per_process)
                    if gpu_idx < args.num_process - 1
                    else len(source_list)
                )
            ],
            source_label=args.source_label,
            offset=gpu_idx * task_per_process,
            device=f"cuda:{gpu_idx}",
            num_of_step=args.num_of_step,
            model_path=args.model_path,
            scheduler=args.scheduler,
            sample_steps=args.sample_steps,
        )
        for gpu_idx in range(args.num_process)
    )
