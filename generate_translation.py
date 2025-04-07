import argparse
import os
from typing import List, Tuple

import torch
from joblib import Parallel, delayed
from loguru import logger
from PIL import Image
from tqdm import tqdm

from config import create_cfg, merge_possible_with_base, show_config
from modeling import build_model
from modeling.translation import TranslationDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--save-folder", default="batch_images", type=str)
    parser.add_argument("--source-root", required=True, type=str)
    parser.add_argument("--source-list", required=True, type=str)
    parser.add_argument("--source-label", required=True, type=int)
    parser.add_argument("--target-label", required=True, type=int)
    parser.add_argument("--num-process", default=1, type=int)
    parser.add_argument("--num-of-step", default=700, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


def generate_image(
    cfg,
    save_folder: str,
    source_list: List[Tuple[str, str]],
    source_label: int,
    target_label: int,
    offset: int,
    device: str,
    num_of_step: int,
):
    torch.cuda.set_device(device)
    model = build_model(cfg).to(device)
    model.eval()

    diffuser = TranslationDiffusion(cfg, device)
    os.makedirs(args.save_folder, exist_ok=True)

    count_error = 0
    with tqdm(
        total=len(source_list), position=int(device.split(":")[-1])
    ) as progress_bar:
        for idx, (source_image, source_mask) in enumerate(source_list):
            save_image_name = os.path.join(save_folder, f"pred_{idx + offset}.png")
            if os.path.exists(save_image_name):
                progress_bar.update(1)
                continue
            if source_mask.endswith("jpg"):
                source_mask = source_mask.replace("jpg", "png")
            try:
                transfer_result = diffuser.domain_translation(
                    source_model=model,
                    target_model=model,
                    source_image=source_image,
                    source_class_label=source_label,
                    target_class_label=target_label,
                    parsing_mask=source_mask,
                    start_from_step=num_of_step,
                )
            except Exception as e:
                logger.error(str(e))
                count_error += 1
                continue
            save_image = Image.fromarray(
                (transfer_result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    "uint8"
                )
            )
            save_image.save(save_image_name)

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

    source_list = []
    with open(args.source_list, "r") as f:
        for line in f.readlines():
            source_line = line.strip()
            image_path = os.path.join(args.source_root, source_line)
            mask_path = image_path.replace("images", "parsing")
            source_list.append((image_path, mask_path))

    task_per_process = len(source_list) // args.num_process
    Parallel(n_jobs=args.num_process)(
        delayed(generate_image)(
            cfg,
            args.save_folder,
            source_list=source_list[
                gpu_idx * task_per_process : (
                    ((gpu_idx + 1) * task_per_process)
                    if gpu_idx < args.num_process - 1
                    else len(source_list)
                )
            ],
            source_label=args.source_label,
            target_label=args.target_label,
            offset=gpu_idx * task_per_process,
            device=f"cuda:{gpu_idx}",
            num_of_step=args.num_of_step,
        )
        for gpu_idx in range(args.num_process)
    )

