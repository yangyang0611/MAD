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
    parser.add_argument("--target-root", required=True, type=str)
    parser.add_argument("--source-list", required=True, type=str)
    parser.add_argument("--target-list", required=True, type=str)
    parser.add_argument("--source-label", required=True, type=int)
    parser.add_argument("--target-label", required=True, type=int)
    parser.add_argument("--num-process", default=1, type=int)
    parser.add_argument("--inpainting", action="store_true", default=False)
    parser.add_argument("--cam", action="store_true", default=False)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


def generate_image(
    cfg,
    save_folder: str,
    source_list: List[Tuple[str, str]],
    target_list: List[Tuple[str, str]],
    source_label: int,
    target_label: int,
    offset: int,
    device: str,
    inpainting: bool,
    use_cam: bool,
):
    torch.cuda.set_device(device)
    model = build_model(cfg).to(device)
    if cfg.MODEL.PRETRAINED:
        logger.info(f"Loading pretrained model from {cfg.MODEL.PRETRAINED}")
        weight = torch.load(cfg.MODEL.PRETRAINED, map_location=device)
        copy_parameters(weight["ema_state_dict"]["shadow_params"], model.parameters())
        del weight
        torch.cuda.empty_cache()
    model.eval()

    diffuser = TranslationDiffusion(cfg, device)
    os.makedirs(args.save_folder, exist_ok=True)

    progress_bar = tqdm(total=len(source_list), position=int(device.split(":")[-1]))
    count_error = 0

    with tqdm(total=len(source_list), position=int(device.split(":")[-1])) as progress_bar:
        for idx, ((source_image, source_mask), (target_image, target_mask)) in enumerate(
            zip(source_list, target_list)
        ):
            save_image_name = os.path.join(save_folder, f"pred_{idx + offset}.png")
            if os.path.exists(save_image_name):
                progress_bar.update(1)
                continue
            if source_mask.endswith("jpg"):
                source_mask = source_mask.replace("jpg", "png")
            if target_mask.endswith("jpg"):
                target_mask = target_mask.replace("jpg", "png")

            try:
                transfer_result = diffuser.image_translation(
                    source_model=model,
                    target_model=model,
                    source_image=source_image,
                    target_image=target_image,
                    source_class_label=source_label,
                    target_class_label=target_label,
                    source_parsing_mask=source_mask,
                    target_parsing_mask=target_mask,
                    use_morphing=True,
                    use_encode_eps=True,
                    use_cam=use_cam,
                    inpainting=inpainting,
                )
            except Exception as e:
                logger.error(f"Error in {device}: {e}")
                count_error += 1
                continue
            save_image = Image.fromarray(
                (transfer_result[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
            save_image.save(save_image_name)
            progress_bar.update(1)
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
    target_list = []

    with open(args.source_list, "r") as f_source, open(args.target_list, "r") as f_target:
        source_lines = f_source.readlines()
        target_lines = f_target.readlines()
        for idx in range(len(source_lines)):
            if source_lines[idx].strip() == "" or target_lines[idx].strip() == "":
                continue
            source_path = os.path.join(args.source_root, source_lines[idx].strip())
            source_list.append((source_path, source_path.replace("images", "parsing")))
            target_path = os.path.join(args.target_root, target_lines[idx].strip())
            target_list.append((target_path, target_path.replace("images", "parsing")))

    assert len(source_list) == len(target_list), "Source and target list should have same length"
    task_per_process = len(source_list) // args.num_process
    Parallel(n_jobs=args.num_process)(
        delayed(generate_image)(
            cfg,
            args.save_folder,
            source_list=source_list[
                gpu_idx
                * task_per_process : (
                    ((gpu_idx + 1) * task_per_process)
                    if gpu_idx < args.num_process - 1
                    else len(source_list)
                )
            ],
            target_list=target_list[
                gpu_idx
                * task_per_process : (
                    ((gpu_idx + 1) * task_per_process)
                    if gpu_idx < args.num_process - 1
                    else len(target_list)
                )
            ],
            source_label=args.source_label,
            target_label=args.target_label,
            offset=gpu_idx * task_per_process,
            device=f"cuda:{gpu_idx}",
            inpainting=args.inpainting,
            use_cam=args.cam,
        )
        for gpu_idx in range(args.num_process)
    )
