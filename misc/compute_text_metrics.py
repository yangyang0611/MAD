import argparse
import json
import os

import clip
import tabulate
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--orig-root", type=str, required=True)
    parser.add_argument("--test-dir", type=str, nargs="+", required=True)
    return parser.parse_args()


def compute_clip_text(model, preprocess, image_dir, test_meta):
    all_images = []
    all_texts = []
    for idx, val in enumerate(tqdm(list(test_meta.values()))):
        all_images.append(preprocess(Image.open(os.path.join(image_dir, f"pred_{idx}.png"))))
        all_texts.append(f"makeup with {', '.join(val['style'])}")
    image = torch.stack(all_images).to(device)
    text = clip.tokenize(all_texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features).mean()
    return similarity


def compute_image_similarity(model, preprocess, image_dir, orig_dir, test_meta):
    all_ori_images = []
    all_result_images = []
    for idx, val in enumerate(tqdm(list(test_meta.values()))):
        all_ori_images.append(preprocess(Image.open(os.path.join(orig_dir, val["nonmakeup"]))))
        all_result_images.append(preprocess(Image.open(os.path.join(image_dir, f"pred_{idx}.png"))))

    ori_image = torch.stack(all_ori_images).to(device)
    result_image = torch.stack(all_result_images).to(device)

    with torch.no_grad():
        ori_image_features = model.encode_image(ori_image)
        result_image_features = model.encode_image(result_image)

        ori_image_features = ori_image_features / ori_image_features.norm(dim=1, keepdim=True)
        result_image_features = result_image_features / result_image_features.norm(
            dim=1, keepdim=True
        )
    similarity = torch.nn.functional.cosine_similarity(
        ori_image_features, result_image_features
    ).mean()
    return similarity


def compute_style_similarity(model, preprocess, image_dir, orig_dir, test_meta):
    all_ori_images = []
    all_result_images = []
    for idx, key in enumerate(tqdm(list(test_meta.keys()))):
        all_ori_images.append(preprocess(Image.open(os.path.join(orig_dir, key))))
        all_result_images.append(preprocess(Image.open(os.path.join(image_dir, f"pred_{idx}.png"))))

    ori_image = torch.stack(all_ori_images).to(device)
    result_image = torch.stack(all_result_images).to(device)

    with torch.no_grad():
        ori_image_features = model.encode_image(ori_image)
        result_image_features = model.encode_image(result_image)

        ori_image_features = ori_image_features / ori_image_features.norm(dim=1, keepdim=True)
        result_image_features = result_image_features / result_image_features.norm(
            dim=1, keepdim=True
        )
    similarity = torch.nn.functional.cosine_similarity(
        ori_image_features, result_image_features
    ).mean()
    return similarity


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    model, preprocess = clip.load(args.model_path, device=device)

    with open(args.test_file, "r") as f:
        test_meta = json.load(f)

    text_match_score = []
    image_match_score = []
    style_score = []
    for test_dir in args.test_dir:
        text_match_score.append(compute_clip_text(model, preprocess, test_dir, test_meta))
        image_match_score.append(
            compute_image_similarity(model, preprocess, test_dir, args.orig_root, test_meta)
        )
        style_score.append(
            compute_style_similarity(model, preprocess, test_dir, args.orig_root, test_meta)
        )

    print(
        tabulate.tabulate(
            [
                ["CLIP text"] + text_match_score,
                ["CLIP image"] + image_match_score,
                ["CLIp style"] + style_score,
            ],
            headers=["Metric"] + [os.path.basename(dir_name) for dir_name in args.test_dir],
            tablefmt="pretty",
        )
    )
