import os
import random
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob

import numpy as np
import tabulate
import torch
import torch.nn
import torchvision.transforms.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

import torch_fidelity

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = {
    "mt": "data/mtdataset/images",
    "mt_removal": "generate_outputs/mt_removal",
    "beauty": "data/beautyface/images",
    "beauty_removal": "generate_outputs/beauty_removal",
    "wild": "data/wild/images",
    "wild_removal": "generate_outputs/wild_removal",
}


def set_seed(seed: int = 385832):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TransformPILtoRGBTensor:
    def __call__(self, img):
        return F.pil_to_tensor(img)


class ImagesPathDataset(torch.utils.data.Dataset):
    def __init__(self, files, img_size=224):
        self.files = files
        self.transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.CenterCrop((img_size, img_size)),
                TransformPILtoRGBTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")

        img = self.transforms(img)
        return img


class CheckImagesPathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        ori_file=None,
        target_files=None,
        order_files=None,
        img_size=224,
    ):
        self.files = []
        if ori_file is None:
            for file in order_files:
                num = int(os.path.basename(file).split(".")[0].split("_")[-1])
                self.files.append(os.path.join(root, f"pred_{num}.png"))
        else:
            with open(ori_file, "r") as f:
                line_of_file = f.readlines()
            for file in target_files:
                num = int(os.path.basename(file).split(".")[0].split("_")[-1])
                self.files.append(os.path.join(root, line_of_file[num].strip()))
        self.transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.CenterCrop((img_size, img_size)),
                TransformPILtoRGBTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img


def main(args):
    table = []
    target_non_makeup_img = []
    with open(args.non_makeup_file, "r") as f:
        for line in f:
            target_non_makeup_img.append(os.path.join(ROOT["mt"], line.strip()))
    for target in tqdm(args.path_list):
        if not os.path.exists(target):
            continue
        target_file_sorted = sorted(
            list(glob.glob(f"{target}/*.png")),
            key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]),
        )
        selected_non_makeup_image = []
        for file in target_file_sorted:
            num = int(os.path.basename(file).split(".")[0].split("_")[-1])
            selected_non_makeup_image.append(target_non_makeup_img[num])
        precision = torch_fidelity.calculate_metrics(
            input1=ImagesPathDataset(files=target_file_sorted),
            input2=ImagesPathDataset(files=selected_non_makeup_image),
            input3=CheckImagesPathDataset(
                root=ROOT[args.type],
                ori_file=args.makeup_file,
                target_files=target_file_sorted,
            ),
            prc=True,
            device="cuda",
            verbose=False,
            cache=False,
            feature_extractor="vgg16",
            feature_extractor_weights_path="vgg.pth",
        )["precision"]

        recall = torch_fidelity.calculate_metrics(
            input1=ImagesPathDataset(files=target_file_sorted),
            input2=CheckImagesPathDataset(
                root=ROOT[args.type],
                ori_file=args.makeup_file,
                target_files=target_file_sorted,
            ),
            input3=CheckImagesPathDataset(  # Use mt to remove original non-makeup feature
                root=ROOT["mt"],
                ori_file=args.non_makeup_file,
                target_files=target_file_sorted,
            ),
            input4=CheckImagesPathDataset(
                root=ROOT[
                    f"{args.type}_removal"
                ],  # Use makeup to non-makeup image to remove non-makeup feature
                order_files=target_file_sorted,
            ),
            prc=True,
            device="cuda",
            cache=False,
            verbose=False,
            feature_extractor="vgg16",
            feature_extractor_weights_path="vgg.pth",
        )["recall"]

        kid = torch_fidelity.calculate_metrics(
            input1=ImagesPathDataset(files=target_file_sorted),
            input2=CheckImagesPathDataset(
                root=ROOT[args.type],
                ori_file=args.makeup_file,
                target_files=target_file_sorted,
            ),
            input3=CheckImagesPathDataset(  # Use mt to remove original non-makeup feature
                root=ROOT["mt"],
                ori_file=args.non_makeup_file,
                target_files=target_file_sorted,
            ),
            input4=CheckImagesPathDataset(
                root=ROOT[
                    f"{args.type}_removal"
                ],  # Use makeup to non-makeup image to remove non-makeup feature
                order_files=target_file_sorted,
            ),
            kid=True,
            device="cuda",
            cache=False,
            verbose=False,
            feature_extractor="inception-v3-compat",
            feature_extractor_weights_path="inception.pth",
            kid_subset_size=min(1000, len(target_file_sorted)),
        )["kernel_inception_distance_mean"]

        table.append(
            [
                target.split("/")[1],
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{kid:.3f}",
            ]
        )
    print(
        tabulate.tabulate(
            table, headers=["Approach", "Precision", "Recall", "KID"], tablefmt="fancy_grid"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KID")
    parser.add_argument(
        "--non-makeup-file",
        help="File denoting the order of makeup image",
        default="data/nomakeup_test_mt.txt",
    )
    parser.add_argument(
        "--makeup-file",
        help="File denoting the order of makeup image",
        default="data/makeup_test_mt.txt",
    )
    parser.add_argument(
        "--path-list",
        help="List of path for evaluation",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--type",
        help="Type of dataset",
        choices=["mt", "beauty", "wild"],
        default="mt",
    )
    args = parser.parse_args()
    set_seed()
    main(args)
