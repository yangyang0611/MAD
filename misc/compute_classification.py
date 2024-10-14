import argparse
import glob
import os

import tabulate
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = {
    "mt": "data/mtdataset/images",
    "mt_removal": "generate_outputs/mt_removal",
    "beauty": "data/beautyface/images",
    "beauty_removal": "generate_outputs/beauty_removal",
    "wild": "data/wild/images",
    "wild_removal": "generate_outputs/wild_removal",
}


class InceptionModel(torch.nn.Module):
    def __init__(self, num_class=1000):
        super(InceptionModel, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE_1(1280)
        self.Mixed_7c = InceptionE_2(2048)
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(299, 299),
            align_corners=False,
        )
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_label(image_root):
    all_images = []
    for root in image_root:
        all_images.extend(
            list(glob.glob(os.path.join(root, "**/*.png"), recursive=True))
            + list(glob.glob(os.path.join(root, "**/*.jpg"), recursive=True))
        )
    label = {name: idx for idx, name in enumerate(sorted(set(all_images)))}
    return label


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dict = get_label(["data/mtdataset/images", "data/beautyface/images", "data/wild/images"])
    print(len(label_dict))
    model = InceptionModel(num_class=len(label_dict)).to(device)
    model.load_state_dict(torch.load("inception.pth", map_location="cpu"))
    model = model.to(device)
    model.eval()

    transform_list = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    table = []
    target_non_makeup_img = []
    with open(args.non_makeup_file, "r") as f:
        for line in f:
            if line.strip() != "":
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

        score = 0
        for img_name, label_name in zip(target_file_sorted, selected_non_makeup_image):
            target_label = label_dict[label_name]

            img = Image.open(img_name).convert("RGB")
            img = transform_list(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
            output = nn.functional.softmax(output, dim=1)[0].cpu().tolist()
            score += output[target_label]
        score /= len(selected_non_makeup_image)
        table.append([target.split("/")[1], f"{score:.3f}"])
    print(tabulate.tabulate(table, headers=["Approach", "Acc"], tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KID")
    parser.add_argument(
        "--non-makeup-file",
        help="File denoting the order of makeup image",
        default="data/nomakeup_test_mt.txt",
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
    main(args)
