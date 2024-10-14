import argparse
import glob
import os

import cv2
import skimage.metrics
import tabulate
from tqdm import tqdm

ROOT = {
    "mt": "data/mtdataset/images",
}


def main(args):
    table = []
    for target in tqdm(args.path_list):
        if not os.path.exists(target):
            raise FileNotFoundError(f"{target} not found")
        target_file_sorted = sorted(
            list(glob.glob(f"{target}/pred_*.png")),
            key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]),
        )
        target_non_makeup_img = []
        with open(args.origin_file, "r") as f:
            for line in f:
                if line.strip() != "":
                    target_non_makeup_img.append(os.path.join(ROOT["mt"], line.strip()))

        total_ssim = 0
        total_psnr = 0
        for file in target_file_sorted:
            idx = int(os.path.basename(file).split(".")[0].rsplit("_", 1)[1])
            pred = cv2.imread(file)
            non_makeup = cv2.imread(target_non_makeup_img[idx])
            pred = cv2.resize(pred, (non_makeup.shape[1], non_makeup.shape[0]))
            ssim = skimage.metrics.structural_similarity(
                cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(non_makeup, cv2.COLOR_BGR2GRAY),
            )
            psnr = skimage.metrics.peak_signal_noise_ratio(
                cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(non_makeup, cv2.COLOR_BGR2GRAY),
            )
            total_ssim += ssim
            total_psnr += psnr
        table.append(
            [
                target,
                total_ssim / len(target_file_sorted),
                total_psnr / len(target_file_sorted),
            ]
        )
    print(tabulate.tabulate(table, headers=["Appraoch", "SSIM", "PSNR"], tablefmt="fancy_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KID")
    parser.add_argument(
        "--origin-file",
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
    args = parser.parse_args()
    main(args)
