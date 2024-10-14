import copy
import glob
import os

import click
import numpy as np
from PIL import Image
from tqdm import tqdm

CONVERT_DICT = {
    12: 9,  # up lip
    1: 4,  # face
    10: 8,  # nose
    14: 10,  # neck
    17: 12,  # hair
    4: 1,  # right eye
    5: 6,  # left eye
    2: 2,  # right eyebrow
    3: 7,  # left eyebrow
    7: 5,  # right ear
    8: 3,  # left ear
    9: 0,  # ear ring
    11: 11,  # teeth
    16: 0,  # shirt
}


@click.command()
@click.option("--original", help="Original json file", type=click.Path(exists=True), required=True)
@click.option("--save_path", help="Original json file", required=True)
def main(original, save_path):
    os.makedirs(save_path, exist_ok=True)
    original_mask = glob.glob(os.path.join(original, "*.png"))

    for mask_name in tqdm(original_mask):
        mask = np.array(Image.open(mask_name))
        new_mask = copy.deepcopy(mask)
        for key, value in CONVERT_DICT.items():
            new_mask[mask == key] = value
        new_mask = Image.fromarray(new_mask)
        new_mask.save(os.path.join(save_path, os.path.basename(mask_name)))


if __name__ == "__main__":
    main()
