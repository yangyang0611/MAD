import glob
import json
import os
import random

import torch
from PIL import Image

COMP_TO_TEXT = {
    "face": "skin",
    "eye": "eyes",
}

CHECK_TEXT = {
    "face": "skin",
    "lips": "lip",
    "eyes": "eye",
}


class MTTextDataset(torch.utils.data.Dataset):
    def __init__(self, root, json_file, tokenizer, train=True, transforms=None):
        self.data = (
            list(glob.glob(os.path.join(root, "makeup/*.png")))
            + list(glob.glob(os.path.join(root, "makeup/*.jpg")))
            + list(glob.glob(os.path.join(root, "no_makeup/*.png")))
            + list(glob.glob(os.path.join(root, "no_makeup/*.jpg")))
        )
        self.train = train
        self.transforms = transforms
        self.tokenizer = tokenizer
        with open(json_file, "r") as f:
            self.desp = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img = Image.open(img_name).convert("RGB")
        img = self.transforms(img)

        if os.path.basename(img_name) in self.desp:
            all_desps = self.desp[os.path.basename(img_name)]
            desp_list = []
            for comp_name, desp in all_desps.items():
                out = random.choice(desp).strip().lower()
                if CHECK_TEXT.get(comp_name, comp_name) not in out:
                    out = f"{out} {COMP_TO_TEXT.get(comp_name, comp_name)}"
                desp_list.append(out)
                random.shuffle(desp_list)
            desp = "makeup with " + ", ".join(desp_list)
        else:
            desp = "no or light makeup "

        desp = self.tokenizer(
            desp,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {"pixel_values": img, "input_ids": desp}
