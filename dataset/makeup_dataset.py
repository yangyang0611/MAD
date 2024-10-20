import glob
import json
import os
import os.path as osp
import random

import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from transformers import CLIPTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

COMP_TO_TEXT = {
    "face": "skin",
    "eye": "eyes",
}

CHECK_TEXT = {
    "face": "skin",
    "lips": "lip",
    "eyes": "eye",
}


class MakeupDataset:
    def __init__(
        self,
        root_list,
        text_label_path,
        use_label=False,
        train=True,
        makeup=False,
        transforms=None,
    ):
        # Important!!!!!!!!!!
        # Adding beautyface or wild to `root_list` is only allowed for training a removal model for evaluation.
        # The removal model should not be used to generate the result, which may lead to unfair comparison.

        if not isinstance(root_list, list):
            root_list = [root_list]

        self.img_list = []
        self.use_label = use_label
        self.label_to_idx = {}

        with open(text_label_path, "r") as f:
            self.text_label = json.load(f)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )

        if use_label:
            for root in root_list:
                for catego in ["", "makeup", "non-makeup"]:
                    if not osp.exists(osp.join(root, catego)):
                        continue
                    self.img_list.extend(
                        list(glob.glob(osp.join(root, catego, "*.png")))
                        + list(glob.glob(osp.join(root, catego, "*.jpg")))
                    )
            # 0: makeup, 1: non-makeup
            for img in self.img_list:
                folder_name = osp.basename(osp.dirname(img))
                if folder_name == "non-makeup":
                    self.label_to_idx[folder_name] = 1
                else:
                    self.label_to_idx[folder_name] = 0
        else:
            for root in root_list:
                img_dir = "makeup" if makeup else "non-makeup"
                if not osp.exists(osp.join(root, img_dir)):
                    continue
                self.img_list.extend(
                    list(glob.glob(osp.join(root, img_dir, "*.png")))
                    + list(glob.glob(osp.join(root, img_dir, "*.jpg")))
                )
        self.img_list.sort()  # ensure the order is the same on different machines

        # Create label list
        train_split, val_split = train_test_split(self.img_list, test_size=0.1, random_state=42)
        self.img_list = train_split if train else val_split
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(img_name).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        folder_name = osp.basename(osp.dirname(img_name))
        label = self.label_to_idx[folder_name] if self.use_label else None

        if random.random() > 0.7:
            if os.path.basename(img_name) in self.text_label:
                all_desps = self.text_label[os.path.basename(img_name)]
                desp_list = []
                for comp_name, desp in all_desps.items():
                    out = random.choice(desp).strip().lower()
                    if CHECK_TEXT.get(comp_name, comp_name) not in out:
                        out = f"{out} {COMP_TO_TEXT.get(comp_name, comp_name)}"
                    desp_list.append(out)
                    random.shuffle(desp_list)
                desp = "makeup with " + ", ".join(desp_list)
            else:
                desp = "no or light makeup"
        else:
            desp = ""

        text_inputs = self.tokenizer(
            desp,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return {
            "image": img,
            "text": text_input_ids,
            "label": (
                torch.nn.functional.one_hot(torch.LongTensor([label]), len(self.label_to_idx))
                .squeeze(0)
                .float()
                if self.use_label
                else None
            ),
        }


def get_makeup_loader(cfg, train, transforms):
    dataset = MakeupDataset(
        cfg.TRAIN.ROOT,
        cfg.TRAIN.TEXT_LABEL_PATH,
        use_label=cfg.MODEL.LABEL_DIM > 0,
        train=train,
        makeup=cfg.TRAIN.MAKEUP,
        transforms=transforms,
    )
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
