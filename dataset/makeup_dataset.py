import glob
import os.path as osp

import torch
from PIL import Image
from sklearn.model_selection import train_test_split


class ImageDataset:
    def __init__(
        self,
        root_list,
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
        img = Image.open(self.img_list[idx]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        if self.use_label:
            folder_name = osp.basename(osp.dirname(self.img_list[idx]))
            label = self.label_to_idx[folder_name]
            return img, torch.nn.functional.one_hot(torch.LongTensor([label]), 2)
        return img


def get_makeup_dataset(root_list, use_label, train, makeup, transforms):
    return ImageDataset(
        root_list=root_list, use_label=use_label, train=train, makeup=makeup, transforms=transforms
    )


def get_makeup_loader(cfg, train, transforms):
    dataset = get_makeup_dataset(
        cfg.TRAIN.ROOT,
        train=train,
        use_label=cfg.MODEL.LABEL_DIM > 0,
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
