#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from albumentations import (
    Compose, Normalize, Resize,
)
from albumentations.torch import ToTensor
from torch.utils.data import DataLoader, Dataset
from augmentation import get_transforms
from utils import run_length_decode


# ## Dataloader

class SIIMDataset(Dataset):
    def __init__(self, df, fnames, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = fnames  # list(self.gb.groups.keys())

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df['EncodedPixels'].tolist()
        image_path = os.path.join(self.root, image_id + ".png")
        image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if annotations[0] != '-1':  #changed
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype('float32')  # for overlap cases
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


def provider(
        fold,
        total_folds,
        data_folder,
        df_path,
        phase,
        size,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
):
    df_all = pd.read_csv(df_path)
    df = df_all.drop_duplicates('ImageId')
    df_with_mask = df[df["EncodedPixels"] != "-1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df["EncodedPixels"] == "-1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask))
    df = pd.concat([df_with_mask, df_without_mask_sampled])
    # NOTE: equal number of positive and negative cases are chosen.

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%
    fnames = df['ImageId'].values
    # image_dataset = SIIMDataset(df, data_folder, size, mean, std, phase)
    image_dataset = SIIMDataset(df_all, fnames, data_folder, size, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


# ## Test prediction
class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])  # sorted(glob.glob(test_data_folder + '/*'))#
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                # HorizontalFlip(p=1),
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples