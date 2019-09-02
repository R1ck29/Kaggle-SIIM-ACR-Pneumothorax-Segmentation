#!/usr/bin/env python
# coding: utf-8

import cv2
from albumentations import (
    Compose, HorizontalFlip, Normalize, Resize, ShiftScaleRotate, RandomContrast,
    GaussNoise, RandomBrightness
)
from albumentations.torch import ToTensor


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                # OneOf([
                #     RandomBrightnessContrast(),
                #     RandomGamma(),
                #     RandomBrightness(),
                # ], p=0.5),
                RandomBrightness(p=0.2, limit=0.2),
                RandomContrast(p=0.1, limit=0.2),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            Resize(size, size),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms