#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import glob2
import pydicom
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_path', default='/home/rick/siim_data/siim-original/dicom-images-train/', type=str, help='path to input dcm')
parser.add_argument('--test_path', default='/home/rick/siim_data/siim-original/dicom-images-test/', type=str, help='path to input dcm')
parser.add_argument('--train_out_path', default='/home/rick/siim_data/1024/train_png/', type=str, help='path to output png')
parser.add_argument('--test_out_path', default='/home/rick/siim_data/1024/test_png/', type=str, help='path to output png')
parser.add_argument('--test_only', default=False, type=bool, help='whether only test png to convert')
parser.add_argument('--resize', default='', type=str, help='image size')

args = parser.parse_args()


def convert_images(filename, outdir):
    ds = pydicom.read_file(str(filename))
    img = ds.pixel_array
    if args.resize != '':
        img = cv2.resize(img, (128, 128))
    cv2.imwrite(outdir + filename.split('/')[-1][:-4] + '.png', img)


if not args.test_only:
    if not os.path.exists(args.train_out_path):
        os.makedirs(args.train_out_path)

    if not os.path.exists(args.test_out_path):
        os.makedirs(args.test_out_path)

    train_dcm_list = glob2.glob(os.path.join(args.train_path, '**/*.dcm'))
    test_dcm_list = glob2.glob(os.path.join(args.test_path, '**/*.dcm'))

    res1 = Parallel(n_jobs=8, backend='threading')(delayed(
        convert_images)(i, args.train_out_path) for i in tqdm(train_dcm_list[:], total=len(train_dcm_list)))

    res2 = Parallel(n_jobs=8, backend='threading')(delayed(
        convert_images)(i, args.test_out_path) for i in tqdm(test_dcm_list[:], total=len(test_dcm_list)))

else:
    if not os.path.exists(args.test_out_path):
        os.makedirs(args.test_out_path)

    test_dcm_list = glob2.glob(os.path.join(args.test_path, '**/*.dcm'))

    res2 = Parallel(n_jobs=8, backend='threading')(delayed(
        convert_images)(i, args.test_out_path) for i in tqdm(test_dcm_list[:], total=len(test_dcm_list)))