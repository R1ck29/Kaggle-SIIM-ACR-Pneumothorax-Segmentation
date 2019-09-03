#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import warnings

import cv2
import pandas as pd
import segmentation_models_pytorch as smp
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm

from lovasz_losses import *
from utils import run_length_encode
from siim_dataset import TestDataset

from pathlib import Path
import json

warnings.filterwarnings("ignore")


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything()

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', default=current_dir + '/input/', type=str, help='path to input csv data')
parser.add_argument('--test_image_path', default='/home/rick/siim_data/1024/test_png', type=str, help='path to image data')
parser.add_argument('--stage2_path', default='', type=str, help='path to stage 2 image data')
parser.add_argument('--output_path', default=current_dir + '/output/', type=str, help='path to output csv')
parser.add_argument('--log_path', default=current_dir + '/logs/', type=str, help='path to log files')
parser.add_argument('--weights_path', default=current_dir + '/weights/', type=str, help='path to weights')

parser.add_argument('--fold', default=0, type=int, help='Fold Number')
parser.add_argument('--version', type=int, help='Model Version Number')
parser.add_argument('--encoder_name', default='resnet34', type=str, help='Encoder Name')
parser.add_argument('--img_size_target', default=512, type=int)
parser.add_argument('--test_batch_size', default=16, type=int)
parser.add_argument('--snapshot', default=2, type=int, help='Number of snapshots per fold')
parser.add_argument('--best_score_weight', default=False, type=bool, help='predict with best score weight')


parser.add_argument('--start_snap', default=0, type=int)
parser.add_argument('--end_snap', default=1, type=int)

args = parser.parse_args()

if args.best_score_weight:
    basic_name = f'best_dice_Unet_{args.encoder_name}_v{args.version}_fold{args.fold}'
else:
    basic_name = f'Unet_{args.encoder_name}_v{args.version}_fold{args.fold}'

save_model_name = basic_name + '.pth'
submission_file = basic_name + '.csv'

print(save_model_name)
print(submission_file)

if not os.path.exists(args.csv_path):
    os.makedirs(args.csv_path)

if not os.path.exists(args.output_path + 'v' + str(args.version)):
    os.makedirs(args.output_path + 'v' + str(args.version))

if not os.path.exists(args.weights_path + 'v' + str(args.version)):
    os.makedirs(args.weights_path + 'v' + str(args.version))

if not os.path.exists(args.log_path + 'v' + str(args.version)):
    os.makedirs(args.log_path + 'v' + str(args.version))


args.output_path_version = args.output_path + 'v' + str(args.version)
params_path = Path(args.output_path_version + '/' + f'params_predict_v{args.version}_fold{args.fold}.json')
params_path.write_text(json.dumps(vars(args), indent=4, sort_keys=True))


if args.stage2_path == '':
    test_data_folder = args.test_image_path
    sample_submission_path = args.csv_path + 'sample_submission.csv'
else:
    sample_submission_path = args.stage2_path + 'sample_submission.csv'
    test_data_folder = args.stage2_path + '/test_png'


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def inference(model):
    size = args.img_size_target
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_workers = 8
    best_threshold = 0.5
    min_size = 3500
    device = torch.device("cuda:0")
    df = pd.read_csv(sample_submission_path)

    # get the model from model_trainer object
    # overall_pred = []
    # for step in range(args.start_snap, args.end_snap + 1):
    #     print('Predicting Snapshot', step)
    #     pred_null = []
    #     pred_flip = []
    testset = DataLoader(
        TestDataset(test_data_folder, df, size, mean, std),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.to(device)
    model.eval()
    model_path = args.weights_path + 'v' + str(args.version) + '/' + save_model_name
    print('Loading...', model_path)
    state = torch.load(model_path,
                       map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])  # ["state_dict"]

    encoded_pixels = []
    pred_flip = []
    for i, batch in enumerate(tqdm(testset)):
        preds = torch.sigmoid(model(batch.to(device)))
        preds = preds.detach().cpu().numpy()[:, 0, :, :]  # (batch_size, 1, size, size) -> (batch_size, size, size)
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(probability, best_threshold, min_size)
            # try:
            #     np.save(args.output_path + 'v' + str(args.version) + '/' + basic_name, predict)
            # except:
            #     pass
            # for idx in range(len(predict)):
            # predict = cv2.flip(predict, 1)
            # pred_flip.append(predict)
            #
            # predict = (predict + pred_flip) / 2

            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    #     for i, batch in enumerate(tqdm(testset)):
    #         preds = torch.sigmoid(model(batch.to(device)))
    #         preds = preds.detach().cpu().numpy()[:, 0, :, :]  # (batch_size, 1, size, size) -> (batch_size, size, size)
    #         for probability in preds:
    #             if probability.shape != (1024, 1024):
    #                 probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    #             predict, num_predict = post_process(probability, best_threshold, min_size)
    #             overall_pred.append(predict)
    # overall_pred /= (args.end_snap - args.start_snap + 1)

    # for i in enumerate(tqdm(testset)):
    #     if num_predict == 0:
    #         encoded_pixels.append('-1')
    #     else:
    #         r = run_length_encode(predict)
    #         encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv(args.output_path + 'v' + str(args.version) + '/' + submission_file, columns=['ImageId', 'EncodedPixels'],
              index=False)


if __name__ == '__main__':

    print('Current Device : ', torch.cuda.current_device())
    print('Is Available : ', torch.cuda.is_available())
    # scheduler_step = args.epochs // args.snapshot
    model = smp.Unet(args.encoder_name, encoder_weights="imagenet", activation=None)

    inference(model)
