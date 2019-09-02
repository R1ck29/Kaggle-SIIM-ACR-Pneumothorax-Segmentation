#!/usr/bin/env python
# coding: utf-8

# This kernel produces a simple average of the three best public submissions. 
# 
# It is possible to add new submission files and change the threshold. The latter is expressed in the form of "minimum number of times that a pixel has been counted as positive in order to be included in the final prediction". 
# 
# At the moment, the final average solution does not produce a better score than the individual best submission.

# In[1]:


import numpy as np
import pandas as pd
import os
from glob import glob
import sys
import skimage.measure
from tqdm import tqdm
import argparse
import requests
from mask_functions import rle2mask, mask2rle
from utils import send_line_notification


current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', default=current_dir + '/output/', type=str, help='path to output csv')

parser.add_argument('--fold', default=0, type=int, help='Fold Number')
parser.add_argument('--version', default=1, type=int, help='Model Version Number')
parser.add_argument('--folder', default='', type=str, help='Folder which contains CSV files')
parser.add_argument('--encoder_name', default='resnet34', type=str, help='Encoder Name')
parser.add_argument('--best_score_weight', default=False, type=bool, help='predict with best score weight')

args = parser.parse_args()


def notify_results():
    competition_name = 'SIIM\n'
    model_name = submission_file
    comment = 'version={}'.format(args.version)
    message = competition_name + '\n' + model_name + '\n' + comment + '\n'
    send_line_notification(message=message)

    print(message)


if args.folder=='':
    if not os.path.exists(args.output_path + "v" + str(args.version) + "/" + "ensemble/"):
        os.makedirs(args.output_path + 'v' + str(args.version) + '/' + 'ensemble/')
else:
    if not os.path.exists(args.output_path + args.folder + "/" + "ensemble/"):
        os.makedirs(args.output_path + args.folder + '/' + 'ensemble/')

version = "v" + str(args.version)

if args.folder=='':
    if args.best_score_weight:
        csv_list = glob(os.path.join(args.output_path, version, "best_dice_*.csv"))
    else:
        csv_list = glob(os.path.join(args.output_path , version, "*.csv"))
else:
    if args.best_score_weight:
        csv_list = glob(os.path.join(args.output_path, args.folder, "best_dice_*.csv"))
    else:
        csv_list = glob(os.path.join(args.output_path, args.folder, "*.csv"))

# read all submissions into daframes and store them in a list
df_sub_list = [pd.read_csv(f) for f in csv_list]

print('Target CSV Files: ', csv_list, '\n', len(df_sub_list))


# create a list of unique image IDs
iid_list = df_sub_list[0]["ImageId"].unique()
print(f"{len(iid_list)} unique image IDs.")


# Create average prediction mask for each image

# set here the threshold for the final mask
# min_solutions is the minimum number of times that a pixel has to be positive in order to be included in the final mask
min_solutions = len(df_sub_list) # a number between 1 and the number of submission files
assert (min_solutions >= 1 and min_solutions <= len(df_sub_list)),     "min_solutions has to be a number between 1 and the number of submission files"


# create empty final dataframe
df_avg_sub = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
df_avg_sub_idx = 0 # counter for the index of the final dataframe

# iterate over image IDs
for iid in tqdm(iid_list):
    # initialize prediction mask
    avg_mask = np.zeros((1024,1024))
    # iterate over prediction dataframes
    for df_sub in df_sub_list:
        # extract rles for each image ID and submission dataframe
        rles = df_sub.loc[df_sub["ImageId"]==iid, "EncodedPixels"]
        # iterate over rles
        for rle in rles:
            # if rle is not -1, build prediction mask and add to average mask
            if "-1" not in str(rle):
                avg_mask += rle2mask(rle, 1024, 1024) / float(len(df_sub_list))
    # threshold the average mask
    avg_mask = (avg_mask >= (min_solutions * 255. / float(len(df_sub_list)))).astype("uint8")
    # extract rles from the average mask
    avg_rle_list = []
    if avg_mask.max() > 0:
        # label regions
        labeled_avg_mask, n_labels = skimage.measure.label(avg_mask, return_num=True)
        # iterate over regions, extract rle, and save to a list
        for label in range(1, n_labels+1):
            avg_rle = mask2rle((255 * (labeled_avg_mask == label)).astype("uint8"), 1024, 1024)
            avg_rle_list.append(avg_rle)
    else:
        avg_rle_list.append("-1")
    # iterate over average rles and create a row in the final dataframe
    for avg_rle in avg_rle_list:
        df_avg_sub.loc[df_avg_sub_idx] = [iid, avg_rle]
        df_avg_sub_idx += 1 # increment index


df_avg_sub["ImageId"].nunique()

if args.best_score_weight:
    submission_file = 'best_dice_Unet_' + args.encoder_name +'_v' + str(args.version) + '_average.csv'
else:
    submission_file = 'Unet_' + args.encoder_name +'_v' + str(args.version) + '_average.csv'

if args.folder=='':
    df_avg_sub.to_csv(args.output_path + version + '/' + 'ensemble/' + submission_file, index=False)
else:
    df_avg_sub.to_csv(args.output_path + args.folder + '/' + 'ensemble/' + submission_file, index=False)

try:
    notify_results()
except:
    pass
