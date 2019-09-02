#!/usr/bin/env python
# coding: utf-8

# This Kernel uses UNet architecture with ResNet34 encoder, I've used [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library
# which has many inbuilt segmentation architectures.
# This kernel is inspired by [Yury](https://www.kaggle.com/deyury)'s discussion thread [here](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/99440#591985). #
#
# * UNet with imagenet pretrained ResNet34 architecture
# * Training on 512x512 sized images/masks with Standard Augmentations
# * MixedLoss (weighted sum of Focal loss and dice loss)
# * Gradient Accumulution

import argparse
import json
import shutil
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.backends.cudnn as cudnn
import torch.optim as optim

from radam import RAdam
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torchcontrib.optim import SWA
from tqdm import tqdm as tqdm

from lovasz_losses import *
from losses import MixedLoss, weighted_bce, weighted_soft_dice, weighted_lovasz
from siim_dataset import provider
from utils import *
from metrics import *


warnings.filterwarnings("ignore")


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.seed_everything = True


seed_everything()

current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', default=current_dir + '/input/', type=str, help='path to input csv data')
parser.add_argument('--train_image_path', default='/home/rick/siim_data/1024/train_png', type=str, help='path to image data')
parser.add_argument('--output_path', default=current_dir + '/output/', type=str, help='path to output csv')
parser.add_argument('--log_path', default=current_dir + '/logs/', type=str, help='path to log files')
parser.add_argument('--weights_path', default=current_dir + '/weights/', type=str, help='path to weights')

parser.add_argument('--fold', default=0, type=int, help='Fold Number')
parser.add_argument('--version', type=int, help='Model Version Number')
parser.add_argument('--encoder_name', default='resnet34', type=str, help='Encoder Name')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--img_size_target', default=512, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate')
parser.add_argument('--optimizer', default='radam', type=str, help='Optimizer')
parser.add_argument('--scheduler', default='reducelronplateau', type=str, help='Scheduler')
parser.add_argument('--patience', default=3, type=int, help='patience for reducelronplateau')
parser.add_argument('--swa', default=False, type=bool, help='SWA')
parser.add_argument('--snapshot', default=2, type=int, help='Number of snapshots per fold')
parser.add_argument('--max_lr', default=1e-3, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=1e-4, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')

parser.add_argument('--start_snap', default=0, type=int)
parser.add_argument('--end_snap', default=1, type=int)
parser.add_argument('--is_stage2', default=True, type=bool, help='whether to use stage 2 train csv')
parser.add_argument('--description', default='', type=str, help='description of the version')

args = parser.parse_args()

basic_name = f'Unet_{args.encoder_name}_v{args.version}_fold{args.fold}'
save_model_name = basic_name + '.pth'
submission_file = basic_name + '.csv'

print(save_model_name)
print(submission_file)

current_dir = os.path.dirname(os.path.realpath(__file__))


if not os.path.exists(args.output_path + 'v' + str(args.version)):
    os.makedirs(args.output_path + 'v' + str(args.version))

if not os.path.exists(args.weights_path + 'v' + str(args.version)):
    os.makedirs(args.weights_path + 'v' + str(args.version))

if not os.path.exists(args.log_path + 'v' + str(args.version)):
    os.makedirs(args.log_path + 'v' + str(args.version))

log_path_version = Path(args.log_path + 'v' + str(args.version))
log = log_path_version.joinpath(f'train_v{args.version}_fold{args.fold}.log').open('at', encoding='utf8')

output_path_version = args.output_path + 'v' + str(args.version)
params_path = Path(output_path_version + '/' + f'params_v{args.version}_fold{args.fold}.json')
params_path.write_text(json.dumps(vars(args), indent=4, sort_keys=True))

if args.is_stage2:
    train_rle_path = args.csv_path + 'stage_2_train.csv'
else:
    train_rle_path = args.csv_path + 'train-rle.csv'

data_folder = args.train_image_path

# dataloader = provider(
#     fold=args.fold,
#     total_folds=5,
#     data_folder=data_folder,
#     df_path=train_rle_path,
#     phase="train",
#     size=args.img_size_target,
#     mean=(0.485, 0.456, 0.406),
#     std=(0.229, 0.224, 0.225),
#     batch_size=batch_size,
#     num_workers=4,
# )


# batch = next(iter(dataloader))  # get a batch from the dataloader
# images, masks = batch


# plot some random images in the `batch`
# idx = random.choice(range(16))
# plt.imshow(images[idx][0], cmap='bone')
# plt.imshow(masks[idx][0], alpha=0.2, cmap='Reds')
# plt.show()


def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou, kaggle_metric = meter.get_metrics()  # kaggle_metric
    dice, dice_neg, dice_pos = dices  # | accuracy: %0.4f
    print("Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f | New Dice: %0.4f" % (
        epoch_loss, dice, dice_neg, dice_pos, iou, kaggle_metric))
    scores = "Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f | New Dice: %0.4f" % (
        epoch_loss, dice, dice_neg, dice_pos, iou, kaggle_metric)
    return dice, iou, scores, kaggle_metric


# ## UNet with ResNet34 model
# Let's take a look at the model

# ## Model Training and validation


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.fold = args.fold
        self.total_folds = 5
        self.num_workers = 6
        self.batch_size = {"train": args.batch_size, "val": args.batch_size}  # 4
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = args.learning_rate
        self.num_epochs = args.epochs
        self.best_loss = float("inf")
        self.best_dice = 0
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = MixedLoss(10.0, 2.0)

        if args.swa is True:
            # base_opt = torch.optim.SGD(self.net.parameters(), lr=args.max_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            base_opt = RAdam(self.net.parameters(), lr=self.lr)
            self.optimizer = SWA(base_opt, swa_start=38, swa_freq=1, swa_lr=args.min_lr)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, scheduler_step, args.min_lr)
        else:
            if args.optimizer.lower() == 'adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            elif args.optimizer.lower() == 'radam':
                self.optimizer = RAdam(self.net.parameters(), lr=self.lr)  # betas=(args.beta1, args.beta2),weight_decay=args.weight_decay
            elif args.optimizer.lower() == 'sgd':
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.max_lr, momentum=args.momentum,
                                                 weight_decay=args.weight_decay)

        if args.scheduler.lower() == 'reducelronplateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=args.patience, verbose=True)
        elif args.scheduler.lower() == 'clr':
            self.scheduler = CyclicLR(self.optimizer, base_lr=self.lr, max_lr=args.max_lr)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=args.fold,
                total_folds=5,
                data_folder=data_folder,
                df_path=train_rle_path,
                phase=phase,
                size=args.img_size_target,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.kaggle_metric = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks) # weighted_lovasz  # lovasz_hinge(outputs, masks) # self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        start = time.time()
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        if args.swa is True:
            self.optimizer.swap_swa_sgd()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches  # running_loss / total_batches
        dice, iou, scores, kaggle_metric = epoch_log(phase, epoch, epoch_loss, meter, start)  # kaggle_metric
        write_event(log, dice, loss=epoch_loss)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        self.kaggle_metric[phase].append(kaggle_metric)
        torch.cuda.empty_cache()
        return epoch_loss, dice, iou, scores, kaggle_metric  # kaggle_metric

    def start(self):
        if os.path.exists(args.log_path + 'v' + str(args.version)+ '/' + str(args.fold)):
            shutil.rmtree(args.log_path + 'v' + str(args.version)+ '/' + str(args.fold))
        else:
            os.makedirs(args.log_path + 'v' + str(args.version)+ '/' + str(args.fold))
        writer = SummaryWriter(args.log_path + 'v' + str(args.version) + '/' + str(args.fold))

        num_snapshot = 0
        best_acc = 0
        model_path = args.weights_path + 'v' + str(args.version) + '/' + save_model_name
        if os.path.exists(model_path):
            state = torch.load(model_path,
                               map_location=lambda storage, loc: storage)
            model.load_state_dict(state["state_dict"])  # ["state_dict"]
            epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.best_dice = state['best_dice']
            state['state_dict'] = state['state_dict']
            state['optimizer'] = state['optimizer']
        else:
            epoch = 1
            self.best_loss = float('inf')
            self.best_dice = 0

        for epoch in range(epoch, self.num_epochs + 1):
            print('-' * 30, 'Epoch:', epoch, '-' * 30)
            train_loss, train_dice, train_iou, train_scores, train_kaggle_metric = self.iterate(epoch, "train")  # train_kaggle_metric
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            try:
                val_loss, val_dice, val_iou, val_scores, val_kaggle_metric = self.iterate(epoch, "val")  # val_kaggle_metric
                self.scheduler.step(val_loss)
                if val_loss < self.best_loss:
                    print("******** New optimal found, saving state ********")
                    state["best_loss"] = self.best_loss = val_loss
                    torch.save(state, model_path)
                    try:
                        scores = val_scores
                    except:
                        scores = 'None'
                if val_dice > self.best_dice:
                    print("******** Best Dice Score, saving state ********")
                    state["best_dice"] = self.best_dice = val_dice
                    best_dice__path = args.weights_path + 'v' + str(args.version) + '/' + 'best_dice_' + basic_name + '.pth'
                    torch.save(state, best_dice__path)
                # if val_dice > best_acc:
                #     print("******** New optimal found, saving state ********")
                #     # state["best_acc"] = self.best_acc = val_dice
                #     best_acc = val_dice
                #     best_param = self.net.state_dict()

                # if (epoch + 1) % scheduler_step == 0:
                #     # torch.save(best_param, args.save_weight + args.weight_name + str(idx) + str(num_snapshot) + '.pth')
                #     save_model_name = basic_name + '.pth' # '_' +str(num_snapshot)
                #     torch.save(best_param, args.weights_path + 'v' + str(args.version) + '/' + save_model_name)
                #     # state
                #     try:
                #         scores = val_scores
                #     except:
                #         scores = 'None'
                #     optimizer = torch.optim.SGD(self.net.parameters(), lr=args.max_lr, momentum=args.momentum,
                #                                 weight_decay=args.weight_decay)
                #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, args.min_lr)
                #     num_snapshot += 1
                #     best_acc = 0
                writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
                writer.add_scalars('dice_score', {'train': train_dice, 'val': val_dice}, epoch)
                writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
                writer.add_scalars('New_Dice', {'train': train_kaggle_metric, 'val': val_kaggle_metric}, epoch)
            except KeyboardInterrupt:
                print('Ctrl+C, saving snapshot')
                torch.save(state, args.weights_path + 'v' + str(args.version) + '/' + save_model_name)
                print('done.')
            # writer.add_scalars('Accuracy', {'train': train_kaggle_metric, 'val': val_kaggle_metric}, epoch)

        # writer.export_scalars_to_json(args.log_path + 'v' + str(args.version) + '/' + basic_name + '.json')
        writer.close()
        return scores


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot');
    plt.xlabel('Epoch');
    plt.ylabel(f'{name}');
    plt.legend();
    # plt.show()
    plt.savefig(args.output_path + 'v' + str(args.version) + '/' + basic_name + '_' + name + '.png')


def notify_results(score):
    competition_name = 'SIIM\n'

    model_name = submission_file

    loss_function = 'MixedLoss'  # 'MixedLoss'  # 'dice_coef_loss_bce' lovasz_hinge
    metrics_name = 'IoU_Dice'  # 'Kaggle_IoU_Precision' and Kaggle Metric

    comment = 'Fold={}, img_size_target={}, BATCH_SIZE={} epochs = {} LR={}, loss={}, metrics={}'.format(
        args.fold, args.img_size_target, args.batch_size, args.epochs, args.learning_rate, loss_function, metrics_name)

    if score == 'None':
        message = competition_name + '\n' + model_name + '\n' + comment + '\n' + args.description
    else:
        message = competition_name + '\n' + model_name + '\n' + comment + '\n' + args.description + '\n' + score
    send_line_notification(message=message)

    print(message)


if __name__ == '__main__':
    print('Current Device : ', torch.cuda.current_device())
    print('Is Available : ', torch.cuda.is_available())
    scheduler_step = args.epochs // args.snapshot
    model = smp.Unet(args.encoder_name, encoder_weights="imagenet", activation=None)

    model_trainer = Trainer(model)
    score = model_trainer.start()

    try:
        # PLOT TRAINING
        losses = model_trainer.losses
        dice_scores = model_trainer.dice_scores  # overall dice
        iou_scores = model_trainer.iou_scores
        kaggle_metric = model_trainer.kaggle_metric

        plot(losses, "BCELoss")
        plot(dice_scores, "Dice score")
        plot(iou_scores, "IoU score")
        plot(kaggle_metric, "New Dice")
    except:
        pass

    try:
        notify_results(score=score)
    except:
        pass
