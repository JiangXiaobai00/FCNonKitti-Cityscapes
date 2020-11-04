#!/usr/bin/env python
#-*- coding:utf-8 -*-
#@Time :20-1-3下午8:09
#@Author:Bristar
#@File :testseg.py

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import sys
import shutil
import argparse
import torchvision.models
from tqdm import tqdm
from tensorboardX import SummaryWriter
from dataloader import Cityscapeloader as CL
import skimage
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
from models import *
from utils import function,drawseg
from utils.loss import cross_entropy2d

CUDA_VISIBLE_DEVICES=0
torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description='MENet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')#192
parser.add_argument('--n_class', type=int, default=19,
                    help='set class for seg')
parser.add_argument('--PSMmodel', default='basic',
                    help='select model')#basic
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--data_path', default='./dataset/cityscapes/',#
                    help='datapath')
parser.add_argument('--batch_size', type=int, default=1,
                    help='number of batch to train')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='number of batch to train')
parser.add_argument('--loadmodel', default='/media/c4007/387ca079-3067-4f93-afce-0233ab11a53c/fcn/model/mix7_city/model_best_iou_iter_mini.tar',)
parser.add_argument('--logdir', default='/media/c4007/387ca079-3067-4f93-afce-0233ab11a53c/fcn/model/project/',
                    help='save log')
parser.add_argument('--saveseg', default='/media/c4007/387ca079-3067-4f93-afce-0233ab11a53c/fcn/model/project/seg/',
                    help='save img2')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

v_loader = CL.CityScapesDataset(args.data_path,phase='val',n_class=args.n_class,flip_rate=0.)#
valloader = data.DataLoader(v_loader,batch_size=1, shuffle=False, num_workers=4, drop_last=False)

model = basic(args.maxdisp,args.n_class)

if args.cuda:
    model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])



def main():
    print('RUNDIR: {}'.format(args.logdir))
    sys.stdout.flush()
    logger = function.get_logger(args.logdir)
    logger.info('test')  # write in log file
    running_metrics_val = function.runningScoreSeg(args.n_class)
    val_loss_meter = function.averageMeter()
    time_meter = function.averageMeter()

    print(len(valloader))
    start_ts = time.time()  # return current time stamp
    model.eval()
    with torch.no_grad():
                    for i_val, (leftimgval,rightimgval, labelval, disp_true,L_name) in tqdm(enumerate(valloader)):
                        imgL = leftimgval.numpy()
                        imgR = rightimgval.numpy()

                        if args.cuda:
                            imgL = torch.FloatTensor(imgL).cuda()
                            imgR = torch.FloatTensor(imgR).cuda()

                        imgL, imgR = Variable(imgL), Variable(imgR)
                        output= model(imgL, imgR)  # 1 1024 2048    1 19 1024 2048
                        pred_seg = output0.data.cpu().numpy()       

                        # FCN OF SEGMETATION

                        N, _, h, w = pred_seg.shape  # 4,12,192,704,numpy
                        pred_segmap = pred_seg.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N,
                                                                                                                      h,
                                                                                                                      w)
                        img = drawseg.direct_render(pred_segmap, args.n_class)
                        skimage.io.imsave(args.saveseg + (L_name[0].split('/')[-1]), img[0])

                        # segmetation mIoU
                        score =torch.from_numpy(pred_seg).cuda()
                        lossval = cross_entropy2d(score, labelval.cuda())# mean pixelwise loss in a batch
                        pred = score.data.max(1)[1].cpu().numpy()  # [batch_size, height, width]#229,485
                        gt = labelval.data.cpu().numpy()  # [batch_size, height, width]#256,512
                        running_metrics_val.update(gt=gt, pred=pred)
                        val_loss_meter.update(lossval.item())

                        torch.cuda.empty_cache()
                    logger.info(" val_loss: %.4f" % ( val_loss_meter.avg))

                    print("val_loss: %.4f" % ( val_loss_meter.avg))
                    #"""
                    # output scores
                    score, class_iou = running_metrics_val.get_scores()
                    for k, v in score.items():

                      print(k, v)
                      sys.stdout.flush()
                      logger.info('{}: {}'.format(k, v))

                    for k, v in class_iou.items():
                      print(k, v)
                      logger.info('{}: {}'.format(k, v))
                    #"""

if __name__ == '__main__':

    main()
