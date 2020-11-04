# -*- coding:utf-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess,drawseg
from models import *
from matplotlib import pyplot
import PIL.Image as Image
from matplotlib import pyplot as plt
CUDA_VISIBLE_DEVICES=1
torch.cuda.set_device(1)

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='./testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./model_best_iou.tar',
                    help='loading model')
parser.add_argument('--model', default='basic',
                    help='select model')
parser.add_argument('--n_class', type=int, default=19,
                    help='set class for seg')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saveseg', default='./result/',
                    help='save img')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.datapath)


model = basic(args.maxdisp,args.n_class)
#model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def val(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output  = model(imgL,imgR)
        pred_seg = output0.data.cpu().numpy()
        return pred_seg


def main():
   processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):
       imgL_o = Image.open(test_left_img[inx]).convert('RGB')
       imgR_o = Image.open(test_right_img[inx]).convert('RGB')
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()


       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
       # pad to (384, 1248)
       top_pad = 384 - imgL.shape[2]
       left_pad = 1248 - imgL.shape[3]
       imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
       imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
       start_time = time.time()
       pred_seg = val(imgL,imgR)#(384,1248),(1,21,384,1248)
       print('time = %.2f' %(time.time() - start_time))
       #FCN OF SEGMETATION
       pred_seg = pred_seg[:,:,top_pad:, :-left_pad]
       N, _, h, w = pred_seg.shape # 4,12,192,704,numpy
       pred_segmap = pred_seg.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N, h, w)
       img = drawseg.direct_render(pred_segmap, args.n_class,imgL_o)
       skimage.io.imsave(args.saveseg+(test_left_img[inx].split('/')[-1]), img[0])


if __name__ == '__main__':
   main()






