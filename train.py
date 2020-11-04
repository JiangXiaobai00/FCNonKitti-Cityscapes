# -*- coding:utf-8 -*-
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
from dataloader import KITTILoader as KL
from models import *
from utils import function
from utils.loss import cross_entropy2d
CUDA_VISIBLE_DEVICES=1
torch.cuda.set_device(1)

parser = argparse.ArgumentParser(description='FCN')
parser.add_argument('--n_class', type=int, default=19,
                    help='set class for seg')
parser.add_argument('--PSMmodel', default='basic',
                    help='select model')#basic
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--data_path', default='./dataset/data_semantics/',#'./dataset/cityscapes',
                    help='datapath')
parser.add_argument('--batch_size', type=int, default=2,
                    help='number of batch to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')#
parser.add_argument('--savemodel', default='./snapshots/',
                    help='save model')
parser.add_argument('--logdir', default='./snapshots/',
                    help='save log')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

t_loader = KL.KITTIDataset(args.data_path,phase='train',n_class=args.n_class,flip_rate=0.5)#CL.CityScapesDataset
v_loader =KL.KITTIDataset(args.data_path,phase='val',n_class=args.n_class,flip_rate=0.)#CL.CityScapesDataset
trainloader = data.DataLoader(t_loader,batch_size=1, shuffle=True, num_workers=4,drop_last=False)
valloader = data.DataLoader(v_loader,batch_size=1, shuffle=False, num_workers=4, drop_last=False)

model = basic(args.maxdisp,args.n_class)


if args.cuda:
    model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))

def main():
    writer = SummaryWriter(log_dir=args.logdir)
    print('RUNDIR: {}'.format(args.logdir))
    sys.stdout.flush()
    #shutil.copy(args.config, args.logdir)  # copy config file to path of logdir
    logger = function.get_logger(args.logdir)
    logger.info('Let the games begin')  # write in log file
    running_metrics_val = function.runningScoreSeg(args.n_class)
    val_loss_meter = function.averageMeter()
    time_meter = function.averageMeter()
    best_iou = -100.0
    i = 0
    flag = True
    train_iters=200000
    print_iter= 150 #150
    val_iters=200 #750
    save_iters=10000
    while i <= train_iters and flag:  # 3000000
        print(len(trainloader))
        for (leftimg,labels) in trainloader:
            start_ts = time.time()  # return current time stamp
            model.train()  # set model to training mode
            imgL = Variable(leftimg).cuda()
            label = Variable(labels).cuda()#1,256,512

            optimizer.zero_grad()  # clear earlier gradients
            output0 = model(imgL)
            loss = cross_entropy2d(output0, label).to(device)

            loss.backward()  # backpropagation loss
            optimizer.step()  # optimizer parameter update
            time_meter.update(time.time() - start_ts)

            if (i + 1) %print_iter == 0:  # 150
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,train_iters,loss.item(),time_meter.val / args.batch_size)
                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i + 1 )
                time_meter.reset()

            if (i + 1) % save_iters == 0:
                savefilename = args.savemodel + 'iter_' + str(i) + '.tar'
                torch.save({
                    'iter': i,
                    'state_dict': model.state_dict(),
                }, savefilename)

            torch.cuda.empty_cache()
            if (i + 1) % val_iters == 0 or (i + 1) == train_iters:  # 750
                model.eval()
                with torch.no_grad():
                    for i_val, (leftimgval, labelval) in tqdm(enumerate(valloader)):
                        imgL = Variable(leftimgval.cuda())
                        output0val = model(imgL) # [batch_size, n_classes, height, width]

                        #segmetation
                        lossval = cross_entropy2d(output0val, labelval.cuda())# mean pixelwise loss in a batch
                        pred = output0val.data.max(1)[1].cpu().numpy()  # [batch_size, height, width]#229,485
                        gt = labelval.data.cpu().numpy()  # [batch_size, height, width]#256,512
                        running_metrics_val.update(gt=gt, pred=pred)
                        val_loss_meter.update(lossval.item())

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i + 1)
                logger.info("Iter %d val_loss: %.4f" % (i + 1, val_loss_meter.avg))
                print("Iter %d val_loss: %.4f" % (i + 1, val_loss_meter.avg))

                # output scores
                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    sys.stdout.flush()
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i + 1)
                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i + 1)
                torch.cuda.empty_cache()

                # SAVE best model for segmentation
                save_model = False
                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    save_model = True
                if save_model:
                    # SAVE
                    savefilename = args.savemodel + 'model_best_iou'+ '.tar'
                    torch.save({
                        'iter': i,
                        'state_dict': model.state_dict(),
                        "best_iou": best_iou,
                    }, savefilename)
                val_loss_meter.reset()
                running_metrics_val.reset()

            if (i + 1) == train_iters:
                flag = False
                break
            i += 1

if __name__ == '__main__':
    main()
