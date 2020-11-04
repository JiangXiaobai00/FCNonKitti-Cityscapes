# -*- coding: utf-8 -*-
from __future__ import print_function
from PIL import Image
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import torch
from torch.utils.data import Dataset
from utils.function import recursive_glob
import dataloader.preprocess as preprocess
import sys
import copy

VGG_MEAN = [103.939, 116.779, 123.68]
means = np.array([103.939, 116.779, 123.68])  # / 255. # mean of three channels in the order of BGR
h, w = 1024, 2048
train_h = int(h / 2)  # 256
train_w = 832  # 512  # 512
val_h = int(1024/2)  # 1024
val_w = int(2048/2) # 2048


class CityScapesDataset(Dataset):
    # 19classes, RGB of maskes
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, phase, n_class=19, flip_rate=0.):

        self.means = means
        self.n_class = n_class
        self.flip_rate = flip_rate
        self.split = phase
        self.files = {}
        self.root = root
        self.leftimages_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.rightimages_base = os.path.join(self.root, "rightImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )
        self.disp_base = os.path.join(
            self.root, "disparity", self.split
        )
        self.files[phase + 'left'] = recursive_glob(rootdir=self.leftimages_base, suffix=".png")
        self.files[phase + 'right'] = recursive_glob(rootdir=self.rightimages_base, suffix=".png")
        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        else:
            self.new_h = val_h
            self.new_w = val_w
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",  #
            "car",
            "truck",
            "bus",  #
            "train",
            "motorcycle",  #
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.decode_class_map = dict(zip(range(19), self.valid_classes))
        self.saliency_eval_depth = False  # False
        if not self.files[phase + 'left'] and not self.files[phase + 'right']:
            raise Exception(
                "No files for split=[%s] found in %s" % ('left', self.leftimages_base)
            )
        print("Found %d %s images" % (len(self.files[phase + 'left']), phase + 'left'))
        sys.stdout.flush()

    def __len__(self):
        return len(self.files[self.split + 'left'])

    def __getitem__(self, idx):
        Limg_name = self.files[self.split + 'left'][idx].rstrip()
        Rimg_name = os.path.join(self.rightimages_base, Limg_name.split(os.sep)[-2],
                                 os.path.basename(Limg_name)[
                                 :-15] + "rightImg8bit.png", )  # self.files[self.split + 'right'][idx].rstrip()
        lbl_path = os.path.join(
            self.annotations_base, Limg_name.split(os.sep)[-2],
            os.path.basename(Limg_name)[:-15] + "gtFine_labelIds.png",
        )
        disp_path = os.path.join(self.disp_base, Limg_name.split(os.sep)[-2],
                                 os.path.basename(Limg_name)[:-15] + "disparity.png",
                                 )

        imgL = Image.open(Limg_name).convert('RGB')
        imgR = Image.open(Rimg_name).convert('RGB')

        w, h = imgL.size
        lbl = scipy.misc.imread(lbl_path)  # original label size: 1024*2048
        disp = Image.open(disp_path)
        disp = np.ascontiguousarray(disp, dtype=np.float32) / 256

        top = random.randint(0, h - self.new_h)
        left = random.randint(0, w - self.new_w)
        imgL = imgL.crop((left, top, self.new_w + left, self.new_h + top))
        imgR = imgR.crop((left, top, self.new_w + left, self.new_h + top))

        lbl = lbl[top:top + self.new_h, left:left + self.new_w]
        label = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        disp = disp[top:top + self.new_h, left:left + self.new_w]

        processed = preprocess.get_transform(augment=False)
        imgL = processed(imgL)
        imgR = processed(imgR)

        label = torch.from_numpy(label.copy()).long()  # int64,256,512,tensor,cpu
        disp = torch.from_numpy(disp.copy()).float()
        return imgL, imgR, label, disp,Limg_name

    def decode_segmap_tocolor(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_class):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def decode_segmap_tolabelId(self, temp):
        labels_ID = temp.copy()
        for i in range(19):
            labels_ID[temp == i] = self.valid_classes[i]
        return labels_ID

    def encode_segmap(self, mask):
        # Put all void classes to 250
        # map valid classes to 0~18
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask