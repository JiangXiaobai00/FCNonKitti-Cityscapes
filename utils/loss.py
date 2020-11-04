import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_entropy2d(input, target, weight=None, size_average=True):
    # taken from https://github.com/meetshah1995/pytorch-semseg
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def compute_mask(input, target):
    # mask out depth values in predicted and target depth which are <= 0
    mask = np.logical_and(input.data.cpu().numpy() > 0, target.data.cpu().numpy() > 0)
    total_pixel = np.prod(input.size(), dtype=np.float32).item()
    total_pixel = total_pixel - np.sum(mask)
    mask = torch.from_numpy(mask.astype(int)).float().to(device)
    return mask, total_pixel

def l1_loss(input, target, smooth=True):
    if not input.size() == target.size():
        _, _, H, W = target.size()
        input = F.upsample(input, size=(H, W), mode='bilinear')

    # mask out depth values in input and target which are <= 0
    mask, total_pixel = compute_mask(input, target)
    diff = torch.abs(target - input)
    diff = diff * mask
    loss = torch.sum(diff) / total_pixel
    if smooth:
        loss = loss + smooth_loss(input=input) / 1000.0   # empirical weight for smooth loss
    return loss

# input, target: [batch_size, 1, h, w]
def scale_invariant_loss(input, target, smooth=True):
    if not input.size() == target.size():
        _, _, H, W = target.size()
        input = F.upsample(input, size=(H, W), mode='bilinear')

    # mask out depth values in input and target which are <= 0
    mask, total_pixel = compute_mask(input, target)

    first_log = torch.log(torch.clamp(input, min=1e-3))
    second_log = torch.log(torch.clamp(target, min=1e-3))
    diff = first_log - second_log
    diff = diff * mask
    loss = torch.sum((diff ** 2))/total_pixel - (torch.sum(diff) ** 2)/(total_pixel ** 2)
    if smooth:
        loss = loss + smooth_loss(input=input) / 1000.0
    return loss

def smooth_loss(input):
    dx, dy = gradient(input)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss = dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
    return loss

def gradient(pred):
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy