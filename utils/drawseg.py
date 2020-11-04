from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
import random
from skimage import io, transform
import os
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

class_name = ('road', 'sidewalk', 'building', 'wall',
                   'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
                   'terrain', 'sky', 'person', 'rider', 'car',
                   'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                   )
class_color = ((128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
                    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), \
                    (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), \
                    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                    )
label_map = np.array(class_color)

class_n = 19
mean = [0.2902, 0.2976, 0.3042]
std = [0.1271, 0.1330, 0.1431]
flip_rate = 0
shrink_rate = 1

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
  image = [img for img in os.listdir(filepath) if img.find('_10') > -1]
  left_test  = [filepath+img for img in image]
  return left_test


def gettransform(img):
    h, w, c = img.shape
    h = int(h // 32 * shrink_rate) * 32
    w = int(w // 32 * shrink_rate) * 32

        # use interpolation for quality
    img=transform.resize(img, (h, w), order=1, mode='constant', preserve_range=True).astype('uint8')
    if np.random.random() < flip_rate:
        img= np.fliplr(img)  # cause error if remove '.copy()' (prevent memory sharing)
    img= transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])(img)
    return img


def draw_img(rgb,segmentation, n_class):
    # mask
    mask = np.zeros_like(rgb, dtype=np.float32)#(384,1248,3)
    for clsid in range(n_class):
        mask += np.dot((segmentation == clsid)[..., np.newaxis], [label_map[clsid]])
    rgb = np.clip(np.round(mask * 1), 0, 255.0).astype(np.uint8)
    return rgb

def direct_render(label,n_class):
    renders = []
    if not isinstance(label, torch.Tensor):
        label = torch.from_numpy(label)
    _,h, w = label.shape
    temp_label = np.zeros((1,h, w, 3), dtype='uint8')# B H W C np
    for i, segmentation in enumerate(label):
        render = draw_img(temp_label[i], segmentation, n_class)
        renders.append(render)
    renders = np.array(renders)
    return renders

def visualize(label):
    if not isinstance(label, torch.Tensor):
        label = torch.from_numpy(label)
    h, w = label.shape
    temp_label = np.zeros((h, w, 3), dtype='uint8')
    for i in range(h):  # how to write more elegantly
        for j in range(w):
            temp_label[i, j] = class_color[int(label[i, j])]

    return transforms.ToTensor()(temp_label)


def denormalize(self, image):
    image = np.transpose(image, (1, 2, 0))
    image[:, :, 0] = image[:, :, 0] * self.std[0] + self.mean[0]
    image[:, :, 1] = image[:, :, 1] * self.std[1] + self.mean[1]
    image[:, :, 2] = image[:, :, 2] * self.std[2] + self.mean[2]
    return np.transpose(image, (2, 0, 1))

#process
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

#__imagenet_stats ={'mean': [0.2902, 0.2976, 0.3042],
                    #                   'std': [0.1271, 0.1330, 0.1431]}


__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        #transforms.RandomSizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    if augment:
            return inception_color_preproccess(input_size, normalize=normalize)
    else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)




class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

def findmax(numpy, n_class):
    N, _, H, W = numpy.shape
    dnumpy = numpy.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, H, W)
    return dnumpy


def toRGB(img, dtype=np.uint8):
    dnumpy = (img.transpose(0, 2, 3, 1) * 255).astype(dtype)  # 1,384,1248,3
    dnumpy = np.round(dnumpy)
    dnumpy = np.clip(dnumpy, 0, 255)
    return dnumpy

def draw_img1(rgb,segmentation, n_class, opacity):
    #rgb[segmentation > 0] *= 1 - opacity
    # mask
    mask = np.zeros_like(rgb, dtype=np.float32)#(384,1248,3)
    for clsid in range(n_class):
        mask += np.dot((segmentation == clsid)[..., np.newaxis], [label_map[clsid]])
    # paste
    #rgb = np.clip(np.round(rgb + mask * opacity), 0, 255.0).astype(np.uint8)
    #rgb = np.clip(np.round(mask * opacity), 0, 255.0).astype(np.uint8)
    rgb = np.clip(np.round(mask * 1), 0, 255.0).astype(np.uint8)
    return rgb

def direct_render1(img, predict_map,n_class=21,opacity=0.5):
    renders = []
    rgb = toRGB(img, dtype=np.float32)#  1,384,1248,3
    for i, segmentation in enumerate(predict_map):
        render = draw_img1(rgb[i], segmentation, n_class=n_class, opacity=opacity)
        renders.append(render)
    renders = np.array(renders)
    return renders