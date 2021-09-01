# encoding=utf-8

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import os
import os.path

import cv2
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    """
    :param filename:
    :return:
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    :param dir:
    :return:
    """
    images = []
    abs_path = os.path.abspath(dir)
    print('Absolute dir path: ', abs_path)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, f_names in sorted(os.walk(dir)):
        for f_name in f_names:
            if is_image_file(f_name):
                path = os.path.join(root, f_name)
                images.append(path)

    return images



def pad_img(input, divide=16):
    """
    :param input:
    :param divide:
    :return:
    """
    h, w, c = input.shape
    height_org, width_org = input.shape[2], input.shape[3]

    if w % divide != 0 or h % divide != 0:
        width_res = w % divide
        height_res = h % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        # padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        # input = padding(input)

        ## using opencv
        input = cv2.copyMakeBorder(input, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def store_dataset(dir, read_method=0):
    """
    Using memory as cache here...
    :param dir:
    :return:
    """
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, f_names in sorted(os.walk(dir)):
        for f_name in f_names:
            if is_image_file(f_name):
                path = os.path.join(root, f_name)

                if read_method == 0:  #
                    img = Image.open(path).convert('RGB')  # Using PIL Image open a image
                elif read_method == 1:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)

                images.append(img)
                all_path.append(path)

    return images, all_path


def default_loader(path):
    """
    :param path:
    :return:
    """
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
