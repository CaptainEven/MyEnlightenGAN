# encoding=utf-8

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    """
    :param opt:
    :return:
    """
    transform_list = []

    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1 * radom.randint(0, 4)
        osize = [int(400 * zoom), int(600 * zoom)]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
        # elif opt.resize_or_crop == 'no':
    #     osize = [384, 512]
    #     transform_list.append(transforms.Scale(osize, Image.BICUBIC))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    """
    :param img:
    :param target_width:
    :return:
    """
    ow, oh = img.size

    if (ow == target_width):
        return img

    w = target_width
    h = int(target_width * oh / ow)

    return img.resize((w, h), Image.BICUBIC)
