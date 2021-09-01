# encoding=utf-8

import os

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import Visualizer

opt = TestOptions().parse()

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

## @even
opt.dataroot = "../testset"
opt.name = "enlighten"
opt.model = "single"
opt.which_direction = "AtoB"
opt.no_dropout = True
opt.dataset_mode = "unaligned"
opt.which_model_netG = "sid_unet_resize"
opt.skip = 1
opt.use_norm = 1
opt.use_wgan = 0
opt.self_attention = True
opt.times_residual = True
opt.instance_norm = 0
opt.resize_or_crop = "no"
opt.which_epoch = "latest"
print(opt.gpu_ids)
opt.gpu_ids = [7]
opt.nThreads = 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s'
                    % (opt.name, opt.phase, opt.which_epoch))

# test
print("Total {:d} images tested.".format(len(dataset)))
for i, data in enumerate(dataset):  # batchSize == 1
    pad_left = int(data["pad_left"][0].data)
    pad_right = int(data["pad_right"][0].data)
    pad_top = int(data["pad_top"][0].data)
    pad_bottom = int(data["pad_bottom"])
    # print(pad_left, pad_right, pad_top, pad_bottom)

    model.set_input(data)
    visuals = model.predict(pad_left, pad_right, pad_top, pad_bottom)
    img_path = model.get_image_paths()

    print('\nprocess image... {:s} | {:d}/{:d}'
          .format(img_path[0], i + 1, len(dataset)))
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
