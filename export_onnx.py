# encoding=utf-8

from collections import OrderedDict

import torch.onnx
from easydict import EasyDict as edict

from models.networks import *


def select_device(device='', apex=False, batch_size=None):
    """
    :param device:
    :param apex:
    :param batch_size:
    :return:
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def load_network(net, save_dir, network_label, epoch_label, is_parallel=True):
    """
    :param net:
    :param network_label:
    :param epoch_label:
    :return:
    """
    save_file_name = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_file_name)
    save_path = os.path.abspath(save_path)

    state_dict = torch.load(save_path)
    new_state_dict = OrderedDict()
    if not is_parallel:
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
    print("{:s} loaded.".format(save_path))


def export():
    """
    :return:
    """
    opt = {
        'device': 'cpu',  # '0'
        'gpu_ids': [0],
        'checkpoint_path': '/mnt/diskc/even/EnlightenGAN/checkpoints/enlighten/latest_net_G_A.pth',
        'input_nc': 3,
        'self_attention': True,
        'use_norm': 1,
        'syn_norm': False,
        'use_avgpool': 0,
        'tanh': False,
        'net_height': 448,
        'net_width': 768,
    }
    opt = edict(opt)

    # Set up device
    device = select_device(opt.device)

    # Define network
    net = Unet_resize_conv(opt, skip=True)

    # Load checkpoint
    save_dir = os.path.split(opt.checkpoint_path)[0]
    load_network(net, save_dir, "G_A", "latest", False)

    # Init bwtwork
    # net.to(device).eval()
    dummy_input = torch.randn(1, 3, opt['net_height'], opt['net_width'])
    input_names = ["actual_input"]
    output_names = ["output"]

    torch.onnx.export(net,
                      dummy_input,
                      "./enlightenGAN.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)


if __name__ == "__main__":
    export()
    print("Done.")
