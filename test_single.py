# encoding=utf-8

from collections import OrderedDict

import cv2
import torch.onnx
import torchvision.transforms as transforms
from PIL import Image
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


def pad_tensor(input, divide=16):
    """
    :param input:
    :param divide:
    :return:
    """
    if len(input.shape) != 4:
        input = torch.unsqueeze(input, dim=0)

    height_org, width_org = input.shape[-2:]

    if width_org % divide != 0 or height_org % divide != 0:
        width_res = width_org % divide
        height_res = height_org % divide
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

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[-2:]
    assert width % divide == 0, 'width cant divided by stride'
    if height % divide != 0:
        print('height cant divided by stride: {:d}//{:d} != 0' \
              .format(height, divide))
        exit(-1)

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    """
    :param input:
    :param pad_left:
    :param pad_right:
    :param pad_top:
    :param pad_bottom:
    :return:
    """
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def tensor2im(image_tensor, img_type=np.uint8):
    """
    :param image_tensor:
    :param img_type:
    :return:
    """
    img_np = image_tensor[0].cpu().float().numpy()
    img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0  # CHW ——> HWC
    img_np = np.maximum(img_np, 0)
    img_np = np.minimum(img_np, 255)
    # np.clip(img_np, 0, 255)
    return img_np.astype(img_type)


def test():
    """
    :return:
    """
    opt = {
        'device': 'cpu',  # '0'
        'gpu_id': '7',
        'checkpoint_path': '/mnt/diskc/even/EnlightenGAN/checkpoints/enlighten/latest_net_G_A.pth',
        'input_nc': 3,
        'self_attention': True,
        'use_norm': 1,
        'syn_norm': False,
        'use_avgpool': 0,
        'tanh': False,
        'net_height': 448,
        'net_width': 768,
        "img_path": "/mnt/diskc/even/testset/testA/0.jpg"
    }
    opt = edict(opt)
    if not os.path.isfile(opt.img_path):
        print("[Err]: invalid image path.")
        exit(-1)

    ## ----- Set up device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    device = select_device()

    ## ----- Define network
    net = Unet_resize_conv(opt, skip=True)

    ## ----- Load checkpoint
    save_dir = os.path.split(opt.checkpoint_path)[0]
    load_network(net, save_dir, "G_A", "latest", False)

    ## ----- Init network
    net.to(device)
    net.eval()  # eval mode

    ## ----- Define transformations
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

    ## ----- Set up input data
    img = Image.open(opt.img_path).convert('RGB')
    img = trans(img)
    img, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(img)
    img = img.to(device)

    with torch.no_grad():
        ## ----- Inference
        fake_B, latent = net.forward(img)

        ## ----- Post processing
        fake_B = pad_tensor_back(fake_B, pad_left, pad_right, pad_top, pad_bottom)
        fake_B = tensor2im(fake_B.data)  # tensor to numpy array
        fake_B = cv2.cvtColor(fake_B, cv2.COLOR_BGR2RGB)
    # print(fake_B.shape)

    # ----- Save output
    save_path = "./test_single.jpg"
    save_path = os.path.abspath(save_path)
    cv2.imwrite(save_path, fake_B)
    print("{:s} saved.".format(save_path))


if __name__ == "__main__":
    test()
    print("Done.")
