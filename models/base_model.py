# encoding=utf-8

import os

import torch


class BaseModel():
    def name(self):
        """
        :return:
        """
        return 'BaseModel'

    def initialize(self, opt):
        """
        :param opt:
        :return:
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        """
        :param input:
        :return:
        """
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        """
        :return:
        """
        return self.input

    def get_current_errors(self):
        """
        :return:
        """
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        """
        :param network:
        :param network_label:
        :param epoch_label:
        :param gpu_ids:
        :return:
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)

        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

        print("{:s} saved.".format(os.path.abspath(save_path)))

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        """
        :param network:
        :param network_label:
        :param epoch_label:
        :return:
        """
        save_file_name = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_file_name)
        save_path = os.path.abspath(save_path)
        network.load_state_dict(torch.load(save_path))
        print("{:s} loaded.".format(save_path))

    def update_learning_rate():
        pass
