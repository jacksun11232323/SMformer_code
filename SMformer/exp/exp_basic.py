import os
import torch
import torch.nn as nn
from models import  Transformer, TimesNet, SMformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'SMformer': SMformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.mask_loss = nn.Parameter(torch.Tensor([args.mask_loss]),requires_grad=True)
        self.shuffle_loss = nn.Parameter(torch.tensor(args.shuffle_loss),requires_grad=True)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
