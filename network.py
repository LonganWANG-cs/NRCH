import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import time
import torch

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoint/' + name)
        return name

    def forward(self, *input):
        pass

class ImgModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        #self.apply(weights_init)
        self.norm = norm
    def forward(self, x):
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out
    

class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
        super(TxtModule, self).__init__()
        self.module_name = "text_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        self.norm = norm

    def forward(self, x):
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out