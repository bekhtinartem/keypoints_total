from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torchvision


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class resnet50(nn.Module):
    def __init__(self, requires_grad, count):
        super(resnet50, self).__init__()

        self.model = pretrainedmodels.__dict__['resnet_50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # change the final layer
        self.l0 = nn.Linear(2048, count)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0

class xception(nn.Module):
    def __init__(self, requires_grad, count):
        super(xception, self).__init__()

        self.model = pretrainedmodels.__dict__['xception'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # change the final layer
        self.l0 = nn.Linear(2048, count)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0


class densenet(nn.Module):
    def __init__(self, requires_grad, count):
        super(densenet, self).__init__()
        self.model = pretrainedmodels.__dict__['densenet169'](pretrained=None)
        self.add_module(name="l0", module=nn.Linear(1664, count))
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True

        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0
class pnasnet5large(nn.Module):
    def __init__(self,  requires_grad, count):
        super(pnasnet5large, self).__init__()
        self.model = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1001, pretrained=None)
        self.add_module(name="l0", module=nn.Linear(4320, count))
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True

        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0


class resnet18(nn.Module):
    def __init__(self, requires_grad, count):
        super(resnet18, self).__init__()
        self.model = pretrainedmodels.__dict__['resnet18'](pretrained=None)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True

        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False

        # change the final layer
        self.l0 = nn.Linear(512, count)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0


class squeezenet1_1(nn.Module):
    def __init__(self, requires_grad, count):
        super(squeezenet1_1, self).__init__()


        self.model = pretrainedmodels.__dict__['squeezenet1_1'](pretrained=None)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # change the final layer
        self.l0 = nn.Linear(512, count)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0


class mobilenet_v3_small(nn.Module):
    def __init__(self, requires_grad, count):
        super(mobilenet_v3_small, self).__init__()

        self.model = torchvision.models.mobilenet_v3_small(pretrained=False)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # change the final layer
        self.l0 = nn.Linear(576, count)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0
class mobilenet_v3_large(nn.Module):
    def __init__(self, requires_grad, count):
        super(mobilenet_v3_large, self).__init__()

        self.model = torchvision.models.mobilenet_v3_large(pretrained=False)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # change the final layer
        self.l0 = nn.Linear(960, count)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0



class regnet_x_8gf(nn.Module):
    def __init__(self, requires_grad, count):
        super(regnet_x_8gf, self).__init__()
        self.model = torchvision.models.regnet_x_8gf(pretrained=False)
        self.model.fc=nn.Linear(1920, count)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
    def forward(self, x):

        return self.model(x)