import os.path as osp


import torch.nn as nn

from .backbone.spectral_resnet import *


class FCN8s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=1ieXWyoG68xqoHJPWdyrDyaIaaWlfmUxI',  # NOQA
            path=cls.pretrained_model,
            md5='de93e540ec79512f8770033849c8ae89',
        )

    def __init__(self, input_dim, spectral_normalization, pretrained=False, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        resnet = resnet18(input_dim,pretrained=pretrained,spectral_normalization=spectral_normalization)
        modules = list(resnet.children())[:-2]      # delete the last fc layer and the average pooling
        self.compute_features = nn.Sequential(*modules)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.classif_big = nn.Conv2d(4096, n_class, 1)
        self.upsample_big = nn.Upsample(size=input_dim, mode='bilinear', align_corners=True)



        ''''# layer usefull

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        '''

    def forward(self, x):
        print('INPUT', x.size())
        x = self.compute_features(x)  # ResNet
        print('OUTPUT',x.size()) # 1/32
        x = self.upsample_big(x)
        x = self.classif_big(x)

        return x

