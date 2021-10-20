import os.path as osp

import torchvision.models as models
import torch.nn as nn

from .backbone.spectral_resnet import *


class FCN8s(nn.Module):


    def __init__(self, input_dim, spectral_normalization, pretrained=False, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        resnet = resnet18(input_dim,pretrained=pretrained,spectral_normalization=spectral_normalization)
        modules = list(resnet.children())[:-2]      # delete the last fc layer and the average pooling
        self.compute_features = nn.Sequential(*modules)


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.classif_big = nn.Conv2d(512, n_class, 1)
        self.upsample_big = nn.Upsample(size=input_dim, mode='bilinear', align_corners=True)





    def forward(self, x):
        #print('INPUT', x.size()) # 1/1 -> 768
        x = self.compute_features(x)  # ResNet


        #print('OUTPUT',x.size()) # 1/32 -> 24
        x = self.upsample_big(x)
        x = self.classif_big(x)

        return x



'''class FCN8s(nn.Module):


    def __init__(self, input_dim, spectral_normalization, pretrained=False, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        resnet = resnet18(input_dim,pretrained=pretrained,spectral_normalization=spectral_normalization)
        modules = list(resnet.children())    # delete the last fc layer and the average pooling
        modules = modules[:-2]
        self.compute_features = nn.Sequential(*modules)


        # fc6
        self.fc6 = nn.Conv2d(512, 2048,  kernel_size=7, stride=1, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        #self.drop6 = nn.Dropout2d()
        self.bn6 = nn.BatchNorm2d(2048)

        # fc7
        self.fc7 = nn.Conv2d(2048, 2048, 1)
        self.relu7 = nn.ReLU(inplace=True)
        #self.drop7 = nn.Dropout2d()
        self.bn7 = nn.BatchNorm2d(2048)

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.relu_fr = nn.ReLU(inplace=True)



        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_big = nn.Upsample(size=input_dim, mode='bilinear', align_corners=True)

        self.upscoreconv2 = nn.Conv2d(n_class, n_class, kernel_size=3, stride=1, padding=1)





    def forward(self, x):
        #print('INPUT', x.size()) # 1/1 -> 768
        x1 = self.compute_features(x)  # ResNet # 1/32



        h = self.relu6(self.fc6(x1))
        h = self.bn6(h)

        h = self.relu7(self.fc7(h))
        h = self.bn7(h)

        h = self.score_fr(h)
        h=self.relu_fr(h)
        h = self.upsample_big(h) # 1
        h =self.upscoreconv2(h) # 1

        return h


class FCN8s_bad(nn.Module):


    def __init__(self, input_dim, spectral_normalization, pretrained=False, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        resnet = resnet18(input_dim,pretrained=pretrained,spectral_normalization=spectral_normalization)
        modules = list(resnet.children())    # delete the last fc layer and the average pooling
        modules1 = modules[:-4]
        modules2 = modules[-4:-3]
        modules3 = modules[-3:-2]
        self.compute_features1 = nn.Sequential(*modules1)
        self.compute_features2 = nn.Sequential(*modules2)
        self.compute_features3 = nn.Sequential(*modules3)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096,  kernel_size=7, stride=1, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(128, n_class, 1)


        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.classif_big = nn.Conv2d(512, n_class, 1)
        self.upsample_big = nn.Upsample(size=input_dim, mode='bilinear', align_corners=True)

        self.upscoreconv2 = nn.Conv2d(n_class, n_class, kernel_size=3, stride=1, padding=1)
        self.upscoreconv8 = nn.Conv2d(n_class, n_class, kernel_size=3, stride=1, padding=1)
        self.upscoreconv4 = nn.Conv2d(n_class, n_class, kernel_size=3, stride=1, padding=1)




    def forward(self, x):
        #print('INPUT', x.size()) # 1/1 -> 768
        x1 = self.compute_features1(x)  # ResNet # 1/8
        x2 = self.compute_features2(x1)  # ResNet # 1/16
        x3 = self.compute_features3(x2)  # ResNet # 1/32


        h = self.relu6(self.fc6(x3))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h) # 1/16
        h =self.upscoreconv2(h) # 1/16
        upscore2 = h  # 1/16

        h = self.score_pool3(x2) # 1/16

        h = upscore2 + h  # 1/16
        h = self.upscore2(h) # 1/8
        h = self.upscoreconv4(h) # 1/8

        upscore_pool4 = h  # 1/8

        h = self.score_pool4(x1) # 1/8


        h = upscore_pool4 + h  # 1/8

        h = self.upsample_big(h)


        return h'''
