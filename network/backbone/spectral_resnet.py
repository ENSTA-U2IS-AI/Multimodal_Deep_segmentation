import torch
from torch import Tensor
import torch.nn as nn
#from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from network.backbone.layers import spectral_norm_conv, spectral_norm_fc, SpectralBatchNorm2d
import math
import torch.nn.functional as F
import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        input_size1: int = 32,
        input_size2: int = 32,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spectral_normalization: bool = True,
        coeff:int =3,
        n_power_iterations: int=1,
        batchnorm_momentum: float=0.1,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size1,input_size2, in_c, out_c, stride=1,dilation=1):
            #padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=dilation, bias=False,dilation=dilation)

            if not spectral_normalization:
                return conv


            # Otherwise use spectral norm conv, with loose bound
            input_dim = (in_c, input_size1,input_size2)
            wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations)

            return wrapped_conv
        self.wrapped_conv = wrapped_conv
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1


        self.input_size1=input_size1
        self.input_size2=input_size2
        if stride != 1 :
            input_size1 = input_size1* stride  #  need to check that line for strie != 2
            input_size2 = input_size2* stride  #  need to check that line for strie != 2
        self.conv1 = wrapped_conv(input_size1,input_size2,inplanes, planes, stride)# (input_size, inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = wrapped_bn(planes)
        self.relu = nn.ReLU(inplace=True)


        self.conv2 = wrapped_conv(self.input_size1,self.input_size2,planes, planes)# wrapped_conv(input_size, planes, planes, kernel_size=3, stride=1,padding=1)
        self.bn2 = wrapped_bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        input_size1: int = 32,
        input_size2: int = 32,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        spectral_normalization: bool = True,
        coeff:int =3,
        n_power_iterations: int=1,
        batchnorm_momentum: float=0.1,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size1,input_size2, in_c, out_c, kernel_size=3, stride=1,dilation=1,padding=1):
            #padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,dilation=dilation)

            if not spectral_normalization:
                return conv


            # Otherwise use spectral norm conv, with loose bound
            input_dim = (in_c, input_size1, input_size2)
            wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations)

            return wrapped_conv
        self.wrapped_conv = wrapped_conv
        self.input_size1=input_size1
        self.input_size2=input_size2

        if stride != 1 :
            input_size1 = input_size1* stride  #  need to check that line for strie != 2
            input_size2 = input_size2* stride  #  need to check that line for strie != 2
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = wrapped_conv(input_size1,input_size2,inplanes, width,1, stride=1,padding=0) #nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False) #conv1x1(inplanes, width)
        self.bn1 = wrapped_bn(width)
        self.conv2 = wrapped_conv(input_size1,input_size2,width, width,3, stride,dilation=dilation)#conv3x3(width, width, stride, groups, dilation)
        self.bn2 = wrapped_bn(width)
        self.conv3 =  wrapped_conv(self.input_size1,self.input_size2,width, planes * self.expansion,1, stride=1,padding=0) # nn.Conv2d(width,planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = wrapped_bn(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        input_size1: int,
        input_size2: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        spectral_normalization: bool = True,
        zero_init_residual: bool = False,
        groups: int = 1,
        coeff:int =3,
        n_power_iterations: int=1,
        batchnorm_momentum: float=0.1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size1,input_size2, in_c, out_c, kernel_size, stride,padding):
            #padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size1, input_size2)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.conv1 = wrapped_conv(input_size1,input_size2, 3, self.inplanes, kernel_size=7, stride=2, padding=3)
        input_size1 = (input_size1 - 1) // 2 + 1
        input_size2 = (input_size2 - 1) // 2 + 1

        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = wrapped_bn(self.inplanes)#norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        input_size1 = (input_size1 - 1) // 2 + 1
        input_size2 = (input_size2 - 1) // 2 + 1
        #print('input_size apres layer0',input_size)

        self.layer1, input_size1,input_size2 = self._make_layer(input_size1,input_size2,block, 64, layers[0])

        self.layer2, input_size1,input_size2 = self._make_layer(input_size1,input_size2,block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3, input_size1,input_size2 = self._make_layer(input_size1,input_size2, block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4, input_size1,input_size2 = self._make_layer(input_size1,input_size2, block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcn = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, input_size1,input_size2, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:


            downsample = nn.Sequential(
                self.wrapped_conv(input_size1,input_size2, self.inplanes,  planes * block.expansion, kernel_size=1, stride=stride, padding=0),
                self.wrapped_bn(planes * block.expansion),
            )
            input_size1 = (input_size1 - 1) // stride + 1
            input_size2 = (input_size2 - 1) // stride + 1

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation,input_size1,input_size2, norm_layer))
        self.inplanes = planes * block.expansion
        #if stride != 1 : input_size = (input_size - 1) // stride + 1
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,input_size1=input_size1,input_size2=input_size2,
                                norm_layer=norm_layer))


        return nn.Sequential(*layers), input_size1,input_size2

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('maxpool',x.size())
        x = self.layer1(x)
        #print(x.size())
        #print('layer1',x.size())
        x = self.layer2(x)
        #print('layer2',x.size())
        x = self.layer3(x)
        #print('layer3',x.size())
        x = self.layer4(x)
        #print('layer4',x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fcn(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    input_size1: int,
    input_size2: int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(input_size1,input_size2,block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model


def resnet18(input_size1: int, input_size2: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',input_size1, input_size2, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(input_size1: int, input_size2: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34',input_size1, input_size2, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(input_size1: int, input_size2: int,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50',input_size1, input_size2, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet101(input_size1: int, input_size2: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101',input_size1, input_size2, Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)



def resnet152(input_size1: int, input_size2: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152',input_size1, input_size2, Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
