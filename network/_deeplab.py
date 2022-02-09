import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .utils import _SimpleSegmentationModel, _SimpleSegmentationModel_DM,_SimpleSegmentationModel_DM2


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass
    
class DeepLabV3DM(_SimpleSegmentationModel_DM):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabV3DM2(_SimpleSegmentationModel_DM2):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

'''class DeepLabHeadV3Plus_DM_v2(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_DM_v2, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.DMlayer = Distanceminimi_Layer_learned_old(in_features=256, out_features=22, dist='cos')
        self.bn = nn.BatchNorm2d(256)
        self.lastlayer = nn.Conv2d(22, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        embedding = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        embedding = self.bn(embedding)
        embedding = rearrange(embedding, 'b h n d -> b n d h')
        embedding = self.DMlayer(embedding)
        embedding = torch.squeeze(embedding)
        embedding0 = torch.exp(embedding)  # **2)
        embedding0 = rearrange(embedding0, 'b n d h -> b h n d')

        out = torch.exp(embedding)
        out = rearrange(out, 'b n d h -> b h n d')
        # out =self.bn(out)
        out = self.lastlayer(out)

        return out, embedding0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)'''

'''class DeepLabHeadV3Plus_DM_v2(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_DM_v2, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.DMlayer= Distanceminimi_Layer_learned_old(in_features=128, out_features=22,dist='cos')
        self.bn=nn.BatchNorm2d(128)
        self.lastlayer = nn.Conv2d(22, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        #print('output_feature',output_feature.size(),'////',low_level_feature.size())
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        embedding =self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        #embedding =self.bn(embedding)
        embedding = rearrange(embedding, 'b h n d -> b n d h')
        embedding =self.DMlayer(embedding)
        embedding = torch.squeeze(embedding)
        embedding0 =torch.exp(embedding)#**2)
        embedding0 = rearrange(embedding0, 'b n d h -> b h n d')
        #print('embedding0',embedding0.size())

        out =torch.exp(embedding)
        out = rearrange(out, 'b n d h -> b h n d')
        #print('out',out.size())
        #out =self.bn(out)
        out = self.lastlayer(out)

        return out, embedding0

    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)'''

'''class DeepLabHeadV3Plus_DM_v3(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_DM_v3, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.DMlayer = Distanceminimi_Layer_learned_old(in_features=256, out_features=22, dist='cos')
        self.lastlayer = nn.Conv2d(22, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        embedding0 = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        embedding1 = rearrange(embedding0, 'b h n d -> b n d h')
        embedding1 = self.DMlayer(embedding1)
        embedding1 = torch.squeeze(embedding1)
        #embedding0 = torch.exp(embedding)  # **2)
        #embedding0 = rearrange(embedding0, 'b n d h -> b h n d')

        embedding1 = torch.exp(embedding1)
        embedding1 = rearrange(embedding1, 'b n d h -> b h n d')
        out = self.lastlayer(embedding1)
        # out =torch.exp(embedding)
        # out = rearrange(out, 'b n d h -> b h n d')
        # out =torch.exp(embedding)
        # out =self.bn(embedding)
        # out = self.lastlayer(out)
        # out = self.lastlayer(out)


        return out, {'bef':embedding0,'aft':embedding1}'''



class DeepLabHeadV3Plus_DM_v3(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_DM_v3, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.DMlayer = Distanceminimi_Layer_learned_old(in_features=256, out_features=19, dist='cos')
        self.lastlayer = nn.Conv2d(22, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        embedding = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        embedding = rearrange(embedding, 'b h n d -> b n d h')
        embedding = self.DMlayer(embedding)
        embedding = torch.squeeze(embedding)
        #embedding0 = torch.exp(embedding)  # **2)
        embedding0 = rearrange(embedding, 'b n d h -> b h n d')

        out = torch.sigmoid(embedding)
        out = rearrange(out, 'b n d h -> b h n d')
        out = self.lastlayer(out)
        # out =torch.exp(embedding)
        # out = rearrange(out, 'b n d h -> b h n d')
        # out =torch.exp(embedding)
        # out =self.bn(embedding)
        # out = self.lastlayer(out)
        # out = self.lastlayer(out)


        return out, embedding0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus_DM_v2(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_DM_v2, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.DMlayer = Distanceminimi_Layer_learned_old(in_features=256, out_features=22, dist='cos')
        self.lastlayer = nn.Conv2d(22, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        embedding = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        embedding = rearrange(embedding, 'b h n d -> b n d h')
        embedding = self.DMlayer(embedding)
        embedding = torch.squeeze(embedding)
        embedding0 = torch.exp(embedding)  # **2)
        embedding0 = rearrange(embedding0, 'b n d h -> b h n d')

        out = torch.exp(embedding)
        out = rearrange(out, 'b n d h -> b h n d')
        out = self.lastlayer(out)
        # out =torch.exp(embedding)
        # out = rearrange(out, 'b n d h -> b h n d')
        # out =torch.exp(embedding)
        # out =self.bn(embedding)
        # out = self.lastlayer(out)
        # out = self.lastlayer(out)


        return out, embedding0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadV3Plus_DM(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus_DM, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.DMlayer= Distanceminimi_Layer_learned_old(in_features=256, out_features=22,dist='cos')
        self.bn=nn.BatchNorm2d(22)
        self.lastlayer = nn.Conv2d(22, num_classes, 1)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        embedding =self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        
        embedding = rearrange(embedding, 'b h n d -> b n d h')
        embedding =self.DMlayer(embedding)
        embedding = torch.squeeze(embedding)
        embedding0 =torch.exp(embedding)#**2)
        embedding0 = rearrange(embedding0, 'b n d h -> b h n d')
        
        out =torch.exp(embedding)
        out = rearrange(out, 'b n d h -> b h n d')
        out =self.bn(out)
        out = self.lastlayer(out)
        #out =torch.exp(embedding)
        #out = rearrange(out, 'b n d h -> b h n d')
        #out =torch.exp(embedding)
        #out =self.bn(embedding)
        #out = self.lastlayer(out)
        #out = self.lastlayer(out)
        
        
        '''embedding = rearrange(embedding, 'b h n d -> b n d h')
        embedding =self.DMlayer(embedding)
        embedding = torch.squeeze(embedding)
        embedding0 =torch.exp(embedding**2)#*0.2)
        embedding0 = rearrange(embedding0, 'b n d h -> b h n d')
        
        out =torch.exp(embedding)
        out = rearrange(out, 'b n d h -> b h n d')
        #out =torch.exp(embedding)
        #out =self.bn(embedding)
        #out = self.lastlayer(out)
        out = self.lastlayer(out)
        '''
        return out, embedding0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Distanceminimi_Layer_learned(nn.Module):
    def __init__(self, in_features=0, out_features=0,dist='lin'):
        super(Distanceminimi_Layer_learned_old, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dist=dist
        self.omega = nn.Parameter(torch.Tensor(out_features,in_features))

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.omega, mean=0, std=1)#/self.out_features)

    def forward(self, x):
        prots = self.omega.unsqueeze(0)
        prots=prots.unsqueeze(1)
        prots=prots.unsqueeze(2)
        #x = x.unsqueeze(2)
        #x = x.unsqueeze(4)
        #print('check',x.size(),prots.size() )
        list_out=[]
        if self.dist == 'l2':
            [list_out.append((torch.pow(x-prots[:,:,:,i], 2).sum(-1)).unsqueeze(3)) for i in range(self.out_features)]
        elif self.dist == 'cos':
            [list_out.append((F.cosine_similarity(x,prots[:,:,:,i]  , dim=-1, eps=1e-30)).unsqueeze(3)) for i in range(self.out_features)]

        return torch.stack(list_out,dim=3)

class Distanceminimi_Layer_learned_old(nn.Module):
    def __init__(self, in_features=0, out_features=0,dist='lin'):
        super(Distanceminimi_Layer_learned_old, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dist=dist
        self.omega = nn.Parameter(torch.Tensor(out_features,in_features))

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.omega, mean=0, std=1)#/self.out_features)

    def forward(self, x):
        prots = self.omega.unsqueeze(0)
        prots=prots.unsqueeze(1)
        prots=prots.unsqueeze(2)
        #x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        #print('prots.size()',prots.size(),x.size())

        if self.dist == 'l2':
            x = -torch.pow(x - prots, 2).sum(-1)  # shape [n_query, n_way]
        elif self.dist == 'cos':
            x = F.cosine_similarity(x, prots, dim=-1, eps=1e-30)
        elif self.dist == 'lin':
            x = torch.einsum('izd,zjd->ij', x, prots)

        return x

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeaddrop(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeaddrop, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
