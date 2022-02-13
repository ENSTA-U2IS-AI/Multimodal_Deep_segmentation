import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x = self.classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x
    '''def compute_features(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            #print('features =',type(features),features.keys(),features['low_level'].size(),features['out'].size())
   
            xembedding = F.interpolate(features['low_level'], size=input_shape, mode='bilinear', align_corners=False)
            return xembedding'''


class _SimpleSegmentationModel_DM3(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel_DM3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x, xembedding,conf = self.classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x

    def compute_features(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x, xembedding,conf = self.classifier(features)
            xembedding = F.interpolate(xembedding, size=input_shape, mode='bilinear', align_corners=False)
            return xembedding,conf

    def loss_kmeans(self):
        param = self.classifier.DMlayer.omega
        loss = torch.mean(torch.cdist(param, param))
        return loss


class _SimpleSegmentationModel_DM(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel_DM, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x,xembedding= self.classifier(features)
            x= F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x
    def compute_features(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x,xembedding = self.classifier(features)
            xembedding = F.interpolate(xembedding, size=input_shape, mode='bilinear', align_corners=False)
            return xembedding
    def loss_kmeans(self):
        param = self.classifier.DMlayer.omega
        loss = torch.mean(torch.cdist(param,param))
        return loss

'''class _SimpleSegmentationModel_DM2(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel_DM2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x, xembedding = self.classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x

    def compute_features(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x, dic = self.classifier(features)
            xembeddingbef = dic['bef']
            xembeddingaft = dic['aft']
            xembeddingaft = F.interpolate(xembeddingaft, size=input_shape, mode='bilinear', align_corners=False)
            xembeddingbef = F.interpolate(xembeddingbef, size=input_shape, mode='bilinear', align_corners=False)
            return {'bef':xembeddingbef,'aft':xembeddingaft}

    def loss_kmeans(self):
        param = self.classifier.DMlayer.omega
        loss = torch.mean(torch.cdist(param, param))
        return loss'''


class _SimpleSegmentationModel_DM2(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel_DM2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x, xembedding = self.classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x

    def compute_features(self, x):
        with torch.cuda.amp.autocast():
            input_shape = x.shape[-2:]
            features = self.backbone(x)
            x,xembedding = self.classifier(features)
            #xembedding = F.interpolate(xembedding, size=input_shape, mode='bilinear', align_corners=False)
            return xembedding
    def loss_kmeans(self):
        param = self.classifier.DMlayer.omega
        loss = torch.mean(torch.cdist(param, param))
        return loss

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

