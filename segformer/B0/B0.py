import torch.optim

from segformer.MIT import *

# model settings
class SegFormerB0(nn.Module):
    def __init__(self,in_channels=[32, 64, 160, 256],
        feature_strides=[4, 8, 16, 32],
        decoder_params=dict(embed_dim=256),
        num_classes=19,
        dropout_ratio=0.1,interpolate=True):
        super().__init__()
        self.encoder = mit_b0()
        self.head = SegFormerHead(in_channels,feature_strides,decoder_params,num_classes,dropout_ratio)
        self.interpolate = interpolate

    def forward(self, x):
        features = self.encoder(x)
        features = self.head(features)
        if self.interpolate:
            features = F.interpolate(features, size=x.shape[2:], mode='bilinear', align_corners=False)
            #features = F.interpolate(features, size=(x.shape[2] // 2, x.shape[3] // 2), mode='bilinear',
            #                         align_corners=False)
        return features



