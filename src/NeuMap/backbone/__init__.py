from statistics import mode
from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4

def build_backbone(config):

    return ResNetFPN_8_2(config['resnetfpn'], 3 if config['rgb'] else 1)
       