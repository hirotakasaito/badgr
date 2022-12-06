import torch
import torch.nn as nn
from cnn_finetune import make_model

from encoder import Encoder

def set_encoder(name,is_pretrained):
    if(name=='resnet18'):
        net = make_model('resnet18', num_classes=10, pretrained=is_pretrained)
        if(is_pretrained):
            for param in resnet.parameters():
                param.requires_grad = False
        encoder = nn.Sequential(*list(resnet.children())[:-1])

    if(name=='resnet34'):
        net = make_model('resnet34', num_classes=10, pretrained=is_pretrained)
        if(is_pretrained):
            for param in net.parameters():
                param.requires_grad = False
        encoder = nn.Sequential(*list(net.children())[:-1])
    
    if(name=='resnet50'):
        net = make_model('resnet50', num_classes=10, pretrained=is_pretrained)
        if(is_pretrained):
            for param in net.parameters():
                param.requires_grad = False
        encoder = nn.Sequential(*list(net.children())[:-1])
    
    if(name=='mobilenet_v2'):
        net = make_model('mobilenet_v2', num_classes=10, pretrained=is_pretrained)
        if(is_pretrained):
            for param in net.parameters():
                param.requires_grad = False
        encoder = nn.Sequential(*list(net.children())[:-1])
    
    else:
        encoder = Encoder()
        if(is_pretrained):
            for param in encoder.parameters():
                param.requires_grad = False
    
    return encoder