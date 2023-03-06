import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import numpy as np
from torchvision import transforms

from .enhancer_model import enhancer
from .Decoder_model import MaskedAutoencoderViT as decoder
from .alexnet import AlexNet as backbone


import time
class model_build(nn.Module):

    def __init__(self):
        super(model_build, self).__init__()
        self.mask_ratio = 0.1
        self.enhancer = enhancer()
        self.decoder = decoder(img_size=22, patch_size=1, in_chans=3,
                 embed_dim=256, 
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
        self.backbone = backbone() 

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate mask
        mask0 = torch.zeros(N, L, device=x.device)
        mask1 = torch.ones(N, L, device=x.device)
        mask = torch.where(ids_restore < len_keep, mask1, mask0)

        # generate x_masked
        mask_x = mask.unsqueeze(-1).repeat(1, 1, D)
        x0=mask0.unsqueeze(-1).repeat(1,1,D)
        x_masked = torch.where(mask_x > 0, x, x0)

        return x_masked, mask, ids_restore
        
    def forward(self, x):
        b1, c1, h1, w1 = x.size()
        x_masked, mask, ids_restore = self.random_masking(x.view(b1,c1,-1).permute(0, 2, 1),self.mask_ratio)
        x_masked = x_masked.permute(0, 2, 1).view(b1, c1, h1, w1)

        enhanced_image,A,N , x_ld, x_sp= self.enhancer(x_masked)

        feature = self.backbone(enhanced_image)

        b, c, h, w = feature.size()
        p=1
        x_c = F.interpolate(x, size = (h*p,w*p), mode = 'bilinear')

        feature = feature.view(b,c,-1).permute(0, 2, 1)

        loss, pred = self.decoder(x_c, feature, mask)
        return enhanced_image,A,N,loss, pred, mask.view(b1, h1, w1), x_masked, x_sp


