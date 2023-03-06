import torch
import torch.nn as nn
import math

import numpy as np
from .ground_perception_model import UNet, UTNet



from torchvision import transforms

import time as tt


class enhancer(nn.Module):

    def __init__(self):
        super(enhancer, self).__init__()

        self.relu = nn.ReLU(inplace=True) 

        number_f = 32

        # light distribution
        self.ldnet = UNet()

        # semantics perception
        self.spnet = UTNet()
        
        # auto-enhance
        self.e_conv4 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,8,3,1,1,bias=True) 
        self.e_conv8 = nn.Conv2d(number_f*2,8,3,1,1,bias=True) 
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def ld2guid(self, x):
        b, c, w, h = x.size()
        g = x.view(b, c, -1)


        # t0=tt.time()		

        mean = g.mean(2, keepdim = True)
        var = ((g - mean) ** 2).mean(2, keepdim = True)
        # t1=tt.time()-t0

        g = (g - mean) / (var + 1e-6).sqrt()
        # t2=tt.time()-t0-t1

        g=1-nn.functional.softmax(g,dim=2)
        # t3=tt.time()-t0-t1-t2

        return g.view(b, c, w, h)
        
    def forward(self, x):
        t0=tt.time()

        x_ld,out3,out_1,out_2,out_3 = self.ldnet(x)
        t1=tt.time()-t0

        # x_ld = self.ld2guid(out)
        t2=tt.time()-t1-t0

        # x1 = self.relu(self.e_conv1(x))
        # x2 = self.relu(self.e_conv2(x1))
        t3=tt.time()-t1-t2-t0

        # b, c, w, h = x2.size()
        x_sp,out_s,out2 = self.spnet(x)
        t4=tt.time()-t1-t2-t3-t0

        x3 = x_sp+x_ld
        t5=tt.time()-t1-t2-t3-t4-t0

        x4 = self.relu(self.e_conv4(torch.cat([x_sp,x3],1)))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x4,x5],1)))
        x_n = torch.tanh(self.e_conv8(torch.cat([x4,x5],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 1, dim=1)
        n1,n2,n3,n4,n5,n6,n7,n8 = torch.split(x_n, 1, dim=1)
        t6=tt.time()-t1-t2-t3-t4-t5-t0

        x = (x-n1) * (r1+1)
        x = (x-n2) * (r2+1)
        x = (x-n3) * (r3+1)
        enhance_image_1 = (x-n4) * (r4+1)
        x = (enhance_image_1-n5) * (r5+1)
        x = (x-n6) * (r6+1)
        x = (x-n7) * (r7+1)
        enhance_image = (x-n8) * (r8+1)
        t7=tt.time()-t1-t2-t3-t4-t5-t6-t0
        
        r = torch.cat([(r1+1),(r2+1),(r3+1),(r4+1),(r5+1),(r6+1),(r7+1),(r8+1)],1)
        n = torch.cat([n1,n2,n3,n4,n5,n6,n7,n8],1)
        return enhance_image,r, n, x_ld, x_sp

    # def test2forward(self, x):
    #     t0=tt.time()

    #     out,out3,out_1,out_2,out_3 = self.ldnet(x)
    #     t1=tt.time()-t0

    #     x_ld = self.ld2guid(out)
    #     t2=tt.time()-t1-t0

    #     # x1 = self.relu(self.e_conv1(x))
    #     # x2 = self.relu(self.e_conv2(x1))
    #     t3=tt.time()-t1-t2-t0

    #     # b, c, w, h = x2.size()
    #     x_sp,out_s,out2 = self.spnet(x)
    #     t4=tt.time()-t1-t2-t3-t0

    #     x3 = x_sp*x_ld+x_sp
    #     t5=tt.time()-t1-t2-t3-t4-t0

    #     x4 = self.relu(self.e_conv4(torch.cat([x_sp,x3],1)))
    #     x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
    #     x_r = torch.tanh(self.e_conv7(torch.cat([x4,x5],1)))
    #     x_n = torch.tanh(self.e_conv8(torch.cat([x4,x5],1)))
    #     r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 1, dim=1)
    #     n1,n2,n3,n4,n5,n6,n7,n8 = torch.split(x_n, 1, dim=1)
    #     t6=tt.time()-t1-t2-t3-t4-t5-t0

    #     x = (x-n1) * (r1+1)
    #     x = (x-n2) * (r2+1)
    #     x = (x-n3) * (r3+1)
    #     enhance_image_1 = (x-n4) * (r4+1)
    #     x = (enhance_image_1-n5) * (r5+1)
    #     x = (x-n6) * (r6+1)
    #     x = (x-n7) * (r7+1)
    #     enhance_image = (x-n8) * (r8+1)
    #     t7=tt.time()-t1-t2-t3-t4-t5-t6-t0
        
    #     r = torch.cat([(r1+1),(r2+1),(r3+1),(r4+1),(r5+1),(r6+1),(r7+1),(r8+1)],1)
    #     n = torch.cat([n1,n2,n3,n4,n5,n6,n7,n8],1)
    #     return enhance_image,r, n,x_sp,out

