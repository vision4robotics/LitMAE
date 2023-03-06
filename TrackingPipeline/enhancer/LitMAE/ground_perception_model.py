import torch
import torch.nn as nn
import torch.nn.functional as F
import time as tt

from timm.models.vision_transformer import PatchEmbed, Block, VisionTransformer
from .pos_embed_model import get_2d_sincos_pos_embed
 
 
def X1conv(in_channel,out_channel,kernel_size=3, stride=1, padding=1):
    """连续两个3*3卷积"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        )
def X2conv(in_channel,out_channel,kernel_size=3, stride=1, padding=1):
    """连续两个3*3卷积"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU())
 
class DownsampleLayer(nn.Module):
    """
    下采样层
    """
    def __init__(self,in_channel,out_channel,kernel_size=2):
        super(DownsampleLayer, self).__init__()
        self.x2conv=X2conv(in_channel,out_channel)
        self.pool=nn.MaxPool2d(kernel_size=kernel_size,ceil_mode=True)
 
    def forward(self,x):
        """
        :param x:上一层pool后的特征
        :return: out_1转入右侧（待拼接），out_1输入到下一层，
        """
        out_1=self.x2conv(x)
        out=self.pool(out_1)
        return out_1,out
 
class UpSampleLayer(nn.Module):
    """
    上采样层
    """
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=2,padding=1):
 
        super(UpSampleLayer, self).__init__()
        self.x2conv = X2conv(in_channel, out_channel)
        self.upsample=nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//2,kernel_size=kernel_size,stride=stride,padding=padding)
 
    def forward(self,x,out):
        '''
        :param x: decoder中：输入层特征，经过x2conv与上采样upsample，然后拼接
        :param out:左侧encoder层中特征（与右侧上采样层进行cat）
        :return:
        '''
        x=self.x2conv(x)
        x=self.upsample(x)
 
        # x.shape中H W 应与 out.shape中的H W相同
        if (x.size(2) != out.size(2)) or (x.size(3) != out.size(3)):
            # 将右侧特征H W大小插值变为左侧特征H W大小
            x = F.interpolate(x, size=(out.size(2), out.size(3)),
                            mode="bilinear", align_corners=True)
 
 
        # Concatenate(在channel维度)
        cat_out = torch.cat([x, out], dim=1)
        return cat_out

class DownsampleLayer1(nn.Module):
    """
    下采样层
    """
    def __init__(self,in_channel,out_channel,kernel_size=2):
        super(DownsampleLayer1, self).__init__()
        self.x1conv=X1conv(in_channel,out_channel)
        self.pool=nn.MaxPool2d(kernel_size=kernel_size,ceil_mode=True)
 
    def forward(self,x):
        """
        :param x:上一层pool后的特征
        :return: out_1转入右侧（待拼接），out_1输入到下一层，
        """
        out_1=self.x1conv(x)
        out=self.pool(out_1)
        return out_1,out
 
class UpSampleLayer1(nn.Module):
    """
    上采样层
    """
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=2,padding=1):
 
        super(UpSampleLayer1, self).__init__()
        self.x1conv = X1conv(in_channel, out_channel)
        self.upsample=nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//2,kernel_size=kernel_size,stride=stride,padding=padding)
 
    def forward(self,x,out):
        '''
        :param x: decoder中：输入层特征，经过x2conv与上采样upsample，然后拼接
        :param out:左侧encoder层中特征（与右侧上采样层进行cat）
        :return:
        '''
        x=self.x1conv(x)
        x=self.upsample(x)
 
        # x.shape中H W 应与 out.shape中的H W相同
        if (x.size(2) != out.size(2)) or (x.size(3) != out.size(3)):
            # 将右侧特征H W大小插值变为左侧特征H W大小
            x = F.interpolate(x, size=(out.size(2), out.size(3)),
                            mode="bilinear", align_corners=True)
 
 
        # Concatenate(在channel维度)
        cat_out = torch.cat([x, out], dim=1)
        return cat_out

            
class UTNet(nn.Module):
    """
    UNet模型,num_classes为分割类别数
    """
    def __init__(self):
        super(UTNet, self).__init__()
        #embeding
        self.embed=X2conv(3,12)
        #下采样
        self.d1=DownsampleLayer1(12,24) #3-64
        #self.d2=DownsampleLayer(12,24)#64-128
        #self.d3=DownsampleLayer(24,48)#128-256
        #self.d4=DownsampleLayer(256,512)#256-512
        
        #语义
        embed_dim = 192
        num_heads = 8
        mlp_ratio = 1
        self.patch = nn.Conv2d(24,192,8,8,0)
        self.sp =  Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
        self.unpatch =nn.Linear(192,384)
        self.pick = X1conv(6,24)
        #上采样
        self.u1=UpSampleLayer1(24,48)#512-1024-512
        #self.u2=UpSampleLayer(96,48)#1024-512-256
        #self.u3=UpSampleLayer(48,24)#512-256-128
        #self.u4=UpSampleLayer(256,128)#256-128-64
 
        #输出:经过一个二层3*3卷积 + 1个1*1卷积
        self.x2conv=X1conv(48,32)
        #self.final_conv=nn.Conv2d(12,24,kernel_size=1)  # 最后一个卷积层的输出通道数为分割的类别数
        self._initialize_weights()
    def unpatchify(self, x, w, h, p=8, c=384):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], w, h, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, w * p, h * p))
        return imgs
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
 
    def forward(self,x):
        #embeding
        x=self.embed(x)
        # 下采样层
        out_1,out1=self.d1(x)
        # out_2,out2=self.d2(out1)
        # out_3,out3=self.d3(out2)
        # #out_4,out4=self.d4(out3)

        #semantics geeeeeeeeeeeeeeeeeeeeeeet
        t0=tt.time()
        out_s=self.patch(out1)
        b,c,w,h=out_s.size()
        out_s=self.unpatch(self.sp(out_s.view(b, c, -1).permute(0, 2, 1)))
        out_s=self.pick(self.unpatchify(out_s, w, h, 8, 6))
        t1=tt.time()-t0
        # 上采样层 拼接
        out2=self.u1(out_s,out_1)
        # out5=self.u2(out4,out_2)
        # out6=self.u3(out5,out_1)
        #out8=self.u4(out7,out_1)
 
        # 最后的三层卷积
        out=self.x2conv(out2)
        #out=self.final_conv(out)
        return out,out_s,out2



class UNet(nn.Module):
    """
    UNet模型,num_classes为分割类别数
    """
    def __init__(self,num_classes=32):
        super(UNet, self).__init__()
        #下采样
        self.d1=DownsampleLayer(3,12) #3-64
        self.d2=DownsampleLayer(12,24)#64-128
        self.d3=DownsampleLayer(24,48)#128-256
        #self.d4=DownsampleLayer(256,512)#256-512
 
        #上采样
        self.u1=UpSampleLayer(48,96)#512-1024-512
        self.u2=UpSampleLayer(96,48)#1024-512-256
        self.u3=UpSampleLayer(48,24)#512-256-128
        #self.u4=UpSampleLayer(256,128)#256-128-64
 
        #输出:经过一个二层3*3卷积 + 1个1*1卷积
        self.x2conv=X2conv(24,24)
        self.final_conv=nn.Conv2d(24,num_classes,kernel_size=1)  # 最后一个卷积层的输出通道数为分割的类别数
        self._initialize_weights()
 
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
 
    def forward(self,x):
        # 下采样层
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        #out_4,out4=self.d4(out3)
 
        # 上采样层 拼接
        out4=self.u1(out3,out_3)
        out5=self.u2(out4,out_2)
        out6=self.u3(out5,out_1)
        #out8=self.u4(out7,out_1)
 
        # 最后的三层卷积
        out=self.x2conv(out6)
        out=self.final_conv(out)
        return out,out3,out_1,out_2,out_3
    

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.u1=UpSampleLayer(512,1024)#512-1024-512
#         self.u2=UpSampleLayer(1024,512)#1024-512-256
#         self.u3=UpSampleLayer(512,256)#512-256-128
#         self.u4=UpSampleLayer(256,128)#256-128-64
#         self.x2conv=X2conv(128,64)
#         self.final_conv=nn.Conv2d(64,32,kernel_size=1)  # 最后一个卷积层的输出通道数为分割的类别数
#     def forward(self,out_1,out_2,out_3,out_4,out4):
#         out5=self.u1(out4,out_4)
#         out6=self.u2(out5,out_3)
#         out7=self.u3(out6,out_2)
#         out8=self.u4(out7,out_1)
#         out=self.x2conv(out8)
#         out=self.final_conv(out)
#         return out


 
# if __name__ == "__main__":
#     img = torch.randn((2, 3, 360, 480))  # 正态分布初始化
#     model = UNet(num_classes=3)
#     output, ou, out_1,out_2,out_3,out_4 = model(img)
#     net = Net()
#     x = net(out_1,out_2,out_3,out_4,ou)
#     print(output.shape)
#     print(x.shape)