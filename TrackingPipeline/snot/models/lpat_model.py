# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from snot.core.config_lpat import cfg
from snot.models.backbone.alexnet import AlexNet_hift
from snot.models.lpat.utile import LOGO
import numpy as np



class ModelBuilderLPAT(nn.Module):
    def __init__(self):
        super(ModelBuilderLPAT, self).__init__()

        self.backbone = AlexNet_hift().cuda()
        self.grader=LOGO(cfg).cuda()
  
        
    def template(self, z):
        with t.no_grad():
            zf = self.backbone(z)
    
            self.zf=zf
            
       # self.zf1=zf1

    
    def track(self, x):
        with t.no_grad():
            
            xf = self.backbone(x)  
            loc,cls2,cls3=self.grader(xf,self.zf)

            return {

                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcenter(self,mapp):

        def con(x):
            return x*143
        
        size=mapp.size()[3]
        #location 
        x=np.tile((16*(np.linspace(0,size-1,size))+63)-287//2,size).reshape(-1)
        y=np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2,size).reshape(-1)
        shap=con(mapp).cpu().detach().numpy()
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))

        # xx=xx.reshape(-1,1).repeat(repeats=cfg.TRAIN.BATCH_SIZE,axis=1)
        # yy=yy.reshape(-1,1).repeat(repeats=cfg.TRAIN.BATCH_SIZE,axis=1)

        # w=np.abs(shap[:,2,yy,xx]*143)
        # h=np.abs(shap[:,3,yy,xx]*143)
        # x=x+shap[:,0,yy,xx]*80
        # y=y+shap[:,1,yy,xx]*80
        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2
        y=y-shap[:,2,yy,xx]+h/2
        # w=np.abs(shap[:,0,yy,xx]*143)
        # h=np.abs(shap[:,1,yy,xx]*143)
        
        
        # w=shap[:,0,yy,xx]
        # h=shap[:,1,yy,xx]
        
        anchor=np.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4))

        anchor[:,:,0]=x+287//2
        anchor[:,:,1]=y+287//2
        anchor[:,:,2]=w
        anchor[:,:,3]=h


        return anchor
    def getcentercuda(self,mapp):

        def con(x):
            return x*143
        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-287//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*143
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+287//2
        y=y-shap[:,2,yy,xx]+h/2+287//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
  