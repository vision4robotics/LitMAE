from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snot.core.config_hift import cfg
from snot.models.backbone.alexnet import AlexNet_hift
from snot.models.hift.utile import hiftmodule


class ModelBuilderHiFT(nn.Module):
    def __init__(self):
        super(ModelBuilderHiFT, self).__init__()

        self.backbone = AlexNet_hift().cuda()
        self.grader=hiftmodule(cfg).cuda()
       
        
    def template(self, z):
        with torch.no_grad():
            zf = self.backbone(z)
    
            self.zf=zf

    
    def track(self, x):
        with torch.no_grad():
            
            xf = self.backbone(x)  
            loc,cls1,cls2=self.grader(xf,self.zf)

            return {
                'cls1': cls1,
                'cls2': cls2,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls


    def getcentercuda(self,mapp):


        def dcon(x):
           x[torch.where(x<=-1)]=-0.99
           x[torch.where(x>=1)]=0.99
           return (torch.log(1+x)-torch.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=torch.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        y=torch.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*(cfg.TRAIN.SEARCH_SIZE//2)
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+cfg.TRAIN.SEARCH_SIZE//2
        y=y-shap[:,2,yy,xx]+h/2+cfg.TRAIN.SEARCH_SIZE//2

        anchor=torch.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
