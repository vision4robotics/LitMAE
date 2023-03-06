import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from MAEmphasizer.model import model_build as MAEmphasizer

import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import pdb
import cv2
from torch.nn import functional as F

os.environ['CUDA_VISIBLE_DEVICES']='1'


def lowlight(image_path, enhancer):
	# os.environ['CUDA_VISIBLE_DEVICES']='1'
	data_lowlight = Image.open(image_path)
 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)
	# data_lowlight = F.interpolate(data_lowlight, size = (256,256), mode = 'bilinear')


	start = time.time()
	enhanced_image,a,n, x_masked, x_sp = enhancer(data_lowlight)
	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	torchvision.utils.save_image(enhanced_image, result_path)

	# result_path = image_path.replace('_data/','_data/a/')
	# if not os.path.exists(result_path.replace('/'+result_path.split("/")[-1],'')):
	# 	os.makedirs(result_path.replace('/'+result_path.split("/")[-1],''))
	# torchvision.utils.save_image(a[:,1,:,:], result_path)

	# result_path = image_path.replace('_data/','_data/n/')
	# if not os.path.exists(result_path.replace('/'+result_path.split("/")[-1],'')):
	# 	os.makedirs(result_path.replace('/'+result_path.split("/")[-1],''))
	# torchvision.utils.save_image(n[:,1,:,:], result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = './data/test_data/'
		file_list = os.listdir(filePath)
		model = MAEmphasizer().cuda()
		model.load_state_dict(torch.load('snapshots/model_1.pth'))
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				print(image)
				lowlight(image, model.enhancer)

