# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "SiamAPN_alexnet"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Attention set
# ------------------------------------------------------------------------ #
__C.ATTE = CN()

# Masksize
__C.ATTE.MASK_SIZE = 5


# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()
__C.TRAIN.CONTINUE = True
# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6

__C.TRAIN.apnchannel = 256

__C.TRAIN.clsandlocchannel = 256

__C.TRAIN.groupchannel = 32

__C.TRAIN.THR_LOW = 0.3

__C.TRAIN.NEG_NUM = 16

__C.TRAIN.POS_NUM = 16

__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.PR = 1

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 3.0

__C.TRAIN.IND_WEIGHT =0.2

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 287 #255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 21 #25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = 1

__C.TRAIN.LARGER=2.0

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 30

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 100

__C.TRAIN.NUM_GPU = 2

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.w1=1.0

__C.TRAIN.w2=1.0

__C.TRAIN.w3=1.0

__C.TRAIN.w4=1.0

__C.TRAIN.w5=1.0

__C.TRAIN.w6=1.0

__C.TRAIN.range=2.0


__C.TRAIN.MASK_WEIGHT = 1

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.2  

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'GOT', 'YOUTUBEBB')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'train_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'train_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'train_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'train_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'train_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'train_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1


__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = 'train_dataset/got10k/crop511'         # GOT dataset path
__C.DATASET.GOT.ANNO = 'train_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 100000

__C.DATASET.VIDEOS_PER_EPOCH = 600000



# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'alexnet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = 'back.pth'

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer3', 'layer4', 'layer5']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# # ------------------------------------------------------------------------ #
# # Anchor options
# # ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# # Anchor stride
__C.ANCHOR.STRIDE = 16


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamAPNtracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

__C.TRACK.w1=1.2 

__C.TRACK.w2=1.0

__C.TRACK.w3=1.6 

__C.TRACK.LARGER=1.4
# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

__C.TRACK.STRIDE = 8 

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127
