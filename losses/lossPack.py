import re
import cv2
import yaml
import torch
import kornia
import losses
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from keras.models import Model
from losses import register, make
from kornia.losses import SSIMLoss
from torchvision import transforms
from torch.autograd import Variable
from memory_profiler import profile
from matplotlib import pyplot as plt
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
import gc

@register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss = nn.CrossEntropyLoss(weight=self.weight, 
                                        size_average=self.size_average, 
                                        ignore_index=self.ignore_index, 
                                        reduce=self.reduce, 
                                        reduction=self.reduction, 
                                        label_smoothing=self.label_smoothing)
        
    def forward(self, v1, v2):
        return self.loss(v1, v2)
