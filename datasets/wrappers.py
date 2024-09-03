import numpy as np
from matplotlib import pyplot as plt
import re
import cv2
import torch
import models
import random
import json

import albumentations as A
import matplotlib.pyplot as plt
import kornia as K
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from datasets import register
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

@register('Ocr_images_lp')
class Ocr_images_lp(Dataset):
    def __init__(
            self,
            alphabet,
            k,
            imgW,
            imgH,
            aug,
            image_aspect_ratio,
            background,
            with_lr = False,
            test = False,
            dataset=None,
            ):
        
        self.imgW = imgW
        self.imgH = imgH
        self.aug = True
        self.ar = image_aspect_ratio
        self.background = eval(background)
        self.test = test
        self.dataset = dataset
        self.k = k
        self.alphabet = alphabet
        self.with_lr = with_lr
        self.transformImg = np.array([
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True, p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, always_apply=True, p=1.0),
            
                A.Posterize(num_bits=4, always_apply=True, p=1.0),
                A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=True, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1.0),
                
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True, p=1.0),
                A.PixelDropout(dropout_prob=0.01, per_channel=True, drop_value=0, mask_drop_value=None, always_apply=True, p=1.0),
                A.ImageCompression(quality_lower=90, quality_upper=100, always_apply=True, p=1.0),
                None
            ])
            
    def Open_image(self, img, cvt=True):
        img = cv2.imread(img)
        if cvt is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def padding(self, img, min_ratio, max_ratio, color = (0, 0, 0)):
        img_h, img_w = np.shape(img)[:2]

        border_w = 0
        border_h = 0
        ar = float(img_w)/img_h

        if ar >= min_ratio and ar <= max_ratio:
            return img, border_w, border_h

        if ar < min_ratio:
            while ar < min_ratio:
                border_w += 1
                ar = float(img_w+border_w)/(img_h+border_h)
        else:
            while ar > max_ratio:
                border_h += 1
                ar = float(img_w)/(img_h+border_h)

        border_w = border_w//2
        border_h = border_h//2

        img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
        
        return img, border_w, border_h
    
    def extract_plate_numbers(self, file_path, pattern):
        # List to store extracted plate numbers
        plate_numbers = []
        
        # Open the text file
        with open(file_path, 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Search for the pattern in the current line
                matches = re.search(pattern, line)
                # If a match is found
                if matches:
                    # Extract the matched string
                    plate_number = matches.group(1)
                    # Add the extracted plate number to the list
                    plate_numbers.append(plate_number)
        
        # Return the list of extracted plate numbers
        return plate_numbers[0]

    
    def collate_fn(self, datas):
        imgs = []
        gts = []
        
        for item in datas:
            if self.with_lr:
                img = self.Open_image(item["img"].replace('HR', 'LR') if random.random() < 0.5 else item["img"])
            else:
                img = self.Open_image(item['img'])
            if self.aug is True:
                augment = np.random.choice(self.transformImg, replace=True)
                if augment is not None:
                    img = augment(image=img)["image"]
            # print(self.ar)
            img, _, _ = self.padding(img, self.ar-0.15, self.ar+0.15, self.background)    
            img = resize_fn(img, (self.imgH, self.imgW))
            imgs.append(img)
            gt = self.extract_plate_numbers(Path(item["img"]).with_suffix('.txt'), pattern=r'plate: (\w+)')
            gts.append(gt)  
        
        batch_txts = gts
        
        batch_imgs = torch.stack(imgs)
        
        return {
            'img': batch_imgs, 'text': batch_txts
        }
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
