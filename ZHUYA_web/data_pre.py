# coding:utf8
import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Tooth(data.Dataset):

    def __init__(self, tooth_path, transforms=None):
        self.path = tooth_path
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img = Image.open(self.path)
        data = img.convert('RGB')
        img_data = self.transforms(data)
        return img_data

    def __len__(self):
        return 1
