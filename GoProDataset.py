
from collections import Counter
import os
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GoProDataset(Dataset):
    def __init__(self, image_dir, image_filename_pattern, length=224, width = 224):
        
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._blur_image_dir = self._image_dir + "blur"
        self._sharp_image_dir = self._image_dir + "sharp"
        self.length = length ## Resize size
        self.width = width

    def __len__(self):
        return len([entry for entry in os.listdir(self._blur_image_dir) if os.path.isfile(os.path.join(self._blur_image_dir, entry))])

    def __getitem__(self, idx):
        
        name = str(idx)
        if len(name) < 6:
            name = '0' * (6 - len(name)) + name
        img_name = self._image_filename_pattern.format(name)
        _img_blur = Image.open(os.path.join(self._blur_image_dir, img_name)).convert('RGB')
        _img_sharp = Image.open(os.path.join(self._sharp_image_dir, img_name)).convert('RGB')
        
        width, height = _img_blur.size
        max_dim = max(width, height)
        preprocessing = transforms.Compose([
            transforms.CenterCrop((672, 672)),
            # transforms.Pad((0, 0, max_dim - width, max_dim - height)),
            transforms.Resize((self.length, self.width)),
            transforms.ToTensor()])
        img_blur = preprocessing(_img_blur)
        img_sharp = preprocessing(_img_sharp)
        
        return (img_blur,img_sharp)
        