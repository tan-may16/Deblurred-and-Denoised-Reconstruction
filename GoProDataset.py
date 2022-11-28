
from collections import Counter
import os
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GoProDataset(Dataset):
    def __init__(self, image_dir, image_filename_pattern, size=224):
        """
        Initialize dataset.

        Args:
            image_dir (str): Path to the directory with COCO images
            question_json_file_path (str): Path to json of questions
            annotation_json_file_path (str): Path to json of mapping
                images, questions, and answers together
            image_filename_pattern (str): The pattern the filenames
                (eg "COCO_train2014_{}.jpg")
        """
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._blur_image_dir = self._image_dir + "blur"
        self._sharp_image_dir = self._image_dir + "sharp"
        self.size = size ## Resize size

    def __len__(self):
        return len([entry for entry in os.listdir(self._blur_image_dir) if os.path.isfile(os.path.join(self._blur_image_dir, entry))])

    def __getitem__(self, idx):
        """
        Load an item of the dataset.

        Args:
            idx: index of the data item

        Returns:
            A dict containing torch tensors for image, question and answers
        """
        # Load and pre-process image
        # name = str(q_anno['image_id'])
        name = str(idx + 1)
        if len(name) < 6:
            name = '0' * (6 - len(name)) + name
        img_name = self._image_filename_pattern.format(name)
        _img_blur = Image.open(os.path.join(self._blur_image_dir, img_name)).convert('RGB')
        _img_sharp = Image.open(os.path.join(self._sharp_image_dir, img_name)).convert('RGB')
        
        width, height = _img_blur.size
        max_dim = max(width, height)
        preprocessing = transforms.Compose([
            transforms.Pad((0, 0, max_dim - width, max_dim - height)),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()])
        img_blur = preprocessing(_img_blur)
        img_sharp = preprocessing(_img_sharp)
        
        return (img_blur,img_sharp)
        