import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from GoProDataset import GoProDataset
import argparse
from model import *
from torchvision.utils import save_image, make_grid
import os
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def main(grad_clip=1):
    
    
    parser = argparse.ArgumentParser(description='Load Dataset')
    parser.add_argument('--latent_size', type=int, default=2048)
    parser.add_argument('--model_name', type=str, default=" ")
    parser.add_argument('--image_path', type= str, default='test_image.jpg')
    
    args = parser.parse_args()
    test_image = args.image_path
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    _img_test = Image.open(test_image).convert('RGB')

    preprocessing = transforms.Compose([
        transforms.CenterCrop((672, 672)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    img_test = preprocessing(_img_test)
    
    PATH = args.model_name
    model_params = torch.load(PATH)
    # print(model_params)
    model = AEModel(args.latent_size, input_shape = (3, 224, 224)).cuda()
    model.load_state_dict(model_params)
        
    model.eval()
    img_test = img_test.to(args.device)
    img_test = torch.unsqueeze(img_test, 0)
    # print(img_test.size())
    latent_vector = model.encoder(img_test)
    x_reconstructed = model.decoder(latent_vector)     
    save_image(make_grid(x_reconstructed.float(), nrow=8),"reconstruction_image.jpg")
    print("Successful! Reconstruction Image has been saved.")

if __name__ == '__main__':
    main(grad_clip=1 )

