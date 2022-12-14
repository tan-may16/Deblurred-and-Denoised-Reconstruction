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
    if not os.path.exists("../eval_rosbag/"):
        os.makedirs("../eval_rosbag/")

    output_dir = "../eval_rosbag/"
    base_dir = "../test/"
    args = parser.parse_args()
    test_image = args.image_path
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # _img_test = Image.open(test_image).convert('RGB')

    preprocessing = transforms.Compose([
        transforms.CenterCrop((448, 448)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    PATH = args.model_name
    model_params = torch.load(PATH)
    model = AEModel(args.latent_size, input_shape = (3, 224, 224)).cuda()
    model.load_state_dict(model_params)
        
    model.eval()
    
    for i in range(200):    
        test_image_path = base_dir + str(i) + ".png"
        _img_test = Image.open(test_image_path).convert('RGB')
        img_test = preprocessing(_img_test)
        img_test = img_test.to(args.device)
        img_test = torch.unsqueeze(img_test, 0)
        latent_vector = model.encoder(img_test)
        x_reconstructed = model.decoder(latent_vector)     
        save_image(make_grid(x_reconstructed.float(), nrow=8),output_dir + "{}_denoised.png".format(i))
        save_image(make_grid(img_test.float(), nrow=8),output_dir + "{}_base.png".format(i))
        
    print("Successful! Reconstructions of data have been saved.")

if __name__ == '__main__':
    main(grad_clip=1 )
