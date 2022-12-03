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
import wandb


def main(grad_clip=1):
    
    
    parser = argparse.ArgumentParser(description='Load Dataset')
    parser.add_argument('--data_path', type=str, default='../dataset/') 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_wandb', default = False)
    parser.add_argument('--latent_size', type=int, default=1024)
    parser.add_argument('--eval_interval', type=int, default = 2)
    parser.add_argument('--model_name', type=str, default=" ")
    
    args = parser.parse_args()
    data_path = args.data_path

    args.train_image_dir = data_path + 'train/'
    args.test_image_dir = data_path + 'test/'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output_path = 'test_data_output/'
    os.makedirs(output_path, exist_ok = True)
    
    train_dataset = GoProDataset( image_dir = args.train_image_dir, image_filename_pattern="{}.png" ,length=224, width = 224)
    test_dataset = GoProDataset(image_dir=args.test_image_dir, image_filename_pattern="{}.png", length=224, width = 224)
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = 4)
    val_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 4)
    
    PATH = args.model_name
    model_params = torch.load(PATH)
    model = AEModel(args.latent_size, input_shape = (3, 224, 224)).cuda()
    model.load_state_dict(model_params)
    
    if (args.use_wandb):
        wandb.init(project="vlr-project")
        
    model.eval()
    i = 0
    for x, x_sharp in val_loader:
        print(i)
        
        if i%args.eval_interval == 0:
            x, x_sharp = x.to(args.device), x_sharp.to(args.device) 
            latent_vector = model.encoder(x)
            x_reconstructed = model.decoder(latent_vector)
            MSE_loss = nn.MSELoss(reduction='none')
            loss = torch.mean(MSE_loss(x_reconstructed,x_sharp).reshape(x.shape[0],-1).sum(dim = 1))
            
            
            save_image(make_grid(x_reconstructed.float(), nrow=8),output_path+"{}_reconstructions.jpg".format(i))
            save_image(make_grid(x_sharp, nrow=8),output_path+"{}_original.jpg".format(i))
            save_image(make_grid(x, nrow=8),output_path+"{}_blur.jpg".format(i))
        i+=1
        

if __name__ == '__main__':
    main(grad_clip=1 )

