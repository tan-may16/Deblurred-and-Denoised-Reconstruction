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


def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics

def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):   
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    def _helper(epoch):
        beta = epoch*target_val/max_epochs
        return beta
    return _helper




def main(beta_mode = 'constant', target_beta_val = 1, grad_clip=1):
    
    
    parser = argparse.ArgumentParser(description='Load Dataset')
    parser.add_argument('--data_path', type=str, default='../dataset/') 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_wandb', default = False)
    parser.add_argument('--latent_size', type=int, default=1024)
    parser.add_argument('--eval_interval', type=int, default = 5)
    
    args = parser.parse_args()
    data_path = args.data_path

    args.train_image_dir = data_path + 'train/'
    args.test_image_dir = data_path + 'test/'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs('output_data/', exist_ok = True)
    train_dataset = GoProDataset( image_dir = args.train_image_dir, image_filename_pattern="{}.png" )
    test_dataset = GoProDataset(image_dir=args.test_image_dir, image_filename_pattern="{}.png")
    
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
    
    model = AEModel(args.latent_size, input_shape = (3, 224, 224)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=args.epochs, target_val = target_beta_val) 
        
        
    if (args.use_wandb):
        wandb.init(project="vlr-hw2")
    
    for epoch in range(args.epochs):
        
        print('epoch', epoch)
        
        model.train()
        train_metrics_list = []
        i = 0
        for x, x_sharp in train_loader:
            # x = preprocess_data(x)
            x, x_sharp = x.to(args.device), x_sharp.to(args.device) 
            latent_vector = model.encoder(x)
            x_reconstructed = model.decoder(latent_vector)
            MSE_loss = nn.MSELoss(reduction='none')
            loss = torch.mean(MSE_loss(x_reconstructed,x_sharp).reshape(x.shape[0],-1).sum(dim = 1))
            if args.use_wandb:
                wandb.log("Loss/train":loss)
            _metric = OrderedDict(recon_loss=loss)
            train_metrics_list.append(_metric)
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            if i == 0:
                save_image(make_grid(x_reconstructed.float(), nrow=8),"output_data/{}_reconstructions.jpg".format(epoch))
                save_image(make_grid(x_sharp, nrow=8),"output_data/{}_original.jpg".format(epoch))
                save_image(make_grid(x, nrow=8),"output_data/{}_blur.jpg".format(epoch))
            i+=1  
            
        train_metrics = avg_dict(train_metrics_list) 
        print("Train Metrics")
        print(epoch, train_metrics)
               
        if args.use_wandb:
                wandb.log(train_metrics)
                
                
        #Validation
        if (epoch)%(args.eval_interval) == 0:
            model.eval()
            val_metrics_list = []
            with torch.no_grad():
                i = 0
                for x, x_sharp in val_loader:
                    x, x_sharp = x.to(args.device), x_sharp.to(args.device) 
                    latent_vector = model.encoder(x)
                    x_reconstructed = model.decoder(latent_vector)
                    MSE_loss = nn.MSELoss(reduction='none')
                    loss = torch.mean(MSE_loss(x_reconstructed,x_sharp).reshape(x.shape[0],-1).sum(dim = 1))
                    if args.use_wandb:
                        wandb.log("Loss/validation":loss)
                    _metric = OrderedDict(recon_loss=loss)
                    val_metrics_list.append(_metric)
                    if i == 0:
                        save_image(make_grid(x_reconstructed.float(), nrow=8),"output_data/{}_V_reconstructions.jpg".format(epoch))
                        save_image(make_grid(x_sharp, nrow=8),"output_data/{}_V_original.jpg".format(epoch))
                        save_image(make_grid(x, nrow=8),"output_data/{}_V_blur.jpg".format(epoch))
                    i+=1
                    
            val_metrics = avg_dict(val_metrics_list)
            print("Val Metrics:")
            print(val_metrics)
            if args.use_wandb:
                wandb.log(val_metrics)    

if __name__ == '__main__':
    main( beta_mode = 'linear', target_beta_val = 1)

