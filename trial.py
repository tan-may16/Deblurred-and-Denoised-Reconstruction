import numpy as np
# from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
from GoProDataset import GoProDataset
import argparse
from model import *
from torchvision.utils import save_image, make_grid
import os
def main():
    parser = argparse.ArgumentParser(description='Load Dataset')
    parser.add_argument('--data_path', type=str, default='../dataset/') # dataset/train/GOPR0374_11_00/
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_wandb', default = False)
    
    
    args = parser.parse_args()
    data_path = args.data_path

    args.train_image_dir = data_path + 'train/GOPR0374_11_00/'
    args.test_image_dir = data_path + 'test/GOPR0374_11_00/'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs('output_data/', exist_ok = True)
    train_dataset = GoProDataset( image_dir = args.train_image_dir, image_filename_pattern="{}.png", size = 224)
    test_dataset = GoProDataset(image_dir=args.test_image_dir, image_filename_pattern="{}.png", size = 224)
    
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
    
    if torch.cuda.is_available()==True:
        device="cuda:0"
    else:
        device ="cpu"

    
    # model=denoising_model().to(device)
    model = AEModel(False, 1024, input_shape = (3, 224, 224)).to(device)
    criterion=nn.MSELoss(reduction='none')
    
    optimizer=optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-5)

    epochs=args.epochs
    l=len(train_loader)
    print("Length",l)
    losslist=list()
    epochloss=0
    running_loss=0
    if (args.use_wandb):
        wandb.init(project="vlr-project")
    
    for epoch in range(epochs):
    
        print("Entering Epoch: ",epoch)
        for i,(data) in enumerate(train_loader):
            blur = data[0]
            sharp = data[1]
            blur, sharp = blur.to(device), sharp.to(device)
            latent_vector = model.encoder(blur)
            output = model.decoder(latent_vector)
            # print(output.shape, sharp.shape)
            loss = torch.mean(criterion(output,sharp).reshape(sharp.shape[0],-1).sum(dim = 1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            epochloss+=loss.item()
            if args.use_wandb:
                wandb.log(loss)
            if (epoch%5 == 0) and i==0:
                # img_out = output[0].cpu().detach()
                # img_oirg = sharp[0].cpu().detach()
                # save_image(img_out, "{}_reconstructed_img.png".format(epoch))
                # save_image(sharp[0].cpu().detach(), "{}_original_img.png".format(epoch))
                save_image(make_grid(output, nrow=8),"output_data/{}_reconstructions.jpg".format(epoch))
                save_image(make_grid(sharp, nrow=8),"output_data/{}_original.jpg".format(epoch))
                
                
                
        losslist.append(running_loss/l)
        running_loss=0

        # print("======> epoch: {}/{}, Loss:{}".format(epoch,epochs,loss.item()))


if __name__ == "__main__":
    main()
    
    

