from GoProDataset import *
import numpy as np
import argparse
import torch
import os
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Load Dataset')
parser.add_argument('--data_path', type=str, default='../dataset/') 
parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--epochs', type=int, default=30)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--eval', action='store_true')
# parser.add_argument('--use_wandb', default = False)
# parser.add_argument('--latent_size', type=int, default=2048)
# parser.add_argument('--eval_interval', type=int, default = 5)

args = parser.parse_args()
data_path = args.data_path

args.train_image_dir = data_path + 'train/'
args.test_image_dir = data_path + 'test/'
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.makedirs('output_data/', exist_ok = True)
train_dataset = GoProDataset( image_dir = args.train_image_dir, image_filename_pattern="{}.png" ,length=224, width = 224)
test_dataset = GoProDataset(image_dir=args.test_image_dir, image_filename_pattern="{}.png", length=224, width = 224)

train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 4)
val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 4)

all_blur = np.zeros((3,))
all_sharp = np.zeros((3,))
   
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for data in tqdm(train_loader):
    blur, sharp = data
    psum  += blur.sum(axis = [0, 2, 3])
    psum_sq += (blur ** 2).sum(axis = [0, 2, 3])
    psum  += sharp.sum(axis = [0, 2, 3])
    psum_sq += (sharp ** 2).sum(axis = [0, 2, 3])

# for data in tqdm(val_loader):
#     blur, sharp = data
#     psum  += blur.sum(axis = [0, 2, 3])
#     psum_sq += (blur ** 2).sum(axis = [0, 2, 3])
#     psum  += sharp.sum(axis = [0, 2, 3])
#     psum_sq += (sharp ** 2).sum(axis = [0, 2, 3])


# count = float((2*len(train_dataset) + 2*len(test_dataset))*224*224)
count = float((2*len(train_dataset))*224*224)

# mean and std
total_mean = psum / count
print(total_mean)
total_var  = (psum_sq / count) - (total_mean ** 2)
print(total_var)
total_std  = torch.sqrt(total_var)
print(total_std)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))

########## For Whole dataset ###############
# mean: tensor([0.4337, 0.4242, 0.4132])
# std:  tensor([0.2395, 0.2335, 0.2373])



# mean: tensor([0.4332, 0.4223, 0.4177])
# std:  tensor([0.2337, 0.2298, 0.2323])
