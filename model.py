# import numpy as np
# # from keras.datasets import mnist
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torchvision import transforms
# import torch.nn as nn
# from torch.utils.data import DataLoader,Dataset
# import torch
# import torch.optim as optim
# from torch.autograd import Variable

# class denoising_model(nn.Module):
#   def __init__(self):
#     super(denoising_model,self).__init__()
#     self.encoder=nn.Sequential(
#                   nn.Linear(28*28,256),
#                   nn.ReLU(True),
#                   nn.Linear(256,128),
#                   nn.ReLU(True),
#                   nn.Linear(128,64),
#                   nn.ReLU(True)
        
#                   )
    
#     self.decoder=nn.Sequential(
#                   nn.Linear(64,128),
#                   nn.ReLU(True),
#                   nn.Linear(128,256),
#                   nn.ReLU(True),
#                   nn.Linear(256,28*28),
#                   nn.Sigmoid(),
#                   )
    
 
#   def forward(self,x):
#     x=self.encoder(x)
#     x=self.decoder(x)
    
#     return x

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256

        self.fc = nn.Linear(self.conv_out_dim, self.latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        self.fc = nn.Linear(self.conv_out_dim, 2*self.latent_dim)

    
    def forward(self, x):
        y = self.convs(x)
        y = y.reshape(x.shape[0],-1)
        y = self.fc(y)
        y_mean = y[:,:self.latent_dim]
        y_log_sigma = y[:,self.latent_dim:]
        return (y_mean,y_log_sigma)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.base_size = (128, self.output_shape[1] // 8, self.output_shape[2] // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        self.deconvs = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

    def forward(self, z):
        z = self.fc(z)
        s0,s1,s2 = self.base_size
        z = z.reshape(-1,s0,s1,s2)
        z = self.deconvs(z)
        return z
    


class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)