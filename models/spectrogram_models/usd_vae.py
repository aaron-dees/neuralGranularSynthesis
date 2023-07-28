import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution
from scripts.configs.hyper_parameters_urbansound import BATCH_SIZE, DEVICE, LATENT_SIZE


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

#############
# Models
#############

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()

        self.Conv_E_1 = nn.Conv2d(1, 512, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(512, 256, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(256, 128, stride=(2,1), kernel_size=3, padding=3//2)
        self.Conv_E_4= nn.Conv2d(128, 64, stride=(2,1), kernel_size=3, padding=3//2)
        self.Conv_E_5= nn.Conv2d(64, 32, stride=(2,1), kernel_size=3, padding=3//2)
        # Dense layer based on the striding used in conv layers
        self.Dense_E_1 = nn.Linear(32 * int(64/pow(2,5)) * int(44/pow(2,2)), LATENT_SIZE)

        self.Flat_E_1 = nn.Flatten()

        self.Norm_E_1 = nn.BatchNorm2d(512)
        self.Norm_E_2 = nn.BatchNorm2d(256)
        self.Norm_E_3 = nn.BatchNorm2d(128)
        self.Norm_E_4 = nn.BatchNorm2d(64)
        self.Norm_E_5 = nn.BatchNorm2d(32)


        self.Act_E_1 = nn.ReLU()


    def forward(self, x):

        # x ---> z

        # Conv layer 1
        Conv_E_1 = self.Conv_E_1(x)
        Act_E_1 = self.Act_E_1(Conv_E_1)
        Norm_E_1 = self.Norm_E_1(Act_E_1)
        # Conv layer 2
        Conv_E_2 = self.Conv_E_2(Norm_E_1)
        Act_E_2 = self.Act_E_1(Conv_E_2)
        Norm_E_2 = self.Norm_E_2(Act_E_2)
        # Conv layer 3 
        Conv_E_3 = self.Conv_E_3(Norm_E_2)
        Act_E_3 = self.Act_E_1(Conv_E_3)
        Norm_E_3 = self.Norm_E_3(Act_E_3)
        # Conv layer 4
        Conv_E_4 = self.Conv_E_4(Norm_E_3)
        Act_E_4 = self.Act_E_1(Conv_E_4)
        Norm_E_4 = self.Norm_E_4(Act_E_4)
        # Conv layer 5
        Conv_E_5 = self.Conv_E_5(Norm_E_4)
        Act_E_5 = self.Act_E_1(Conv_E_5)
        Norm_E_5 = self.Norm_E_5(Act_E_5)
        # Dense layer for mu and log variance
        Flat_E_1 = self.Flat_E_1(Norm_E_5)
        mu = self.Dense_E_1(Flat_E_1)
        
        log_variance = self.Dense_E_1(Flat_E_1)

        z = sample_from_distribution(mu, log_variance)

        return z, mu, log_variance

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # Check is calculating this power and cast kills performance
        self.Dense_D_1 = nn.Linear(LATENT_SIZE, 32 * int(64/pow(2,5)) * int(44/pow(2,2)))
        self.ConvT_D_1 = nn.ConvTranspose2d(32, 32, stride=(2,1), kernel_size=3, padding = 3//2, output_padding = (1,0))
        # Note the need to add output padding here for tranposed dimensions to match
        self.ConvT_D_2 = nn.ConvTranspose2d(32, 64, stride=(2,1), kernel_size=3, padding = 3//2, output_padding = (1,0))
        self.ConvT_D_3 = nn.ConvTranspose2d(64, 128, stride=(2,1), kernel_size=3, padding = 3//2, output_padding = (1,0))
        self.ConvT_D_4 = nn.ConvTranspose2d(128, 256, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_5 = nn.ConvTranspose2d(256, 1, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        
        self.Norm_D_1 = nn.BatchNorm2d(32)
        self.Norm_D_2 = nn.BatchNorm2d(64)
        self.Norm_D_3 = nn.BatchNorm2d(128)
        self.Norm_D_4 = nn.BatchNorm2d(256)

        self.Act_D_1 = nn.ReLU()
        self.Act_D_2 = nn.Sigmoid()

    def forward(self, z):

        # z ---> x_hat

        # Dense Layer
        Dense_D_1 = self.Dense_D_1(z)
        # TODO: Find a nicer way of doing this programatically
        # Reshape based on the number striding used in encoder
        Reshape_D_1 = torch.reshape(Dense_D_1, (BATCH_SIZE, 32, int(64/pow(2,5)), int(44/pow(2,2))))
        # Conv layer 1
        Conv_D_1 = self.ConvT_D_1(Reshape_D_1)
        Act_D_1 = self.Act_D_1(Conv_D_1)
        Norm_D_1 = self.Norm_D_1(Act_D_1)
        # Conv layer 2
        Conv_D_2 = self.ConvT_D_2(Norm_D_1)
        Act_D_2 = self.Act_D_1(Conv_D_2)
        Norm_D_2 = self.Norm_D_2(Act_D_2)
        # Conv layer 3
        Conv_D_3 = self.ConvT_D_3(Norm_D_2)
        Act_D_3 = self.Act_D_1(Conv_D_3)
        Norm_D_3 = self.Norm_D_3(Act_D_3)
        # Conv layer 4
        Conv_D_4 = self.ConvT_D_4(Norm_D_3)
        Act_D_4 = self.Act_D_1(Conv_D_4)
        Norm_D_4 = self.Norm_D_4(Act_D_4)
        # Conv layer 5 (output)
        Conv_D_5= self.ConvT_D_5(Norm_D_4)
        x_hat = self.Act_D_2(Conv_D_5)

        return x_hat

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = Encoder()
        self.Decoder = Decoder()

        # Number of convolutional layers

    def forward(self, x):

        # x ---> z
        z, mu, log_variance = self.Encoder(x);

        # z ---> x_hat
        x_hat = self.Decoder(z)

        return x_hat, z, mu, log_variance