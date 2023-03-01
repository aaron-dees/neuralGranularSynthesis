import torch
import torch.nn as nn
# import torchaudio
# import librosa
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision as tv
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('mps:0')
# device = torch.device('cpu')
TRAIN = False
BATCH_SIZE = 6000
EPOCHS = 100
SAVE_MODEL = True
LOAD_MODEL = True
MODEL_PATH = './saved_models/mnist_vae_mine.pt'

# MNIST Preprocessing
transform = tv.transforms.ToTensor()
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=0)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # pad = (3 // 2 + (3 - 2 * (3 // 2)) - 1, 3 // 2)

        self.Conv_E_1 = nn.Conv2d(1, 32, stride=1, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_4= nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=3//2)
        self.Dense_E_1 = nn.Linear(3136,2)

        self.Flat_E_1 = nn.Flatten()

        self.Norm_E_1 = nn.BatchNorm2d(32)
        self.Norm_E_2 = nn.BatchNorm2d(64)
        self.Norm_E_3 = nn.BatchNorm2d(64)
        self.Norm_E_4 = nn.BatchNorm2d(64)


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
        # Dense layer
        Flat_E_1 = self.Flat_E_1(Norm_E_4)
        z = self.Dense_E_1(Flat_E_1)

        return z

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # pad = (3 // 2 + (3 - 2 * (3 // 2)) - 1, 3 // 2)

        self.Dense_D_1 = nn.Linear(2, 3136)
        self.ConvT_D_1 = nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding = 3//2, output_padding = 0)
        # Note the need to add output padding here for tranposed dimensions to match
        self.ConvT_D_2 = nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_3 = nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_4 = nn.ConvTranspose2d(64, 1, stride=1, kernel_size=3, padding = 3//2, output_padding = 0)
        
        self.Norm_D_1 = nn.BatchNorm2d(64)
        self.Norm_D_2 = nn.BatchNorm2d(64)
        self.Norm_D_3 = nn.BatchNorm2d(64)

        self.Act_D_1 = nn.ReLU()
        self.Act_D_2 = nn.Sigmoid()

    def forward(self, z):

        # z ---> x_hat

        # Dense Layer
        Dense_D_1 = self.Dense_D_1(z)
        Reshape_D_1 = torch.reshape(Dense_D_1, (BATCH_SIZE, 64, 7, 7))
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
        Norm_D_3 = self.Norm_D_1(Act_D_3)
        # Conv layer 4 (output)
        Conv_D_4= self.ConvT_D_4(Norm_D_3)
        x_hat = self.Act_D_2(Conv_D_4)

        return x_hat

# Model from AI and Sound tutorial
class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = Encoder()
        self.Decoder = Decoder()

        # Number of convolutional layers

    def forward(self, x):

        # x ---> z
        z = self.Encoder(x);

        # z ---> x_hat
        x_hat = self.Decoder(z)

        return x_hat, z


def show_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10,10))
    plt.scatter(latent_representations[:, 0],
        latent_representations[:, 1],
        cmap="rainbow",
        c = sample_labels,
        alpha = 0.5,
        s = 2)
    plt.colorbar
    plt.savefig("laetnt_rep.png") 

def show_image_comparisons(images, x_hat):

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
            
    # input images on top row, reconstructions on bottom
    for images, row in zip([images, x_hat], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("comparisons.png")


if "main":

    model = VAE()

    model.to(device)

    if TRAIN:

        ##########
        # Training
        ########## 

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        
        for epoch in range(EPOCHS):
            train_loss = 0.0
            for data in dataloader:
                img, _ = data 
                img = Variable(img).to(device)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients
                x_hat, z = model(img)                 # forward pass: compute predicted outputs 
                loss = loss_fn(x_hat, img)       # calculate the loss
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step
                train_loss += loss.item()*img.size(0)   # update running training loss
            
            # print avg training statistics 
            train_loss = train_loss/len(dataloader)
            print('Epoch: {}'.format(epoch+1),
            '\tTraining Loss: {:.4f}'.format(train_loss))

        if(SAVE_MODEL == True):
            torch.save(model.state_dict(), MODEL_PATH)


    else:
        with torch.no_grad():

            # Load Model
            if(LOAD_MODEL == True):
                model.load_state_dict(torch.load(MODEL_PATH))

            # Lets get batch of test images
            dataiter = iter(testloader)
            images, labels = next(dataiter)
            images = images.to(device)

            x_hat, z = model(images)                     # get sample outputs
            images = images.cpu().numpy()                    # prep images for display
            x_hat = x_hat.detach().cpu().numpy()           # use detach when it's an output that requires_grad

            # plot the first ten input images and then reconstructed images
            show_image_comparisons(images, x_hat)
            show_latent_space(z.detach().cpu().numpy(), labels)

            #

    print("Done")