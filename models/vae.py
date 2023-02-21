import torch
import torch.nn as nn
# import torchaudio
# import librosa
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision as tv
import time

device = torch.device('mps:0')
# device = torch.device('cpu')
TRAIN = True

# MNIST Preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # pad = (3 // 2 + (3 - 2 * (3 // 2)) - 1, 3 // 2)

        self.Conv_E_1 = nn.Conv2d(1, 32, stride=1, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_4= nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=3//2)
        self.Dense_E_1 = nn.Linear(3136*32,2)

        self.Act_E_1 = nn.ReLU()

        # Do I need batch mormalisations?

    def forward(self, x, training):

        # x ---> z
        Conv_E_1 = self.Conv_E_1(x)
        Act_E_1 = self.Act_E_1(Conv_E_1)
        # Do I need batch normalisaion here??
        Conv_E_2 = self.Conv_E_2(Act_E_1)
        Act_E_2 = self.Act_E_1(Conv_E_2)
        Conv_E_3 = self.Conv_E_3(Act_E_2)
        Act_E_3 = self.Act_E_1(Conv_E_3)
        Conv_E_4 = self.Conv_E_4(Act_E_3)
        Act_E_4 = self.Act_E_1(Conv_E_4)
        # Is it ok to use the below flatten
        Flat_E_1 = Act_E_4.flatten()
        z = self.Dense_E_1(Flat_E_1)
        # I could in theory replace the below with a flatten and dense layer, like that in tutorial
        # https://youtu.be/TtyoFTyJuEY

        return z

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # pad = (3 // 2 + (3 - 2 * (3 // 2)) - 1, 3 // 2)

        self.Dense_D_1 = nn.Linear(2, 3136*32)
        self.ConvT_D_1 = nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding = 3//2, output_padding = 0)
        # Note the need to add output padding here for tranposed dimensions to match
        self.ConvT_D_2 = nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_3 = nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_4 = nn.ConvTranspose2d(32, 1, stride=1, kernel_size=3, padding = 3//2, output_padding = 0)

        self.Act_D_1 = nn.ReLU()
        self.Act_D_2 = nn.Sigmoid()
        self.Pixel_Shuffle_D = nn.PixelShuffle(4)

    def forward(self, z):

        # z ---> x_hat
        Dense_D_1 = self.Dense_D_1(z)
        Reshape_D_1 = torch.reshape(Dense_D_1, (32, 64, 7, 7))
        Conv_D_1 = self.ConvT_D_1(Reshape_D_1)
        Act_D_1 = self.Act_D_1(Conv_D_1)
        # Do I need batch norm here ?  
        Conv_D_2 = self.ConvT_D_2(Act_D_1)
        Act_D_2 = self.Act_D_1(Conv_D_2)
        Conv_D_3 = self.ConvT_D_3(Act_D_2)
        Act_D_3 = self.Act_D_1(Conv_D_3)
        Conv_D_4= self.ConvT_D_4(Act_D_3)
        x_hat = self.Act_D_2(Conv_D_4)

        return x_hat

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = Encoder()
        self.Decoder = Decoder()

        # Number of convolutional layers

    def forward(self, x, training):

        # x ---> z
        z = self.Encoder(x, training);

        # z ---> x_hat
        x_hat = self.Decoder(z)

        return x_hat 

if "main":

    vae = VAE()

    vae.to(device)

    if TRAIN:

        ##########
        # Training
        ########## 

        num_epochs = 10
        batch_size = 32
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(vae.parameters(),weight_decay=1e-5)
        
        for epoch in range(num_epochs):
            start = time.time()
            for data in dataloader:
                img, _ = data
                img = Variable(img).to(device)
                # ===================forward=====================
                output = vae(img, True)
                loss = distance(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ===================log========================
            print('epoch [{}/{}], loss:{:.16f}'.format(epoch+1, num_epochs, loss.item()))
            end = time.time()
            print('epoch time: ', end-start)


    else:
        with torch.no_grad():

            x = torch.ones(1, 28, 28, device=device)

            x_hat = vae(x, False)


            # print("Latent Shape: ", z.shape)
            # print("Latent Shape: ", z.to('cpu'))
            print("Reconstruction Shape: ", x_hat.shape)
            print("x_hat sum: ", torch.sum(x_hat.to('cpu')))
            # print("z_hat sum: ", torch.sum(z.to('cpu')))

    print("Done")