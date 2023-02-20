import torch
import torch.nn as nn
import torchaudio
import librosa

device = torch.device('mps:0')
TRAIN = False

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.Conv_E_1 = nn.Conv2d(3, 192, stride=1, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(192, 192, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(192, 192, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_4 = nn.Conv2d(192, 12, stride=1, kernel_size=3, padding=3//2)

        self.Act_E_1 = nn.PReLU(init=0.2)

    def forward(self, x, training):

        # x ---> z
        Conv_E_1 = self.Conv_E_1(x)
        Act_E_1 = self.Act_E_1(Conv_E_1)
        Conv_E_2 = self.Conv_E_2(Act_E_1)
        Act_E_2 = self.Act_E_1(Conv_E_2)
        Conv_E_3 = self.Conv_E_3(Act_E_2)
        Act_E_3 = self.Act_E_1(Conv_E_3)
        y = self.Conv_E_4(Act_E_3)

        return y

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.Conv_D_1 = nn.Conv2d(12, 192, stride=1, kernel_size=3, padding=3//2)
        self.Conv_D_2 = nn.Conv2d(192, 192, stride=1, kernel_size=3, padding=3//2)
        self.Conv_D_3 = nn.Conv2d(192, 192, stride=1, kernel_size=3, padding=3//2)
        self.Conv_D_4 = nn.Conv2d(192, 3*(4**2), stride=1, kernel_size=3, padding=3//2)

        self.Act_D_1 = nn.PReLU(init=0.2)
        #self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upsamp = nn.Interpolate(mode='nearest', scale_factor=2)
        self.Pixel_Shuffle_D = nn.PixelShuffle(4)

    def forward(self, y_hat):

        # z ---> x_hat
        Conv_D_1 = self.Conv_D_1(y_hat)
        Act_D_1 = self.Act_D_1(Conv_D_1)
        Conv_D_2 = self.Conv_D_2(Act_D_1)
        Act_D_2 = self.Act_D_1(Conv_D_2)
        Conv_D_3 = self.Conv_D_3(Act_D_2)
        Act_D_3 = self.Act_D_1(Conv_D_3)
        Conv_D_4= self.Conv_D_4(Act_D_3)
        x_hat = self.Pixel_Shuffle_D(Conv_D_4)

        return x_hat

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.Encoder = Encoder()
        self.Decoder = Decoder()

    def forward(self, x, training):

        # x ---> z
        z = self.Encoder(x, training);

        # x ---> x_hat
        x_hat = self.Decoder(z)

        return x_hat, z 

if "main":

    vae = VAE()

    vae.to(device)

    if TRAIN:

        ##########
        # Training
        ########## 

        print("Training")

    else:
        with torch.no_grad():

            x = torch.ones(3, 256, 256, device=device)

            x_hat, z = vae(x, False)

            print("Latent Shape: ", z.shape)
            print("Reconstruction Shape: ", x_hat.shape)
            print("x_hat sum: ", torch.sum(x_hat.to('cpu')))
            print("z_hat sum: ", torch.sum(z.to('cpu')))

    print("Done")