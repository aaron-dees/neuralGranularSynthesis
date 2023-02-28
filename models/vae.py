import os

import sys
sys.path.append('../')

from utils.audio_preprocessing import MinMaxNormaliser
from utils.utilities import sample_from_distribution
from models.dataloaders import FSDDDataset

import librosa
import pickle
import soundfile as sf

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('mps:0')
# device = torch.device('cpu')
TRAIN = True
BATCH_SIZE = 10
EPOCHS = 20
LEARNING_RATE = 0.0005
SAVE_MODEL = True
LOAD_MODEL = True
MODEL_PATH = '/Users/adees/Code/neural_granular_synthesis/models/saved_models/fsdd_vae_gpu_20epochs_10batch.pt'
HOP_LENGTH = 256
MIN_MAX_VALUES_PATH = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/min_max_values.pkl"
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions"

#################
# Loss Functions
#################

def calc_reconstruction_loss(target, prediction):

    error = target - prediction
    reconstruction_loss = torch.mean(error**2)

    return reconstruction_loss

def calc_kl_loss(mu, log_variance):

    kl_loss = - 0.5 * torch.sum(1 + log_variance - torch.square(mu) - torch.exp(log_variance))

    return kl_loss

def calc_combined_loss(target, prediction, mu, log_variance, reconstruction_loss_weight):

    reconstruction_loss = calc_reconstruction_loss(target, prediction)
    kl_loss = calc_kl_loss(mu, log_variance)
    combined_loss = (reconstruction_loss_weight * reconstruction_loss) + kl_loss

    return combined_loss, kl_loss, reconstruction_loss

#############
# Models
#############

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()

        # pad = (3 // 2 + (3 - 2 * (3 // 2)) - 1, 3 // 2)

        self.Conv_E_1 = nn.Conv2d(1, 512, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(512, 256, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(256, 128, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_4= nn.Conv2d(128, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_5= nn.Conv2d(64, 32, stride=(2,1), kernel_size=3, padding=3//2)
        self.Dense_E_1 = nn.Linear(1024,128)

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

        z = sample_from_distribution(mu, log_variance, device, shape=(128))

        return z, mu, log_variance

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # pad = (3 // 2 + (3 - 2 * (3 // 2)) - 1, 3 // 2)

        self.Dense_D_1 = nn.Linear(128, 1024)
        self.ConvT_D_1 = nn.ConvTranspose2d(32, 32, stride=(2,1), kernel_size=3, padding = 3//2, output_padding = (1,0))
        # Note the need to add output padding here for tranposed dimensions to match
        self.ConvT_D_2 = nn.ConvTranspose2d(32, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_3 = nn.ConvTranspose2d(64, 128, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
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
        Reshape_D_1 = torch.reshape(Dense_D_1, (BATCH_SIZE, 32, 8, 4))
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
        z, mu, log_variance = self.Encoder(x);

        # z ---> x_hat
        x_hat = self.Decoder(z)

        return x_hat, z, mu, log_variance


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

# some utility functions for generating audio

def convert_spectrograms_to_audio(log_spectrograms, min_max_values):
    signals = []
    min_max_normaliser = MinMaxNormaliser(0, 1)
    for log_spectrogram, min_max_value in zip(log_spectrograms, min_max_values):
        # reshape log spectrogram
        log_spectrogram = log_spectrogram.squeeze()
        # apply de-normalisation
        denorm_log_spec = min_max_normaliser.denormalise(log_spectrogram, min_max_value["min"],  min_max_value["max"])
        # log spectrogram -> spectrogram
        spectrogram = librosa.db_to_amplitude(denorm_log_spec)
        # apply griffin-lim
        signal = librosa.istft(spectrogram, hop_length = HOP_LENGTH)
        # append signal to 'signals'
        signals.append(signal)

    return signals

def save_signals(signals, file_paths ,save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, "reconstructed_" + file_paths[i][74:-8] + ".wav")
        sf.write(save_path, signal, sample_rate)

## Script

if "main":

    model = VAE()

    model.to(device)

    train_dir = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/log_spectograms/" 
    training_data = FSDDDataset(root_dir=train_dir)
    fsdd_dataloader = torch.utils.data.DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    if TRAIN:

        ##########
        # Training
        ########## 

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            train_loss = 0.0
            kl_loss_sum = 0.0
            reconstruction_loss_sum = 0.0
            for data in fsdd_dataloader:
                img, _ = data 
                img = Variable(img).to(device)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients
                x_hat, z, mu, log_variance = model(img)                 # forward pass: compute predicted outputs 
                loss, kl_loss, reconstruction_loss = calc_combined_loss(x_hat, img, mu, log_variance, 1000000)       # calculate the loss
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step
                # I don't thinnk it's necisary to multiply by the batch size here in reporting the loss, or is it?
                train_loss += loss.item()*img.size(0)  # update running training loss
                kl_loss_sum += kl_loss.item()*img.size(0)
                reconstruction_loss_sum += reconstruction_loss.item()*img.size(0)
            
            # print avg training statistics 
            train_loss = train_loss/len(fsdd_dataloader) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = kl_loss_sum/len(fsdd_dataloader)
            reconstruction_loss = reconstruction_loss_sum/len(fsdd_dataloader)
            print('Epoch: {}'.format(epoch+1),
            '\tTraining Loss: {:.4f}'.format(train_loss))
            print(f'--- KL Loss: {kl_loss}; Reconstruction Loss: {reconstruction_loss}')

        if(SAVE_MODEL == True):
            torch.save(model.state_dict(), MODEL_PATH)


    else:
        with torch.no_grad():

            # Load Model
            if(LOAD_MODEL == True):
                model.load_state_dict(torch.load(MODEL_PATH))

            # Lets get batch of test images
            dataiter = iter(fsdd_dataloader)
            spectrograms, file_paths = next(dataiter)
            spectrograms = spectrograms.to(device)
            
            # load spectrograms + min max values
            with open(MIN_MAX_VALUES_PATH, "rb") as f:
                min_max_values = pickle.load(f)
            
            sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]

            x_hat, z, _, _ = model(spectrograms)                     # get sample outputs
            x_hat = x_hat.detach().cpu().numpy()           # use detach when it's an output that requires_grad

            reconstructed_signals = convert_spectrograms_to_audio(x_hat, sampled_min_max_values)

            save_signals(reconstructed_signals, file_paths, RECONSTRUCTION_SAVE_DIR)

            print(len(reconstructed_signals))

            # plot the first ten input images and then reconstructed images
            # show_image_comparisons(images, x_hat)
            # show_latent_space(z.detach().cpu().numpy(), labels)

    print("Done")