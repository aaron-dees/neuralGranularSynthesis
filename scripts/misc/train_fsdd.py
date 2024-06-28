import sys
sys.path.append('../../')

from models.spectrogram_models.fsdd_vae import VAE
from models.dataloaders.spec_dataloaders import FSDDSpectrogramDataset
from utils.audio_preprocessing import convert_spectrograms_to_audio, save_signals
from models.loss_functions import calc_combined_loss
from models.dataloaders.waveform_dataloaders import make_audio_dataloaders_noPadding
from scripts.configs.hyper_parameters_fsdd import *


import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal


if __name__ == "__main__":

    model = VAE()

    model.to(DEVICE)
    if(LOAD_MODEL == True):
        model.load_state_dict(torch.load(MODEL_PATH))

    train_dir = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/log_spectrograms/under1sec/"
    # train_dir = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/log_spectograms/" 
    training_data = FSDDSpectrogramDataset(root_dir=train_dir)
    fsdd_dataloader_subset = torch.utils.data.Subset(training_data, list([0]))
    fsdd_dataloader = torch.utils.data.DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)
    fsdd_dataloader_subset = torch.utils.data.DataLoader(fsdd_dataloader_subset, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    if TRAIN:

        ##########
        # Training
        ########## 

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            train_loss = 0.0
            kl_loss_sum = 0.0
            reconstruction_loss_sum = 0.0
            start = time.time()
            for data in fsdd_dataloader_subset:
                img, _ = data 
                img = Variable(img).to(DEVICE)                       # we are just intrested in just images
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
            end = time.time()
            # print(f"Epoch time: {end-start}s")
            train_loss = train_loss/len(fsdd_dataloader_subset) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = kl_loss_sum/len(fsdd_dataloader_subset)
            reconstruction_loss = reconstruction_loss_sum/len(fsdd_dataloader_subset)
            print('Epoch: {}'.format(epoch+1),
            '\tTraining Loss: {:.4f}'.format(train_loss),
            '\tKL Loss: {:.4f}'.format(kl_loss),
            '\tRecon Loss: {:.4f}'.format(reconstruction_loss),
            '\tTime: {:.4f}'.format(end-start))
            # print(f'--- KL Loss: {kl_loss}; Reconstruction Loss: {reconstruction_loss}')

            if(SAVE_MODEL == True):
                torch.save(model.state_dict(), MODEL_PATH)

        if(SAVE_RECON):

            dataiter = iter(fsdd_dataloader_subset)
            spectrograms, file_paths = next(dataiter)
            spectrograms = spectrograms.to(DEVICE)

            # load spectrograms + min max values
            with open(MIN_MAX_VALUES_PATH, "rb") as f:
                min_max_values = pickle.load(f)
            
            sampled_min_max_values = [min_max_values[file_path] for file_path in
                        file_paths]

            x_hat, z, _, _ = model(spectrograms)                     # get sample outputs
            x_hat = x_hat.detach().cpu().numpy()           # use detach when it's an output that requires_grad

            reconstructed_signals = convert_spectrograms_to_audio(x_hat, sampled_min_max_values, HOP_LENGTH)
            reconstructed_signals_originals = convert_spectrograms_to_audio(spectrograms.cpu().numpy(), sampled_min_max_values, HOP_LENGTH)

            save_signals(reconstructed_signals, file_paths, RECONSTRUCTION_SAVE_DIR)
            save_signals(reconstructed_signals_originals, file_paths, RECONSTRUCTION_SAVE_DIR+"/originals/")

            print("Reconstructions saved")


    else:
        with torch.no_grad():

            # Load Model
            if(LOAD_MODEL == True):
                model.load_state_dict(torch.load(MODEL_PATH))

            # Lets get batch of test images
            dataiter = iter(fsdd_dataloader)
            spectrograms, file_paths = next(dataiter)
            spectrograms = spectrograms.to(DEVICE)
            
            # load spectrograms + min max values
            with open(MIN_MAX_VALUES_PATH, "rb") as f:
                min_max_values = pickle.load(f)
            
            sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]

            x_hat, z, _, _ = model(spectrograms)                     # get sample outputs
            x_hat = x_hat.detach().cpu().numpy()           # use detach when it's an output that requires_grad

            reconstructed_signals = convert_spectrograms_to_audio(x_hat, sampled_min_max_values, HOP_LENGTH)
            reconstructed_signals_originals = convert_spectrograms_to_audio(spectrograms.cpu().numpy(), sampled_min_max_values, HOP_LENGTH)

            save_signals(reconstructed_signals, file_paths, RECONSTRUCTION_SAVE_DIR)
            save_signals(reconstructed_signals_originals, file_paths, RECONSTRUCTION_SAVE_DIR+"/originals/")

            print("Reconstructions saved")

            # plot the first ten input images and then reconstructed images
            # show_image_comparisons(images, x_hat)
            # show_latent_space(z.detach().cpu().numpy(), labels)
