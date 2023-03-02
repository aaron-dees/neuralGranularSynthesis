import sys
sys.path.append('../')

from models.usd_vae import VAE
from models.dataloaders import UrbanSoundDataset
from utils.audio_preprocessing import convert_mel_spectrograms_to_waveform, save_signals
from models.loss_functions import calc_combined_loss
from scripts.hyper_parameters_urbansound import *


import torch
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import pickle
import time
import wandb

# start a new wandb run to track this script
if WANDB:
    wandb.init(
        # set the wandb project where this run will be logged
        project="urbanSound_0",
        name= "run_0.0005lr_2batch_20epochs_128latentsize",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "VAE",
        "dataset": "UrbanSound8K",
        "epochs": EPOCHS,
        "latent size": LATENT_SIZE,
        "recon loss weight": RECONSTRUCTION_LOSS_WEIGHT, 
        }
    )

if __name__ == "__main__":

    model = VAE()

    model.to(DEVICE)

    # create the mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = FRAME_SIZE,
        hop_length = HOP_LENGTH,
        n_mels = NUM_MELS
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, DATALOADER_DEVICE)

    usd_dataloader = torch.utils.data.DataLoader(usd, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    if TRAIN:

        ##########
        # Training
        ########## 

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            train_loss = 0.0
            kl_loss_sum = 0.0
            reconstruction_loss_sum = 0.0
            for data in usd_dataloader:
                spec, label = data 
                spec = Variable(spec).to(DEVICE)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients
                x_hat, z, mu, log_variance = model(spec)                 # forward pass: compute predicted outputs 
                loss, kl_loss, reconstruction_loss = calc_combined_loss(x_hat, spec, mu, log_variance, RECONSTRUCTION_LOSS_WEIGHT)       # calculate the loss
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step
                # I don't thinnk it's necisary to multiply by the batch size here in reporting the loss, or is it?
                train_loss += loss.item()*spec.size(0)  # update running training loss
                kl_loss_sum += kl_loss.item()*spec.size(0)
                reconstruction_loss_sum += reconstruction_loss.item()*spec.size(0)
            
            # print avg training statistics 
            train_loss = train_loss/len(usd_dataloader) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = kl_loss_sum/len(usd_dataloader)
            reconstruction_loss = reconstruction_loss_sum/len(usd_dataloader)
            # wandb logging
            if WANDB:
                wandb.log({"recon_loss": reconstruction_loss, "kl_loss": kl_loss, "loss": train_loss})

            print('Epoch: {}'.format(epoch+1),
            '\tTraining Loss: {:.4f}'.format(train_loss))
            print(f'--- KL Loss: {kl_loss}; Reconstruction Loss: {reconstruction_loss}')

        if(SAVE_MODEL == True):
            torch.save(model.state_dict(), MODEL_PATH)

    else:
        # with torch.no_grad():

            # Load Model
        if(LOAD_MODEL == True):
            model.load_state_dict(torch.load(MODEL_PATH))

        # Lets get batch of test images
        dataiter = iter(usd_dataloader)
        spec, labels = next(dataiter)
        spec = spec.to(DEVICE)
            
        mel_spec_hat, z, _, _ = model(spec)                     # get sample outputs
        mel_spec_hat = mel_spec_hat.detach().cpu()          # use detach when it's an output that requires_grad

        reconstructed_signals = convert_mel_spectrograms_to_waveform(mel_spec_hat,
                                                                     sample_rate=SAMPLE_RATE,
                                                                     n_stft=FRAME_SIZE//2 +1, 
                                                                     n_fft=FRAME_SIZE, n_mels=NUM_MELS, 
                                                                     hop_length=HOP_LENGTH)


        # Why is the reconstructed waveform slightly smaller than the original 22050 -> 22016
        print(reconstructed_signals.shape)

        # reconstructed_signals = convert_spectrograms_to_audio(x_hat, sampled_min_max_values, HOP_LENGTH)

        # save_signals(reconstructed_signals, file_paths, RECONSTRUCTION_SAVE_DIR)

        print("Reconstructions saved")

            # plot the first ten input images and then reconstructed images
            # show_image_comparisons(images, x_hat)
            # show_latent_space(z.detach().cpu().numpy(), labels)
    if WANDB:
        wandb.finish()