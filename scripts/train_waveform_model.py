import sys
sys.path.append('../')

from models.waveform_models.usd_waveform_model import WaveformEncoder, WaveformDecoder, WaveformVAE
from models.dataloaders.waveform_dataloaders import WaveformDataset
# from utils.audio_preprocessing import convert_mel_spectrograms_to_waveform, save_signals
# from models.loss_functions import calc_combined_loss
from scripts.configs.hyper_parameters_waveform import *


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

    model = WaveformVAE(kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    n_linears=3,
                    num_samples=NUM_SAMPLES,
                    h_dim=512,
                    z_dim=128)
    
    model.to(DEVICE)

    usd_waveforms = WaveformDataset(ANNOTATIONS_FILE, AUDIO_DIR, None, SAMPLE_RATE, NUM_SAMPLES, DATALOADER_DEVICE)

    usd_dataloader = torch.utils.data.DataLoader(usd_waveforms, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    if TRAIN:

        print("Training")

        # ##########
        # # Training
        # ########## 

        # optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        
        # for epoch in range(EPOCHS):
        #     train_loss = 0.0
        #     kl_loss_sum = 0.0
        #     reconstruction_loss_sum = 0.0
        #     for data in usd_dataloader:
        #         spec, label = data 
        #         spec = Variable(spec).to(DEVICE)                       # we are just intrested in just images
        #         # no need to flatten images
        #         optimizer.zero_grad()                   # clear the gradients
        #         x_hat, z, mu, log_variance = model(spec)                 # forward pass: compute predicted outputs 
        #         loss, kl_loss, reconstruction_loss = calc_combined_loss(x_hat, spec, mu, log_variance, RECONSTRUCTION_LOSS_WEIGHT)       # calculate the loss
        #         loss.backward()                         # backward pass
        #         optimizer.step()                        # perform optimization step
        #         # I don't thinnk it's necisary to multiply by the batch size here in reporting the loss, or is it?
        #         train_loss += loss.item()*spec.size(0)  # update running training loss
        #         kl_loss_sum += kl_loss.item()*spec.size(0)
        #         reconstruction_loss_sum += reconstruction_loss.item()*spec.size(0)
            
        #     # print avg training statistics 
        #     train_loss = train_loss/len(usd_dataloader) # does len(fsdd_dataloader) return the number of batches ?
        #     kl_loss = kl_loss_sum/len(usd_dataloader)
        #     reconstruction_loss = reconstruction_loss_sum/len(usd_dataloader)
        #     # wandb logging
        #     if WANDB:
        #         wandb.log({"recon_loss": reconstruction_loss, "kl_loss": kl_loss, "loss": train_loss})

        #     print('Epoch: {}'.format(epoch+1),
        #     '\tTraining Loss: {:.4f}'.format(train_loss))
        #     print(f'--- KL Loss: {kl_loss}; Reconstruction Loss: {reconstruction_loss}')

        # if(SAVE_MODEL == True):
        #     torch.save(model.state_dict(), MODEL_PATH)

    else:
        # with torch.no_grad():

            # Load Model
        if(LOAD_MODEL == True):
            model.load_state_dict(torch.load(MODEL_PATH))

        # Lets get batch of test images
        dataiter = iter(usd_dataloader)
        spec, labels = next(dataiter)
        spec = spec.to(DEVICE)
            
        x_hat ,z, mu, logvar = model(spec)                     # get sample outputs


        print(f"Reconstruction shape: {x_hat.shape}")
        print(z.shape)
        print(mu.shape)
        print(logvar.shape)
        print("Reconstructions saved")

            # plot the first ten input images and then reconstructed images
            # show_image_comparisons(images, x_hat)
            # show_latent_space(z.detach().cpu().numpy(), labels)

    
