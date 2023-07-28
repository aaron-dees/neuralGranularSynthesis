import sys
sys.path.append('../')

from models.waveform_models.usd_waveform_model import WaveformEncoder, WaveformDecoder, WaveformVAE
from models.dataloaders.waveform_dataloaders import WaveformDataset
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance
from scripts.configs.hyper_parameters_waveform import *
from utils.utilities import plot_latents


import torch
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import pickle
import time
import wandb
import numpy as np
from datetime import datetime


# start a new wandb run to track this script
if WANDB:
    wandb.init(
        # set the wandb project where this run will be logged
        project="urbanSound_waveformVAE",
        name= "run_0.0005lr_37batch_10epochs_128latentsize_beta0.0001_envdist0",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Waveform_VAE",
        "dataset": "UrbanSound8K",
        "epochs": EPOCHS,
        "latent size": LATENT_SIZE,
        "env_dist": ENV_DIST,
        "beta": BETA
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

    # Split into train and validation
    # Program this better
    split = [(5*len(usd_waveforms))//6, (1*len(usd_waveforms))//6+1]
    train_set, val_set = torch.utils.data.random_split(usd_waveforms, split)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    test_set = torch.utils.data.Subset(usd_waveforms, range(0,TEST_SIZE))
    test_dataloader = torch.utils.data.DataLoader(val_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)


    if TRAIN:

        print("Training Mode")

        ###########
        # Training
        ########### 

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

        start_epoch = 0

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_FILE_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            train_loss = checkpoint['loss']


            print("----- Checkpoint File Loaded -----")
            print(f'Epoch: {start_epoch}')
            print(f'Loss: {train_loss}')

        # Model in training mode
        
        
        for epoch in range(start_epoch, EPOCHS):

            # Turn gradient trackin on for training loop
            model.train()

            ###############
            # Training loop - maybe abstract this out
            ###############
            running_train_loss = 0.0
            running_kl_loss = 0.0
            running_spec_loss = 0.0
            running_env_loss = 0.0

            for data in train_dataloader:
                waveform, label = data 
                waveform = Variable(waveform).to(DEVICE)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients
                x_hat, z, mu, log_variance = model(waveform)                 # forward pass: compute predicted outputs 

                # Compute loss
                kld = compute_kld(mu, log_variance)
                spec_dist = spectral_distances(sr=SAMPLE_RATE)
                spec_loss = spec_dist(x_hat, waveform)
                # Notes this won't work when using grains, need to look into this
                if ENV_DIST > 0:
                    env_loss =  envelope_distance(x_hat, waveform, n_fft=1024,log=True)
                else:
                    env_loss = 0

                loss = (kld*BETA) + spec_loss + (env_loss*ENV_DIST)

                # Compute gradients and update weights
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step

                # Accumulate loss for reporting
                running_train_loss += loss.item() 
                running_kl_loss += kld.item()
                running_spec_loss += spec_loss.item()
                running_env_loss += env_loss

            # get avg training statistics 
            train_loss = running_train_loss/len(train_dataloader) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = running_kl_loss/len(train_dataloader)
            spec_loss = running_spec_loss/len(train_dataloader)
            env_loss = running_env_loss/len(train_dataloader)

            # Validate - turn gradient tracking off for validation. 
            model.eval()
            
            #################
            # Validation loop - maybe abstract this out
            #################
            running_val_loss = 0.0
            running_kl_val_loss = 0.0
            running_spec_val_loss = 0.0
            running_env_val_loss = 0.0

            with torch.no_grad():
                for data in val_dataloader:
                    waveform, label = data 
                    waveform = waveform.to(DEVICE)
                    x_hat, z, mu, log_variance = model(waveform)
                    # Compute loss
                    kld = compute_kld(mu, log_variance)
                    spec_dist = spectral_distances(sr=SAMPLE_RATE)
                    spec_loss = spec_dist(x_hat, waveform)
                    # Notes this won't work when using grains, need to look into this
                    if ENV_DIST > 0:
                        env_loss =  envelope_distance(x_hat, waveform, n_fft=1024,log=True)
                    else:
                        env_loss = 0

                    loss = (kld*BETA) + spec_loss + (env_loss*ENV_DIST)

                    running_val_loss += loss.item()
                    running_kl_val_loss += kld.item()
                    running_spec_val_loss += spec_loss.item()
                    running_env_val_loss += env_loss
                
                # Get avg stats
                val_loss = running_val_loss/len(val_dataloader)
                kl_val_loss = running_kl_val_loss/len(val_dataloader)
                spec_val_loss = running_spec_val_loss/len(val_dataloader)
                env_val_loss = running_env_val_loss/len(val_dataloader)

            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "spec_loss": spec_loss, "env_loss": env_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "spec_val_loss": spec_val_loss, "env_val_loss": env_val_loss, "val_loss": val_loss})

            print('Epoch: {}'.format(epoch+1),
            '\tTraining Loss: {:.4f}'.format(train_loss))
            print(f'Validations Loss: {val_loss}')

            if SAVE_CHECKPOINT:
                if (epoch+1) % CHECKPOINT_REGULAIRTY == 0:
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{BETA}beta_{ENV_DIST}envdist_{datetime.now()}.pt")
                    # Save as latest also
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{BETA}beta_{ENV_DIST}envdist_latest.pt")

    else:

        print("----- Inference Mode -----")

        ###########
        # Inference
        ########### 

        # with torch.no_grad():
        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_FILE_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])

        # Put model in eval mode
        model.eval()

        # Lets get batch of test images
        dataiter = iter(test_dataloader)
        signal, labels = next(dataiter)
        print(signal.shape)
        print(labels.shape)
        signal = signal.to(DEVICE)
            
        x_hat, z, mu, logvar = model(signal)                     # get sample outputs
        
        z = z.reshape(z.shape[0] ,1, z.shape[1])
        z = z.detach()

        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)


        classes = ["air_conditioner", 
                    "car_horn", 
                    "children_playing", 
                    "dog_bark", 
                    "drilling", 
                    "engine_idling", 
                    "gun_shot", 
                    "jackhammer", 
                    "siren", 
                    "street_music"]

        if VIEW_LATENT:
            plot_latents(z,labels, classes,"./")
        
        if SAVE_RECONSTRUCTIONS:
            for i, signal in enumerate(x_hat):
                torchaudio.save(f"./audio_tests/usd_vae_{classes[labels[i]]}_{i}.wav", signal, SAMPLE_RATE)
                print(f'{classes[labels[i]]} saved')


