import sys
sys.path.append('../')

from models.waveform_models.usd_waveform_model import WaveformEncoder, WaveformDecoder, WaveformVAE
from models.dataloaders.waveform_dataloaders import ESC50WaveformDataset, make_audio_dataloaders
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance
from scripts.configs.hyper_parameters_waveform import *
from utils.utilities import plot_latents, export_latents


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
        project="SeaWaves_waveformVAE",
        name= f"run_{datetime.now()}",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Waveform_VAE",
        "dataset": "UrbanSound8K",
        "epochs": EPOCHS,
        "latent size": LATENT_SIZE,
        "env_dist": ENV_DIST,
        "beta": BETA,
        "grain_length": GRAIN_LENGTH
        }
    )

if __name__ == "__main__":


    train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

    # Test dataloader
    test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)

    model = WaveformVAE(n_grains = n_grains,
                    hop_size=hop_size,
                    normalize_ola=NORMALIZE_OLA,
                    pp_chans=POSTPROC_CHANNELS,
                    pp_ker=POSTPROC_KER_SIZE,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    n_linears=3,
                    num_samples=l_grain,
                    l_grain=l_grain,
                    h_dim=512,
                    z_dim=128)
    
    model.to(DEVICE)

    # # Split into train and validation
    # # Program this better
    # usd_waveforms = ESC50WaveformDataset(ANNOTATIONS_FILE, AUDIO_DIR, None, SAMPLE_RATE, NUM_SAMPLES, DATALOADER_DEVICE)
    # split = [(5*len(usd_waveforms))//6, (1*len(usd_waveforms))//6+1]
    # train_set, val_set = torch.utils.data.random_split(usd_waveforms, split)
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)
    # val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    # print("Sizes")
    # print(len(train_dataloader))
    # print(len(val_dataloader))


    if TRAIN:

        print("-------- Training Mode --------")

        ###########
        # Training
        ########### 

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

        start_epoch = 0

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
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
            train_loss = running_train_loss/len(train_dataloader.dataset) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = running_kl_loss/len(train_dataloader.dataset)
            spec_loss = running_spec_loss/len(train_dataloader.dataset)
            env_loss = running_env_loss/len(train_dataloader.dataset)

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
                val_loss = running_val_loss/len(val_dataloader.dataset)
                kl_val_loss = running_kl_val_loss/len(val_dataloader.dataset)
                spec_val_loss = running_spec_val_loss/len(val_dataloader.dataset)
                env_val_loss = running_env_val_loss/len(val_dataloader.dataset)

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
                        }, f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{BETA}beta_{ENV_DIST}envdist_{epoch+1}epoch_{datetime.now()}.pt")
                    # Save as latest also
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{BETA}beta_{ENV_DIST}envdist_latest.pt")

    elif EXPORT_LATENTS:

        print("-------- Exporting Latents --------")

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(DEVICE)
        model.eval()

        train_latents,train_labels,test_latents,test_labels = export_latents(model,test_dataloader,test_dataloader)
        # train_latents,train_labels,test_latents,test_labels = export_latents(model,train_dataloader,val_dataloader)
        
        print("-------- Done Exporting Latents --------")


    else:

        print("-------- Inference Mode --------")

        ###########
        # Inference
        ########### 

        # with torch.no_grad():
        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])

        # Put model in eval mode
        model.to(DEVICE)
        model.eval()

        # Lets get batch of test images
        dataiter = iter(test_dataloader)
        waveforms, labels = next(dataiter)
        # print(signal.shape)
        # print(labels.shape)
        waveforms = waveforms.to(DEVICE)

            
        x_hat, z, mu, logvar = model(waveforms)                     # get sample outputs

        print("Z shape:", z.shape)

        

        spec_dist = spectral_distances(sr=SAMPLE_RATE)

        # spec_loss = spec_dist(x_hat, waveforms)
        # print("Average: ", spec_loss)

        z = z.reshape(z.shape[0] ,1, z.shape[1])
        z = z.detach()


        # if VIEW_LATENT:
        #     plot_latents(z,labels, classes,"./")
        if COMPARE_ENERGY:
            for i, signal in enumerate(x_hat):
                # Check the energy differences
                # print(labels[i][:-4])
                print("Reconstruction Energy    : ", (x_hat[i] * x_hat[i]).sum().data)
                print("Original Energy          : ", (waveforms[i] * waveforms[i]).sum().data)
                print("Average Reconstruction Energy    : ", (x_hat[i] * x_hat[i]).sum().data/x_hat[i].shape[0])
                print("Average Original Energy          : ", (waveforms[i] * waveforms[i]).sum().data/waveforms[i].shape[0])


        if SAVE_RECONSTRUCTIONS:
            for i, signal in enumerate(x_hat):
                # torchaudio.save(f"./audio_tests/usd_vae_{classes[labels[i]]}_{i}.wav", signal, SAMPLE_RATE)
                spec_loss = spec_dist(x_hat[i], waveforms[i])
                # Check the energy differences
                # print("Saving ", labels[i][:-4])
                print("Saving ", i)
                print("Loss: ", spec_loss)
                # torchaudio.save(f"./audio_tests/reconstructions/2048/recon_{labels[i][:-4]}_{spec_loss}.wav", signal, SAMPLE_RATE)
                # torchaudio.save(f"./audio_tests/reconstructions/2048/{labels[i][:-4]}.wav", waveforms[i], SAMPLE_RATE)
                torchaudio.save(f"./audio_tests/reconstructions/one_sec/recon_{i}_{spec_loss}.wav", signal.unsqueeze(0), SAMPLE_RATE)
                torchaudio.save(f"./audio_tests/reconstructions/one_sec/{i}.wav", waveforms[i].unsqueeze(0), SAMPLE_RATE)
                # print(f'{classes[labels[i]]} saved')


