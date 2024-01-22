import sys
sys.path.append('../')

from models.waveform_models.waveform_model import WaveformVAE, CepstralCoeffsAE
from models.dataloaders.waveform_dataloaders import make_audio_dataloaders
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance
from scripts.configs.hyper_parameters_waveform import *
from utils.utilities import plot_latents, export_latents, init_beta, print_spectral_shape, filter_spectral_shape


import torch
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import pickle
import time
import wandb
import numpy as np
from datetime import datetime

print("--- Device: ", DEVICE)
# print("--- Venv: ", sys.prefix)
print(LATENT_SIZE)

# start a new wandb run to track this script
if WANDB:
    wandb.login(key='31e9e9ed4e2efc0f50b1e6ffc9c1e6efae114bd2')
    wandb.init(
        # set the wandb project where this run will be logged
        project="SeaWaves_ccVAE_GPU",
        name= f"run_{datetime.now()}",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "cc_VAE",
        "dataset": "UrbanSound8K",
        "epochs": EPOCHS,
        "latent size": LATENT_SIZE,
        "env_dist": ENV_DIST,
        "tar_beta": TARGET_BETA,
        "grain_length": GRAIN_LENGTH
        }
    )

if __name__ == "__main__":


    train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

    # Test dataloader
    test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)

    model = CepstralCoeffsAE(n_grains=n_grains, hop_size=hop_size, z_dim=LATENT_SIZE, normalize_ola=NORMALIZE_OLA)
    
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
        accum_iter = 0
        # Calculate the max number of steps based on the number of epochs, number_of_epochs * batches_in_single_epoch
        max_steps = EPOCHS * len(train_dataloader)
        beta, beta_step_val, beta_step_size, warmup_start = init_beta(max_steps, TARGET_BETA, BETA_STEPS, BETA_WARMUP_START_PERC)

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            train_loss = checkpoint['loss']
            accum_iter = checkpoint['accum_iter']
            beta = checkpoint['beta']
            beta_step_val = checkpoint['beta_step_val']
            beta_step_size = checkpoint['beta_step_size']
            warmup_start = checkpoint['warmup_start']


            print("----- Checkpoint File Loaded -----")
            print(f'Epoch: {start_epoch}')
            print(f'Loss: {train_loss}')

        # Model in training mode

        # Set spectral distances
        spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)

        
        for epoch in range(start_epoch, EPOCHS):

            start = time.time()

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

                # set the beta for weighting the KL Divergence
                # note beta will only start on multiple of step size
                if (accum_iter+1)%beta_step_size==0:
                    if accum_iter<warmup_start:
                        beta = 0
                    elif beta<TARGET_BETA:
                        beta += beta_step_val
                        beta = np.min([beta,TARGET_BETA])
                    else:
                        beta = TARGET_BETA

                waveform, label = data 
                waveform = Variable(waveform).to(DEVICE)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients
                x_hat, z = model(waveform)                 # forward pass: compute predicted outputs 
                # x_hat, z, mu, log_variance, spec = model(waveform)                 # forward pass: compute predicted outputs 

                # print(x_hat.sum())

                # Compute loss

                spec_loss = spec_dist(x_hat, waveform)
                # if beta > 0:
                #     kld_loss = compute_kld(mu, log_variance) * beta
                # else:
                #     kld_loss = 0.0
                # # Notes this won't work when using grains, need to look into this
                # if ENV_DIST > 0:
                #     env_loss =  envelope_distance(x_hat, waveform, n_fft=1024,log=True) * ENV_DIST
                # else:
                #     env_loss = 0.0

                kld_loss = 0
                env_loss = 0

                # loss = kld_loss + spec_loss + env_loss
                loss = spec_loss

                # Compute gradients and update weights
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step

                # Accumulate loss for reporting
                running_train_loss += loss
                running_kl_loss += kld_loss
                running_spec_loss += spec_loss
                running_env_loss += env_loss

                accum_iter+=1
                
            # get avg training statistics 
            train_loss = running_train_loss/len(train_dataloader) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = running_kl_loss/len(train_dataloader)
            train_spec_loss = running_spec_loss/len(train_dataloader)
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
                    x_hat, z = model(waveform)
                    # x_hat, z, mu, log_variance, spec = model(waveform)

                    # Compute loss
                    spec_loss = spec_dist(x_hat, waveform)
                    # if beta > 0:
                    #     kld_loss = compute_kld(mu, log_variance) * beta
                    # else:
                    #     kld_loss = 0.0
                    # # Notes this won't work when using grains, need to look into this
                    # if ENV_DIST > 0:
                    #     env_loss =  envelope_distance(x_hat, waveform, n_fft=1024,log=True) * ENV_DIST
                    # else:
                    #     env_loss = 0.0

                    # For VAE
                    kld_loss = 0
                    env_loss = 0

                    # loss = kld_loss + spec_loss + env_loss
                    loss = spec_loss 

                    running_val_loss += loss
                    running_kl_val_loss += kld_loss
                    running_spec_val_loss += spec_loss
                    running_env_val_loss += env_loss
                
                # Get avg stats
                val_loss = running_val_loss/len(val_dataloader)
                kl_val_loss = running_kl_val_loss/len(val_dataloader)
                spec_val_loss = running_spec_val_loss/len(val_dataloader)
                env_val_loss = running_env_val_loss/len(val_dataloader)

            end = time.time()

            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "spec_loss": train_spec_loss, "env_loss": env_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "spec_val_loss": spec_val_loss, "env_val_loss": env_val_loss, "val_loss": val_loss})

            print('Epoch: {}'.format(epoch+1),
            '\tStep: {}'.format(accum_iter+1),
            '\t Beta: {:.5f}'.format(beta),
            '\tTraining Loss: {:.4f}'.format(train_loss),
            '\tValidations Loss: {:.4f}'.format(val_loss),
            '\tTime: {:.2f}s'.format(end-start))

            if SAVE_CHECKPOINT:
                if (epoch+1) % CHECKPOINT_REGULAIRTY == 0:
                    torch.save({
                        'epoch': epoch+1,
                        'accum_iter': accum_iter,
                        'beta': beta,
                        'beta_step_val': beta_step_val,
                        'beta_step_size': beta_step_size,
                        'warmup_start': warmup_start,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                       }, f"{SAVE_DIR}/waveform_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{beta}beta_{ENV_DIST}envdist_{epoch+1}epoch_{datetime.now()}.pt")
                    # Save as latest also
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"{SAVE_DIR}/waveform_vae_latest.pt")

    elif EXPORT_LATENTS:

        print("-------- Exporting Latents --------")

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(DEVICE)
        model.eval()

        train_latents,train_labels,val_latents,val_labels = export_latents(model,test_dataloader,test_dataloader, DEVICE)
        # train_latents,train_labels,val_latents,val_labels = export_latents(model,train_dataloader,val_dataloader, DEVICE)
        
        print("-------- Done Exporting Latents --------")


    else:

        print("-------- Inference Mode --------")


        model = CepstralCoeffsAE(n_grains=n_grains, hop_size=hop_size, z_dim=LATENT_SIZE, normalize_ola=NORMALIZE_OLA)


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

        with torch.no_grad():

            # Lets get batch of test images
            dataiter = iter(test_dataloader)
            waveforms, labels = next(dataiter)

            waveforms = waveforms.to(DEVICE)

            x_hat, z = model(waveforms)                     # get sample outputs

            spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
            spec_loss = spec_dist(x_hat, waveforms)

            print(x_hat.shape)
            print(waveforms.shape)

            print("Spectral Loss: ", spec_loss)

            # print_spectral_shape(waveforms[0,:], spec[0,:,:].cpu().numpy(), hop_size, l_grain)

            filter_spectral_shape(waveforms[0,:], hop_size, l_grain, n_grains, tar_l)

        #     spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)

        #     # spec_loss = spec_dist(x_hat, waveforms)
        #     # print("Average: ", spec_loss)

        #     z = z.reshape(z.shape[0] ,1, z.shape[1])
        #     z = z.detach()

        #     # if VIEW_LATENT:
        #     #     plot_latents(z,labels, classes,"./")

        #     if COMPARE_ENERGY:
        #         for i, signal in enumerate(x_hat):
        #             # Check the energy differences
        #             # print(labels[i][:-4])
        #             print("Reconstruction Energy    : ", (x_hat[i] * x_hat[i]).sum().data)
        #             print("Original Energy          : ", (waveforms[i] * waveforms[i]).sum().data)
        #             print("Average Reconstruction Energy    : ", (x_hat[i] * x_hat[i]).sum().data/x_hat[i].shape[0])
        #             print("Average Original Energy          : ", (waveforms[i] * waveforms[i]).sum().data/waveforms[i].shape[0])

        #     if SAVE_RECONSTRUCTIONS:
        #         for i, signal in enumerate(x_hat):
        #             # torchaudio.save(f"./audio_tests/usd_vae_{classes[labels[i]]}_{i}.wav", signal, SAMPLE_RATE)
        #             spec_loss = spec_dist(x_hat[i], waveforms[i])
        #             # Check the energy differences
        #             # print("Saving ", labels[i][:-4])
        #             print("Saving ", i)
        #             print("Loss: ", spec_loss)
        #             # torchaudio.save(f"./audio_tests/reconstructions/2048/recon_{labels[i][:-4]}_{spec_loss}.wav", signal, SAMPLE_RATE)
        #             # torchaudio.save(f"./audio_tests/reconstructions/2048/{labels[i][:-4]}.wav", waveforms[i], SAMPLE_RATE)
        #             torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_{i}_{spec_loss}.wav', signal.unsqueeze(0).cpu(), SAMPLE_RATE)
        #             torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/{i}.wav", waveforms[i].unsqueeze(0).cpu(), SAMPLE_RATE)
        #             # print(f'{classes[labels[i]]} saved')


