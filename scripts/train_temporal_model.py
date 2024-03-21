import sys
sys.path.append('../')

from models.noiseFiltering_models.spectral_shape_model import SpectralVAE_v1, SpectralVAE_v2
from models.temporal_models.temporal_model import LatentVAE
from models.dataloaders.waveform_dataloaders import  make_audio_dataloaders
from models.dataloaders.latent_dataloaders import make_latent_dataloaders
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance
from scripts.configs.hyper_parameters_temporal import *
# from scripts.configs.hyper_parameters_waveform import LATENT_SIZE, AUDIO_DIR, SAMPLE_RATE, NORMALIZE_OLA, POSTPROC_KER_SIZE, POSTPROC_CHANNELS, HOP_SIZE_RATIO, GRAIN_LENGTH, TARGET_LENGTH, HIGH_PASS_FREQ
from scripts.configs.hyper_parameters_spectral import LATENT_SIZE, AUDIO_DIR, TEST_AUDIO_DIR, SAMPLE_RATE, NORMALIZE_OLA, POSTPROC_KER_SIZE, POSTPROC_CHANNELS, HOP_SIZE_RATIO, GRAIN_LENGTH, TARGET_LENGTH, HIGH_PASS_FREQ, H_DIM
from utils.utilities import plot_latents, export_latents, init_beta, export_embedding_to_audio_reconstructions, export_random_samples


import torch
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import pickle
import time
import wandb
import numpy as np
from datetime import datetime

print(LATENT_SIZE)
print(LOAD_WAVEFORM_CHECKPOINT)

# start a new wandb run to track this script
if WANDB:
    wandb.login(key='31e9e9ed4e2efc0f50b1e6ffc9c1e6efae114bd2')
    wandb.init(
        # set the wandb project where this run will be logged
        project="temporalModel",
        name= f"run_{datetime.now()}",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Latent_VAE",
        "dataset": "UrbanSound8K",
        "epochs": EPOCHS,
        "latent size": LATENT_SIZE,
        "env_dist": ENV_DIST,
        "tar_beta": TARGET_BETA,
        "grain_length": GRAIN_LENGTH
        }
    )

if __name__ == "__main__":


    print("-------- Load model and exporting Latents --------")

    train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

    # Test dataloader
    # test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
    # test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)
    test_dataloader, _, _, _, _, _, _, _ = make_audio_dataloaders(data_dir=TEST_AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=TEST_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

    w_model = SpectralVAE_v1(n_grains=n_grains, l_grain=l_grain, h_dim=H_DIM, z_dim=LATENT_SIZE)
    # w_model = SpectralVAE_v2(n_grains=n_grains, l_grain=l_grain, h_dim=[2048, 1024, 512], z_dim=LATENT_SIZE)
    # w_model = SpectralVAE_v3(n_grains=n_grains, l_grain=l_grain, h_dim=[2048, 1024, 512], z_dim=LATENT_SIZE, channels = 32, kernel_size = 3, stride = 2)
    
    if LOAD_WAVEFORM_CHECKPOINT:
        checkpoint = torch.load(WAVEFORM_CHECKPOINT_LOAD_PATH)
        w_model.load_state_dict(checkpoint['model_state_dict'])

    w_model.to(DEVICE)
    w_model.eval()

    print("--- Exporting latents")

    # train_latents,train_labels,val_latents,val_labels = export_latents(w_model,train_dataloader,val_dataloader, DEVICE)
    # # train_latents,train_labels,val_latents,val_labels = export_latents(w_model,test_dataloader,test_dataloader, DEVICE)
    # test_latents,test_labels,_,_ = export_latents(w_model,test_dataloader,test_dataloader, DEVICE)
    train_latents,train_labels,val_latents,val_labels = export_latents(w_model,train_dataloader,val_dataloader, l_grain, n_grains, hop_size, BATCH_SIZE,DEVICE)
    test_latents,test_labels, _, _ = export_latents(w_model,test_dataloader,test_dataloader, l_grain, n_grains, hop_size, TEST_SIZE, DEVICE)

    print("--- Creating dataset")

    train_latentloader,val_latentloader = make_latent_dataloaders(train_latents,train_labels,val_latents,val_labels, batch_size=BATCH_SIZE ,num_workers=0)
    # train_latentloader,val_latentloader = make_latent_dataloaders(train_latents,train_labels,val_latents,val_labels, batch_size=10 ,num_workers=0)
    test_latentloader, _ = make_latent_dataloaders(test_latents,test_labels,test_latents,test_labels, batch_size=TEST_SIZE ,num_workers=0)

    l_model = LatentVAE(
        e_dim=TEMPORAL_LATENT_SIZE,
        z_dim=LATENT_SIZE,
        h_dim=HIDDEN_SIZE,
        n_linears=NO_LINEAR_LAYERS,
        n_grains=n_grains,
        rnn_type=RNN_TYPE,
        n_RNN=NO_RNN_LAYERS,
        classes=["seaWaves"],
        conditional=False
    )

    l_model.to(DEVICE)

    if TRAIN:

        print("-------- Training Mode --------")

        ###########
        # Training
        ########### 

        optimizer = torch.optim.Adam(l_model.parameters(),lr=LEARNING_RATE)

        start_epoch = 0
        accum_iter = 0


        # Calculate the max number of steps based on the number of epochs, number_of_epochs * batches_in_single_epoch
        max_steps = EPOCHS * len(train_dataloader)
        beta, beta_step_val, beta_step_size, warmup_start = init_beta(max_steps, TARGET_BETA, BETA_STEPS, BETA_WARMUP_START_PERC)

        mse_loss = nn.MSELoss()

        if LOAD_LATENT_CHECKPOINT:
            checkpoint = torch.load(LATENT_CHECKPOINT_LOAD_PATH)
            l_model.load_state_dict(checkpoint['model_state_dict'])
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

        for epoch in range(start_epoch, EPOCHS):

            # Turn gradient trackin on for training loop
            l_model.train()

            ###############
            # Training loop - maybe abstract this out
            ###############
            running_train_loss = 0.0
            running_kl_loss = 0.0
            running_rec_loss = 0.0
            running_train_sample_count = 0

            for data in train_latentloader:

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

                latent, label = data 
                # print("--- ",latent.shape)
                latent = Variable(latent).to(DEVICE)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients

                # Note that conds is just the label (numeric)
                z_hat, e, mu, log_variance = l_model(latent, conds=label)                 # forward pass: compute predicted outputs 
                # print("--- ", z_hat.shape)

                # Compute loss
                # [bs, latent_size]
                rec_loss = mse_loss(z_hat,latent) # we train with a deterministic output
                # print("--- MSE Loss: ", rec_loss)
                # TODO: compare with gaussian output and KLD distance ?
                if beta>0:
                    kld_loss = compute_kld(mu,log_variance)*beta
                else:
                    kld_loss = 0

                loss = kld_loss + rec_loss 

                # Compute gradients and update weights
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step

                # Accumulate loss for reporting
                running_train_loss += loss
                running_kl_loss += kld_loss
                running_rec_loss += rec_loss

                accum_iter+=1
                
            # get avg training statistics - note that I'm dividing by size of data set but the last batch is thrown out
            # train_loss = running_train_loss # does len(fsdd_dataloader) return the number of batches ?
            # kl_loss = running_kl_loss
            # rec_loss = running_rec_loss
            train_loss = running_train_loss/len(train_latentloader) # does len(fsdd_dataloader) return the number of batches ?
            kl_loss = running_kl_loss/len(train_latentloader)
            rec_loss = running_rec_loss/len(train_latentloader)
            # print("Training Loss: ", train_loss)
            # print("KL Loss: ", kl_loss)

            # Validate - turn gradient tracking off for validation. 
            l_model.eval()
            
            #################
            # Validation loop - maybe abstract this out
            #################
            running_val_loss = 0.0
            running_kl_val_loss = 0.0
            running_rec_val_loss = 0.0

            with torch.no_grad():
                for data in val_latentloader:
                    latent, label = data 
                    latent = latent.to(DEVICE)
                    z_hat, e, mu, log_variance = l_model(latent, conds=label)

                    # Compute loss
                    rec_loss = mse_loss(z_hat, latent)
                    if beta > 0:
                        kld_loss = compute_kld(mu, log_variance) * beta
                    else:
                        kld_loss = 0.0

                    loss = kld_loss + rec_loss 

                    running_val_loss += loss
                    running_kl_val_loss += kld_loss
                    running_rec_val_loss += rec_loss
                
                # Get avg stats

                val_loss = running_val_loss/len(val_latentloader)
                kl_val_loss = running_kl_val_loss/len(val_latentloader)
                rec_val_loss = running_rec_val_loss/len(val_latentloader)

            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "recon_loss": rec_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "recon_val_loss": rec_val_loss, "val_loss": val_loss})

            print('Epoch: {}'.format(epoch+1),
            '\tStep: {}'.format(accum_iter+1),
            '\t Beta: {:.5f}'.format(beta),
            '\tTraining Loss: {:.8f}'.format(train_loss),
            '\tValidations Loss: {:.8f}'.format(val_loss))

            if SAVE_CHECKPOINT:
                if (epoch+1) % CHECKPOINT_REGULAIRTY == 0:
                    torch.save({
                        'epoch': epoch+1,
                        'accum_iter': accum_iter,
                        'beta': beta,
                        'beta_step_val': beta_step_val,
                        'beta_step_size': beta_step_size,
                        'warmup_start': warmup_start,
                        'model_state_dict': l_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"{SAVE_MODEL_DIR}/latent_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{beta}beta_{ENV_DIST}envdist_{epoch+1}epoch_{datetime.now()}.pt")
                    # Save as latest also
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': l_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"{SAVE_MODEL_DIR}/latent_vae_latest.pt")

    if EXPORT_AUDIO_RECON:

        print("-------- Exporting Audio Reconstructions --------")

        if LOAD_LATENT_CHECKPOINT:
            checkpoint = torch.load(LATENT_CHECKPOINT_LOAD_PATH)
            l_model.load_state_dict(checkpoint['model_state_dict'])

        for batch in test_latentloader:
            export_embedding_to_audio_reconstructions(l_model, w_model, batch, EXPORT_AUDIO_DIR, SAMPLE_RATE, DEVICE,trainset=True)
        
        print("-------- Exporting Audio Reconstructions DONE --------")


        #print("-------- Exporting Random Latent Audio Reconstructions --------")

        #export_random_samples(l_model,w_model, EXPORT_RANDOM_LATENT_AUDIO_DIR, LATENT_SIZE, TEMPORAL_LATENT_SIZE,SAMPLE_RATE, ["SeaWaves"], DEVICE)

        #print("-------- Exporting Random Latent Audio Reconstructions Done --------")

    #     model.to(DEVICE)
    #     model.eval()

    #     train_latents,train_labels,test_latents,test_labels = export_latents(model,test_dataloader,test_dataloader)
    #     # train_latents,train_labels,test_latents,test_labels = export_latents(model,train_dataloader,val_dataloader)
        
    #     print("-------- Done Exporting Latents --------")


    # else:

    #     print("-------- Inference Mode --------")

    #     ###########
    #     # Inference
    #     ########### 

    #     # with torch.no_grad():
    #     if LOAD_CHECKPOINT:
    #         checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
    #         model.load_state_dict(checkpoint['model_state_dict'])

    #     # Put model in eval mode
    #     model.to(DEVICE)
    #     model.eval()

    #     # Lets get batch of test images
    #     dataiter = iter(test_dataloader)
    #     waveforms, labels = next(dataiter)
    #     # print(signal.shape)
    #     # print(labels.shape)
    #     waveforms = waveforms.to(DEVICE)

            
    #     x_hat, z, mu, logvar = model(waveforms)                     # get sample outputs

    #     print("Z shape:", z.shape)

        

    #     spec_dist = spectral_distances(sr=SAMPLE_RATE)

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
    #             torchaudio.save(f"./audio_tests/reconstructions/one_sec/recon_{i}_{spec_loss}.wav", signal.unsqueeze(0), SAMPLE_RATE)
    #             torchaudio.save(f"./audio_tests/reconstructions/one_sec/{i}.wav", waveforms[i].unsqueeze(0), SAMPLE_RATE)
    #             # print(f'{classes[labels[i]]} saved')


