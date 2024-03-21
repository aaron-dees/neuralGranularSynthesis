import sys
sys.path.append('../')

from models.noiseFiltering_models.waveform_model import WaveformVAE
from models.dataloaders.waveform_dataloaders import make_audio_dataloaders
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance
from scripts.configs.hyper_parameters_waveform import *
from utils.utilities import plot_latents, export_latents, init_beta, print_spectral_shape, filter_spectral_shape
from utils.dsp_components import safe_log10, noise_filtering, mod_sigmoid
import torch_dct as dct
import librosa


import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from torch.autograd import Variable
from scipy import signal
import pickle
import time
import wandb
import numpy as np
from datetime import datetime

print("--- Device: ", DEVICE)
# print("--- Venv: ", sys.prefix)
# sdcsff

# start a new wandb run to track this script
if WANDB:
    wandb.login(key='31e9e9ed4e2efc0f50b1e6ffc9c1e6efae114bd2')
    wandb.init(
        # set the wandb project where this run will be logged
        project="SeaWaves_ccVAE_CPU_Local",
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

    print("-----Dataset Loaded-----")
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
                z_dim=LATENT_SIZE)
    
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
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=4000)
        # decayRate = 0.99
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

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
            # accum_iter = checkpoint['accum_iter']
            beta = checkpoint['beta']
            beta_step_val = checkpoint['beta_step_val']
            beta_step_size = checkpoint['beta_step_size']
            warmup_start = checkpoint['warmup_start']


            print("----- Checkpoint File Loaded -----")
            print(f'Epoch: {start_epoch}')
            print(f'Loss: {train_loss}')

        # TEST
        # accum_iter = 100
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
                # x_hat, z = model(waveform)                 # forward pass: compute predicted outputs 

                # ---------- Turn Waveform into grains ----------
                # This turns our sample into overlapping grains
                ola_window = signal.hann(l_grain,sym=False)
                ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
                ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
                ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
                ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

                slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
                mb_grains = F.conv1d(waveform.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
                mb_grains = mb_grains.permute(0,2,1)
                bs = mb_grains.shape[0]
                # repeat the overlap add windows acrosss the batch and apply to the grains
                mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
                grain_fft = torch.fft.rfft(mb_grains)

                # ---------- Turn Waveform into grains END ----------

                # ---------- Get CCs, or MFCCs and invert ----------
                # CCs
                grain_db = 20*safe_log10(torch.abs(grain_fft))
                cepstral_coeff = dct.dct(grain_db)
                # Take the first n_cc cepstral coefficients
                cepstral_coeff[:, :,NUM_CC:] = 0
                inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)

                # MFCCs  - use librosa function as they are more reliable
                # Optimise this for speed
                # grain_fft = grain_fft.permute(0,2,1)
                # grain_mel= safe_log10(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft.cpu().numpy())**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
                # mfccs = dct.dct(grain_mel)
                # inv_mfccs = dct.idct(mfccs).cpu().numpy()       
                # inv_mfccs = torch.from_numpy(librosa.feature.inverse.mel_to_stft(M=10**inv_mfccs, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH)).to(DEVICE)
                # inv_mfccs = inv_mfccs.permute(0,2,1)
                
                # ---------- Get CCs, or MFCCs and invert END ----------

                # ---------- Run Model ----------

                x_hat, z, mu, log_variance = model(waveform)   
                # # x_hat, z, mu, log_variance = model(inv_mfccs)   

                # # ---------- Run Model END ----------


                # # ---------- Noise Filtering ----------
                # # Step 5 - Noise Filtering
                # # Reshape for noise filtering - TODO Look if this is necesary
                # x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

                # # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
                # filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)
                # audio = noise_filtering(x_hat, filter_window, n_grains, l_grain, HOP_SIZE_RATIO)

                # # ---------- Noise Filtering END ----------

                # # ---------- Concatonate Grains ----------

                # # Check if number of grains wanted is entered, else use the original
                # if n_grains is None:
                #     audio = audio.reshape(-1, n_grains, l_grain)
                # else:
                #     audio = audio.reshape(-1,n_grains,l_grain)
                # bs = audio.shape[0]

                # # Check if an overlapp add window has been passed, if not use that used in encoding.
                # audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

                # # Folder
                # # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
                # # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
                # ola_folder = nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
                # # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
                # # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
                # # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
                # # since kernel size is l_grain, this is needed in the second dimension.
                # audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

                # # Normalise the energy values across the audio samples
                # if NORMALIZE_OLA:
                #     unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
                #     input_ones = torch.ones(1,1,tar_l,1)
                #     ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
                #     ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(DEVICE)
                #     audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

                # # ---------- Concatonate Grains END ----------
                    
                # # ---------- Postprocessing Kernel ----------

                # # NOTE Removed the post processing step for now
                # # This module applies a multi-channel temporal convolution that
                # # learns a parallel set of time-invariant FIR filters and improves
                # # the audio quality of the assembled signal.
                # # TODO Add back in ?
                # # audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)
                    
                # # ---------- Postprocessing Kernel END ----------
                    
                # # ---------- Normalise the audio ----------
                    
                # # Normalise the audio, as is done in dataloader.
                # audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
                # audio_sum = audio_sum * 0.9

                # ---------- Normalise the audio END ----------

                # Compute loss
                spec_loss = spec_dist(x_hat, waveform)
                if beta > 0:
                    kld_loss = compute_kld(mu, log_variance) * beta
                else:
                    kld_loss = 0.0
                # Notes this won't work when using grains, need to look into this
                if ENV_DIST > 0:
                    env_loss =  envelope_distance(x_hat, waveform, n_fft=1024,log=True) * ENV_DIST
                else:
                    env_loss = 0.0

                loss = kld_loss + spec_loss + env_loss
                # loss = spec_loss

                # Compute gradients and update weights
                loss.backward()                         # backward pass
                optimizer.step()                        # perform optimization step

                # Accumulate loss for reporting
                running_train_loss += loss
                running_kl_loss += kld_loss
                running_spec_loss += spec_loss
                running_env_loss += env_loss

                accum_iter+=1

            # Decay the learning rate
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
                
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
                    
                    # ---------- Turn Waveform into grains ----------
                    ola_window = signal.hann(l_grain,sym=False)
                    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
                    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
                    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
                    ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

                    slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
                    mb_grains = F.conv1d(waveform.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
                    mb_grains = mb_grains.permute(0,2,1)
                    bs = mb_grains.shape[0]
                    # repeat the overlap add windows acrosss the batch and apply to the grains
                    mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
                    grain_fft = torch.fft.rfft(mb_grains)
                    # ---------- Turn Waveform into grains END ----------

                    # ---------- Get CCs, or MFCCs and invert ----------
                    # CCs
                    grain_db = 20*safe_log10(torch.abs(grain_fft))
                    cepstral_coeff = dct.dct(grain_db)
                    # Take the first n_cc cepstral coefficients
                    cepstral_coeff[:, :,NUM_CC:] = 0
                    inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)

                    # MFCCs  - use librosa function as they are more reliable
                    # grain_fft = grain_fft.permute(0,2,1)
                    # grain_mel= safe_log10(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft.cpu().numpy())**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
                    # mfccs = dct.dct(grain_mel)
                    # inv_mfccs = dct.idct(mfccs).cpu().numpy()       
                    # inv_mfccs = torch.from_numpy(librosa.feature.inverse.mel_to_stft(M=10**inv_mfccs, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH)).to(DEVICE)
                    # inv_mfccs = inv_mfccs.permute(0,2,1)

                    # ---------- Get CCs, or MFCCs and invert END ----------

                    # ---------- Run Model ----------

                    x_hat, z, mu, log_variance = model(waveform)   
                    # x_hat, z, mu, log_variance = model(inv_mfccs)   

                    # # ---------- Run Model END ----------

                    # # ---------- Noise Filtering ----------

                    # # Reshape for noise filtering - TODO Look if this is necesary
                    # x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

                    # # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
                    # filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)
                    # audio = noise_filtering(x_hat, filter_window, n_grains, l_grain, HOP_SIZE_RATIO)

                    # # ---------- Noise Filtering END ----------

                    # # ---------- Concatonate Grains ----------

                    # # Check if number of grains wanted is entered, else use the original
                    # if n_grains is None:
                    #     audio = audio.reshape(-1, n_grains, l_grain)
                    # else:
                    #     audio = audio.reshape(-1,n_grains,l_grain)
                    # bs = audio.shape[0]

                    # # Check if an overlapp add window has been passed, if not use that used in encoding.
                    # audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

                    # # Folder
                    # # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
                    # # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
                    # ola_folder = nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
                    # # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
                    # # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
                    # # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
                    # # since kernel size is l_grain, this is needed in the second dimension.
                    # audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

                    # # Normalise the energy values across the audio samples
                    # if NORMALIZE_OLA:
                    #     unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
                    #     input_ones = torch.ones(1,1,tar_l,1)
                    #     ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
                    #     ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(DEVICE)
                    #     audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

                    # # ---------- Concatonate Grains END ----------
                        
                    # # ---------- Post Processing Kernel ----------

                    # # NOTE Removed the post processing step for now
                    # # This module applies a multi-channel temporal convolution that
                    # # learns a parallel set of time-invariant FIR filters and improves
                    # # the audio quality of the assembled signal.
                    # # TODO Add back in ?
                    # # audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)
                        
                    # # ---------- Post Processing Kernel END ----------
                        
                    # # ---------- Normalise the audio ----------

                    # # Normalise the audio, as is done in dataloader.
                    # audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
                    # audio_sum = audio_sum * 0.9

                    # ---------- Normalise the audio END ----------

                    # Compute loss
                    spec_loss = spec_dist(x_hat, waveform)
                    if beta > 0:
                        kld_loss = compute_kld(mu, log_variance) * beta
                    else:
                        kld_loss = 0.0
                    # Notes this won't work when using grains, need to look into this
                    if ENV_DIST > 0:
                        env_loss =  envelope_distance(x_hat, waveform, n_fft=l_grain,log=True) * ENV_DIST
                    else:
                        env_loss = 0.0

                    loss = kld_loss + spec_loss + env_loss
                    # loss = spec_loss 

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

            if SAVE_RECONSTRUCTIONS:
                if (epoch+1) % CHECKPOINT_REGULAIRTY == 0:
                    for i, recon_signal in enumerate(x_hat):
                        # spec_loss = spec_dist(x_hat[i], waveforms[i])
                        # Check the energy differences
                        print("Saving ", i)
                        print("Loss: ", spec_loss)
                        torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/CC_recon_{i}_{spec_loss}_{epoch+1}.wav', recon_signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                        # torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/CC_{i}.wav", waveform[i].unsqueeze(0).cpu(), SAMPLE_RATE)
                        torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/CC_{i}.wav", waveform[i].unsqueeze(0).cpu(), SAMPLE_RATE)

            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "spec_loss": train_spec_loss, "env_loss": env_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "spec_val_loss": spec_val_loss, "env_val_loss": env_val_loss, "val_loss": val_loss})

            print('Epoch: {}'.format(epoch+1),
            '\tStep: {}'.format(accum_iter+1),
            '\t KL Loss: {:.5f}'.format(kl_loss),
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
                        'accum_iter': accum_iter,
                        'beta': beta,
                        'beta_step_val': beta_step_val,
                        'beta_step_size': beta_step_size,
                        'warmup_start': warmup_start,
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

            # This turns our sample into overlapping grains
            ola_window = signal.hann(l_grain,sym=False)
            ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
            ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
            ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
            ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

            slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
            mb_grains = F.conv1d(waveforms.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
            mb_grains = mb_grains.permute(0,2,1)
            bs = mb_grains.shape[0]
            # repeat the overlap add windows acrosss the batch and apply to the grains
            mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
            grain_fft = torch.fft.rfft(mb_grains)

            # CCs
            grain_db = 20*safe_log10(torch.abs(grain_fft))
            cepstral_coeff = dct.dct(grain_db)
            # Take the first n_cc cepstral coefficients
            cepstral_coeff[:, :,NUM_CC:] = 0
            inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)

            # MFCCs  - use librosa function as they are more reliable
            grain_fft = grain_fft.permute(0,2,1)
            grain_mel= safe_log10(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft)**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
            mfccs = dct.dct(grain_mel)
            inv_mfccs = dct.idct(mfccs).cpu().numpy()       
            inv_mfccs = torch.from_numpy(librosa.feature.inverse.mel_to_stft(M=10**inv_mfccs, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH))
            inv_mfccs = inv_mfccs.permute(0,2,1)

            x_hat, z, mu, log_variance = model(inv_cep_coeffs)   
             # return the spectral shape 
            # CC part
            
            # filter_coeffs = torch.zeros(x_hat.shape[0], x_hat.shape[1], int(l_grain/2)+1)
            # filter_coeffs[:, :, :NUM_CC] = x_hat

            # # NOTE Do i need to  do the scaling back from decibels, also note this introduces 
            # # NOTE is there a torch implementation of this, bit of a bottleneck if not?
            # # NOTE issue with gradient flowwing back
            # # NOTE changed this from idct_2d to idct
            # # inv_filter_coeffs = (dct.idct(filter_coeffs))
            # inv_filter_coeffs = 10**(dct.idct(filter_coeffs) / 20)
            # x_hat = inv_filter_coeffs

            # get sample outputs
            # Step 5 - Noise Filtering
            # Reshape for noise filtering - TODO Look if this is necesary
            x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

            # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
            filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)
            audio = noise_filtering(x_hat, filter_window, n_grains, l_grain, HOP_SIZE_RATIO)

            # Check if number of grains wanted is entered, else use the original
            if n_grains is None:
                audio = audio.reshape(-1, n_grains, l_grain)
            else:
                audio = audio.reshape(-1,n_grains,l_grain)
            bs = audio.shape[0]

            # Check if an overlapp add window has been passed, if not use that used in encoding.
            audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

            # Folder
            # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
            # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
            ola_folder = nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
            # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
            # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
            # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
            # since kernel size is l_grain, this is needed in the second dimension.
            audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

            # Normalise the energy values across the audio samples
            if NORMALIZE_OLA:
                unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
                input_ones = torch.ones(1,1,tar_l,1)
                ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
                ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(DEVICE)
                audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

            # NOTE Removed the post processing step for now
            # This module applies a multi-channel temporal convolution that
            # learns a parallel set of time-invariant FIR filters and improves
            # the audio quality of the assembled signal.
            # TODO Add back in ?
            # audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)


            # Normalise the audio, as is done in dataloader.
            audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
            audio_sum = audio_sum * 0.9

            spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
            spec_loss = spec_dist(audio_sum, waveforms)

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
                    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/CC_recon_{i}_{spec_loss}.wav', signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                    torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/CC_{i}.wav", waveforms[i].unsqueeze(0).cpu(), SAMPLE_RATE)
                    # print(f'{classes[labels[i]]} saved')


