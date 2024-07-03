import sys
sys.path.append('../')

from models.noiseFiltering_models.spectral_shape_model import SpectralVAE_v1, SpectralVAE_v2
from models.dataloaders.waveform_dataloaders import make_audio_dataloaders, make_audio_dataloaders_noPadding
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance, calc_reconstruction_loss
from scripts.configs.hyper_parameters_spectral import *
from utils.utilities import plot_latents, export_latents, init_beta, print_spectral_shape, filter_spectral_shape
from utils.dsp_components import safe_log10, noise_filtering, mod_sigmoid
import torch_dct as dct
import librosa
from frechet_audio_distance import FrechetAudioDistance
import matplotlib.pyplot as plt
import librosa.display


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
        project="fadScoring",
        # name= f"run_{datetime.now()}",
        name= f"run_{datetime.now()}",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "v1",
        "dataset": "Full_Seawaves_UrbanSound8k",
        "epochs": EPOCHS,
        "latent size": LATENT_SIZE,
        "env_dist": ENV_DIST,
        "tar_beta": TARGET_BETA,
        "grain_length": GRAIN_LENGTH
        }
    )

# Evaluation metric
# TODO Do i need to resample audio before saving to 16kHz?
frechet = FrechetAudioDistance(
    model_name="vggish",
    # Do I need to resample these?
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

if __name__ == "__main__":


    train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders_noPadding(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)
    # train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0, center_pad=True)
    # Update the number of grains to account for centre padding
    if (HOP_SIZE_RATIO*100 == 25) :
        n_grains = 4*((tar_l+l_grain)//l_grain)-3
        tar_l = tar_l+(l_grain)
    elif (HOP_SIZE_RATIO*100 == 50) :
        n_grains = 2*(tar_l+l_grain//l_grain)-1
        tar_l = tar_l+(l_grain)
    print(tar_l)
    print(n_grains)
    print(l_grain)
    print(hop_size)

    print("-----Dataset Loaded-----")
    # Test dataloader
    # test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
    # test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)
    test_dataloader, _, _, _, _, _, _, _ = make_audio_dataloaders_noPadding(data_dir=TEST_AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=TEST_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)
    # test_dataloader, _, _, _, _, _, _, _ = make_audio_dataloaders(data_dir=TEST_AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=TEST_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0, center_pad=True)

    model = SpectralVAE_v1(n_grains=n_grains, l_grain=l_grain, h_dim=H_DIM, z_dim=LATENT_SIZE)
    # model = SpectralVAE_v2(n_grains=n_grains, l_grain=l_grain, h_dim=[2048, 1024, 512], z_dim=LATENT_SIZE)
    # model = SpectralVAE_v3(n_grains=n_grains, l_grain=l_grain, h_dim=[2048, 1024, 512], z_dim=LATENT_SIZE, channels = 32, kernel_size = 3, stride = 2)
    
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

                ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32)
                stft_audio = torch.stft(waveform, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
                grain_fft = stft_audio.permute(0,2,1)

                # # ---------- Turn Waveform into grains ----------
                # This turns our sample into overlapping grains
                ola_window = signal.hann(l_grain,sym=False)
                ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
                ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
                ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
                ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

                # slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
                # mb_grains = F.conv1d(waveform.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
                # mb_grains = mb_grains.permute(0,2,1)
                # bs = mb_grains.shape[0]
                # # repeat the overlap add windows acrosss the batch and apply to the grains
                # mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
                # grain_fft = torch.fft.rfft(mb_grains)

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

                grain_fft_abs = torch.abs(grain_fft)
                grain_min = grain_fft_abs.min()
                grain_max = grain_fft_abs.max()
                norm_array = (grain_fft_abs - grain_fft_abs.min()) / (grain_fft_abs.max() - grain_fft_abs.min())
                norm_array = norm_array * (1 - 0) + 0

                # ---------- Run Model ----------

                # x_hat, z, mu, log_variance = model(mb_grains)   
                x_hat, z, mu, log_variance = model(torch.abs(norm_array))   
                # x_hat, z, mu, log_variance = model(torch.abs(grain_fft))   
                # x_hat, z, mu, log_variance = model(inv_mfccs)   

                # ---------- Run Model END ----------

                # denorm array
                denorm_array = (x_hat - 0) / (1 - 0)
                denorm_array = denorm_array * (grain_max - grain_min) + grain_min 

                transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
                audio_sum = transform(torch.abs(denorm_array.permute(0,2,1)))

                ## ---------- Noise Filtering ----------
                ## Step 5 - Noise Filtering
                ## Reshape for noise filtering - TODO Look if this is necesary
                #x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

                ## Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
                #filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)
                #audio = noise_filtering(x_hat, filter_window, n_grains, l_grain, HOP_SIZE_RATIO)

                ## ---------- Noise Filtering END ----------

                ## ---------- Concatonate Grains ----------

                ## Check if number of grains wanted is entered, else use the original
                #if n_grains is None:
                    #audio = audio.reshape(-1, n_grains, l_grain)
                #else:
                    #audio = audio.reshape(-1,n_grains,l_grain)
                #bs = audio.shape[0]

                ## Check if an overlapp add window has been passed, if not use that used in encoding.
                #audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

                ## Folder
                ## Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
                ## can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
                #ola_folder = nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
                ## Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
                ## This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
                ## Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
                ## since kernel size is l_grain, this is needed in the second dimension.
                #audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

                ## Normalise the energy values across the audio samples
                #if NORMALIZE_OLA:
                    #unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
                    #input_ones = torch.ones(1,1,tar_l,1)
                    #ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
                    #ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(DEVICE)
                    #audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

                ## ---------- Concatonate Grains END ----------
                    
                ## ---------- Postprocessing Kernel ----------

                ## NOTE Removed the post processing step for now
                ## This module applies a multi-channel temporal convolution that
                ## learns a parallel set of time-invariant FIR filters and improves
                ## the audio quality of the assembled signal.
                ## TODO Add back in ?
                ## audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)
                    
                ## ---------- Postprocessing Kernel END ----------
                    
                ## ---------- Normalise the audio ----------
                    
                ## Normalise the audio, as is done in dataloader.
                ## audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
                ## audio_sum = audio_sum * 0.9

                ## ---------- Normalise the audio END ----------

                #mu shape: [bs*n_grains, z_dim]
                # Compute loss
                spec_loss = spec_dist(audio_sum, waveform)
                # spec_loss = spec_dist(audio_sum[:, l_grain//2:-l_grain//2], waveform)
                # spec_loss = calc_reconstruction_loss(torch.abs(grain_fft), x_hat_old)
                if beta > 0:
                    kld_loss = compute_kld(mu, log_variance) * beta
                else:
                    kld_loss = 0.0
                # Notes this won't work when using grains, need to look into this
                if ENV_DIST > 0:
                    env_loss =  envelope_distance(audio_sum, waveform, n_fft=1024,log=True) * ENV_DIST
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

                    ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32)
                    stft_audio = torch.stft(waveform, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
                    grain_fft = stft_audio.permute(0,2,1)
                    
                    # # ---------- Turn Waveform into grains ----------
                    ola_window = signal.hann(l_grain,sym=False)
                    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
                    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
                    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
                    ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

                    # slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
                    # mb_grains = F.conv1d(waveform.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
                    # mb_grains = mb_grains.permute(0,2,1)
                    # bs = mb_grains.shape[0]
                    # # repeat the overlap add windows acrosss the batch and apply to the grains
                    # mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
                    # grain_fft = torch.fft.rfft(mb_grains)
                    # # ---------- Turn Waveform into grains END ----------

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
                    grain_fft_abs = torch.abs(grain_fft)
                    grain_min = grain_fft_abs.min()
                    grain_max = grain_fft_abs.max()
                    norm_array = (grain_fft_abs - grain_fft_abs.min()) / (grain_fft_abs.max() - grain_fft_abs.min())
                    norm_array = norm_array * (1 - 0) + 0

                    # ---------- Run Model ----------

                    # x_hat, z, mu, log_variance = model(mb_grains)   
                    x_hat, z, mu, log_variance = model(torch.abs(norm_array))   
                    # x_hat, z, mu, log_variance = model(torch.abs(grain_fft))   
                    # x_hat, z, mu, log_variance = model(inv_mfccs)   

                    # ---------- Run Model END ----------

                    # denorm array
                    denorm_array = (x_hat - 0) / (1 - 0)
                    denorm_array = denorm_array * (grain_max - grain_min) + grain_min 

                    transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
                    audio_sum = transform(torch.abs(denorm_array.permute(0,2,1)))


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
                    # # audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
                    # # audio_sum =audio_sum * 0.9

                    # # ---------- Normalise the audio END ----------

                    # Compute loss
                    # spec_loss = spec_dist(audio_sum[:, l_grain//2:-l_grain//2], waveform)
                    spec_loss = spec_dist(audio_sum, waveform)
                    # spec_loss = calc_reconstruction_loss(torch.abs(grain_fft), x_hat_old)
                    if beta > 0:
                        kld_loss = compute_kld(mu, log_variance) * beta
                    else:
                        kld_loss = 0.0
                    # Notes this won't work when using grains, need to look into this
                    if ENV_DIST > 0:
                        env_loss =  envelope_distance(audio_sum, waveform, n_fft=l_grain,log=True) * ENV_DIST
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


            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "spec_loss": train_spec_loss, "env_loss": env_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "spec_val_loss": spec_val_loss, "env_val_loss": env_val_loss, "val_loss": val_loss, "beta": beta})

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
                    

            if SAVE_RECONSTRUCTIONS:
                if (epoch+1) % CHECKPOINT_REGULAIRTY == 0:

                    # Get data using test dataset
                    with torch.no_grad():

                        # Lets get batch of test images
                        dataiter = iter(test_dataloader)
                        waveform, labels = next(dataiter)
                        waveform = waveform.to(DEVICE)
                        print("initial shape", waveform.shape)

                        ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32)
                        stft_audio = torch.stft(waveform, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
                        grain_fft = stft_audio.permute(0,2,1)

                        # # ---------- Turn Waveform into grains ----------
                        ola_window = signal.hann(l_grain,sym=False)
                        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
                        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
                        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
                        ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

                        # slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
                        # mb_grains = F.conv1d(waveform.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
                        # mb_grains = mb_grains.permute(0,2,1)
                        # bs = mb_grains.shape[0]
                        # # repeat the overlap add windows acrosss the batch and apply to the grains
                        # mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
                        # grain_fft = torch.fft.rfft(mb_grains)
                        # # ---------- Turn Waveform into grains END ----------

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
                        grain_fft_abs = torch.abs(grain_fft)
                        grain_min = grain_fft_abs.min()
                        grain_max = grain_fft_abs.max()
                        norm_array = (grain_fft_abs - grain_fft_abs.min()) / (grain_fft_abs.max() - grain_fft_abs.min())
                        norm_array = norm_array * (1 - 0) + 0

                        # ---------- Run Model ----------

                        # x_hat, z, mu, log_variance = model(mb_grains)   
                        x_hat, z, mu, log_variance = model(torch.abs(norm_array))   
                        # x_hat, z, mu, log_variance = model(torch.abs(grain_fft))   
                        # x_hat, z, mu, log_variance = model(inv_mfccs)   

                        # ---------- Run Model END ----------

                        # denorm array
                        denorm_array = (x_hat - 0) / (1 - 0)
                        denorm_array = denorm_array * (grain_max - grain_min) + grain_min 

                        transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
                        audio_sum = transform(torch.abs(denorm_array.permute(0,2,1)))


                        # print("recon shape", y_inv.shape)
                        # torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/fft_audio/griffinLim_recon.wav', y_inv.cpu(), SAMPLE_RATE)

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

                        # # audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
                        # # audio_sum =audio_sum * 0.9

                        # Get the spectral loss
                        spec_loss = spec_dist(audio_sum, waveform)
                        # spec_loss = spec_dist(audio_sum[:, l_grain//2:-l_grain//2], waveform)
                        # spec_loss = calc_reconstruction_loss(torch.abs(grain_fft), x_hat_old)

                        for i, recon_signal in enumerate(audio_sum):
                            # spec_loss = spec_dist(x_hat[i], waveforms[i])
                            # Check the energy differences
                            print("Saving ", i)
                            torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/CC_recon_{i}_{spec_loss}_{epoch+1}.wav', recon_signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                            # Saving for FAD scoring
                            # TODO, do i need to resample this to 16kHz? 
                            torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/fake_audio/CC_recon.wav', recon_signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                            torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/real_audio/CC_{i}.wav", waveform[i].unsqueeze(0).cpu(), SAMPLE_RATE)

                        fad_score = frechet.score(f'{RECONSTRUCTION_SAVE_DIR}/real_audio', f'{RECONSTRUCTION_SAVE_DIR}/fake_audio', dtype="float32")

                        print('Test Spec Loss: {}'.format(spec_loss),
                            '\tTest FAD Score: {}'.format(fad_score))

                        if WANDB:
                            wandb.log({"test_spec_loss": spec_loss, "test_fad_score": fad_score})

    elif EXPORT_LATENTS:

        print("-------- Exporting Latents --------")

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(DEVICE)
        model.eval()

        print(len(test_dataloader))
        train_latents,train_labels,val_latents,val_labels = export_latents(model,test_dataloader,test_dataloader, l_grain, n_grains, hop_size, TEST_SIZE, DEVICE)
        # train_latents,train_labels,val_latents,val_labels = export_latents(model,train_dataloader,val_dataloader, l_grain, n_grains, hop_size, BATCH_SIZE, DEVICE)
        
        print("-------- Done Exporting Latents --------")


    else:

        print("-------- Inference Mode --------")

        ###########
        # Inference
        ########### 


        # with torch.no_grad():
        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

        # Put model in eval mode
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():

            # Lets get batch of test images
            dataiter = iter(test_dataloader)
            waveforms, labels = next(dataiter)

            waveforms = waveforms.to(DEVICE)

            ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32)
            stft_audio = torch.stft(waveforms, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
            grain_fft = stft_audio.permute(0,2,1)

            # ---------- Turn Waveform into grains ----------
            ola_window = signal.hann(l_grain,sym=False)
            ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
            ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
            ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
            ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(DEVICE)

            # slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False).to(DEVICE)
            # mb_grains = F.conv1d(waveforms.unsqueeze(1),slice_kernel,stride=hop_size,groups=1,bias=None)
            # mb_grains = mb_grains.permute(0,2,1)
            # bs = mb_grains.shape[0]
            # # repeat the overlap add windows acrosss the batch and apply to the grains
            # mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
            # grain_fft = torch.fft.rfft(mb_grains)
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

            grain_fft_abs = torch.abs(grain_fft)
            grain_min = grain_fft_abs.min()
            grain_max = grain_fft_abs.max()
            norm_array = (grain_fft_abs - grain_fft_abs.min()) / (grain_fft_abs.max() - grain_fft_abs.min())
            norm_array = norm_array * (1 - 0) + 0
            
            denorm_array = (norm_array - 0) / (1 - 0)
            denorm_array = denorm_array * (grain_max - grain_min) + grain_min
            transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
            audio_sum = transform(denorm_array.permute(0,2,1))
            torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/griffinLim.wav", audio_sum[0].unsqueeze(0).cpu(), SAMPLE_RATE)
            mod_grain_fft = mod_sigmoid(torch.abs(grain_fft))
            print("Max value: ", mod_grain_fft.max())

            plt.figure()
            librosa.display.specshow(denorm_array[0].permute(1,0).cpu().numpy(), n_fft=l_grain, hop_length=hop_size, sr=SAMPLE_RATE, x_axis='time', y_axis='log')
            plt.colorbar()
            plt.savefig("test.png")


            # ---------- Run Model ----------

            x_hat, z, mu, log_variance = model(torch.abs(grain_fft))

            # ---------- Run Model END ----------

            # plt.figure()
            # librosa.display.specshow(x_hat[0].permute(1,0).cpu().numpy(), n_fft=l_grain, hop_length=hop_size, sr=SAMPLE_RATE, x_axis='time', y_axis='log')
            # plt.colorbar()
            # plt.savefig("test.png")

            transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
            audio_sum = transform(torch.abs(x_hat.permute(0,2,1)))

            # ---------- Noise Filtering ----------

            # Reshape for noise filtering - TODO Look if this is necesary
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

            # # audio_sum = audio_sum / torch.max(torch.abs(audio_sum))
            # # audio_sum = audio_sum * 0.9

            # # NOTE Removed the post processing step for now
            # # This module applies a multi-channel temporal convolution that
            # # learns a parallel set of time-invariant FIR filters and improves
            # # the audio quality of the assembled signal.
            # # TODO Add back in ?
            # # audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)

            spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
            spec_loss = spec_dist(audio_sum, waveforms)

            print(x_hat.shape)
            print(waveforms.shape)

            print("Spectral Loss: ", spec_loss)

            # print_spectral_shape(waveforms[0,:], spec[0,:,:].cpu().numpy(), hop_size, l_grain)

            # filter_spectral_shape(waveforms[0,:], hop_size, l_grain, n_grains, tar_l)

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
                for i, signal in enumerate(audio_sum):
                    # torchaudio.save(f"./audio_tests/usd_vae_{classes[labels[i]]}_{i}.wav", signal, SAMPLE_RATE)
                    spec_loss = spec_dist(audio_sum[i], waveforms[i])
                    # Check the energy differences
                    # print("Saving ", labels[i][:-4])
                    print("Saving ", i)
                    print("Loss: ", spec_loss)
                    # torchaudio.save(f"./audio_tests/reconstructions/2048/recon_{labels[i][:-4]}_{spec_loss}.wav", signal, SAMPLE_RATE)
                    # torchaudio.save(f"./audio_tests/reconstructions/2048/{labels[i][:-4]}.wav", waveforms[i], SAMPLE_RATE)
                    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/fake_audio/CC_recon_{i}_{spec_loss}.wav', signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                    torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/real_audio/CC_{i}.wav", waveforms[i].unsqueeze(0).cpu(), SAMPLE_RATE)
                    # print(f'{classes[labels[i]]} saved')

                    fad_score = frechet.score(f'{RECONSTRUCTION_SAVE_DIR}/real_audio', f'{RECONSTRUCTION_SAVE_DIR}/fake_audio', dtype="float32")
                    print("FAD Score: ", fad_score)


