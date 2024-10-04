import sys
sys.path.append('../')

from models.filterbank.filterbank_vae import SpectralVAE_v1, get_noise_bands
from models.dataloaders.waveform_dataloaders import make_audio_dataloaders_noPadding, AudioDataset
from models.loss_functions import calc_combined_loss, compute_kld, spectral_distances, envelope_distance, calc_reconstruction_loss
from scripts.configs.hyper_parameters_spectral import *
from utils.utilities import plot_latents, export_latents, init_beta, print_spectral_shape, filter_spectral_shape
from utils.dsp_components import safe_log10, noise_filtering, mod_sigmoid
import torch_dct as dct
import librosa
from frechet_audio_distance import FrechetAudioDistance
import matplotlib.pyplot as plt
import librosa.display
from models.filterbank.filterbank import FilterBank



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
        project="filterbank_model",
        # name= f"run_{datetime.now()}",
        name= f"removed_reshape_addedtoKLLossCalc",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "v1",
        "dataset": "Full_Seawaves_UrbanSound8k",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "latent size": LATENT_SIZE,
        "target_beta": TARGET_BETA,
        "beta_steps": BETA_STEPS,
        "beta_warmup_start": BETA_WARMUP_START_PERC,
        "hidden_dim": H_DIM,
        "env_dist": ENV_DIST,
        "grain_length": GRAIN_LENGTH,
        "hop_size_ratio": HOP_SIZE_RATIO,
        "num_ccs": NUM_CC,
        "num_mels": NUM_MELS,
        "sample_rate": SAMPLE_RATE,
        "compression_factor": COMPRESSION_FACTOR
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

# 2**9 seems to be the limit
# AUDIO_SAMPLE_SIZE = 2**16
AUDIO_SAMPLE_SIZE = 2**16
print("Audio Sample Size: ", AUDIO_SAMPLE_SIZE)

if __name__ == "__main__":

    audio_dataset = AudioDataset(dataset_path=AUDIO_DIR, audio_size_samples=AUDIO_SAMPLE_SIZE, min_batch_size=BATCH_SIZE, sampling_rate=SAMPLE_RATE, device=DEVICE)
    n_samples = len(audio_dataset)
    train_split=0.8
    n_train = int(n_samples*train_split)
    train_dataset,test_dataset = torch.utils.data.random_split(audio_dataset, [n_train, n_samples-n_train])
    # dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    # TODO Make a test dataloader

    hop_size = int(GRAIN_LENGTH * HOP_SIZE_RATIO)
    l_grain = GRAIN_LENGTH

    print("-----Dataset Loaded-----")

    model = SpectralVAE_v1(l_grain=l_grain, h_dim=H_DIM, z_dim=LATENT_SIZE, synth_window=hop_size, n_band=2048)
    
    model.to(DEVICE)

    # torch.autograd.set_detect_anomaly(True)

    if TRAIN:

        print("-------- Training Mode --------")

        ###########
        # Training
        ########### 

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=4000)
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

            # for data in train_dataloader:
            for waveform in train_dataloader:

                if PROFILE:
                    start1 = time.time()

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

                # waveform, label = data 
                # print(waveform.shape)
                # print(img)

                waveform = Variable(waveform).to(DEVICE)                       # we are just intrested in just images
                # no need to flatten images
                optimizer.zero_grad()                   # clear the gradients
                # x_hat, z = model(waveform)                 # forward pass: compute predicted outputs 

                # ---------- Turn Waveform into grains ----------
                ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32).to(DEVICE)
                stft_audio = torch.stft(waveform, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
                

                # ---------- Turn Waveform into grains END ----------


                # # ---------- Get CCs, or MFCCs and invert ----------
                # CCs
                # print(torch.abs(stft_audio.sum()))
                grain_db = 20*safe_log10(torch.abs(stft_audio))
                # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
                cepstral_coeff = dct.dct(grain_db.permute(0,2,1))
                cepstral_coeff[:,:,NUM_CC:] = 0
                inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)
                inv_cep_coeffs_log_mag = (dct.idct(cepstral_coeff))
                
                # # ---------- Get CCs, or MFCCs and invert END ----------

                if PROFILE:
                    end1 = time.time()
                    start2 = time.time()
                
                # ---------- Run Model ----------


                x_hat, z, mu, log_variance = model(waveform)

                mu = mu.reshape(mu.shape[0]*mu.shape[1],mu.shape[2])
                log_variance = mu.reshape(log_variance.shape[0]*log_variance.shape[1], log_variance.shape[2])

                if PROFILE:
                    end2 = time.time()
                    start3 = time.time()

                # ---------- Run Model END ----------

                if PROFILE:
                    end3 = time.time()
                    start4 = time.time()

                spec_loss = spec_dist(x_hat, waveform)

                # Compute loss
                # spec_loss = spec_dist(audio_sum, waveform)
                if beta > 0:
                    kld_loss = compute_kld(mu, log_variance) * beta
                    # kld_loss=0
                else:
                    kld_loss = 0.0
                # Notes this won't work when using grains, need to look into this
                if ENV_DIST > 0:
                    env_loss =  envelope_distance(x_hat, waveform, n_fft=1024,log=True) * ENV_DIST
                else:
                    env_loss = 0.0

                loss = kld_loss + spec_loss + env_loss
                # loss = spec_loss
                if PROFILE:
                    end4 = time.time()
                    start5 = time.time()

                # Compute gradients and update weights
                loss.backward()                         # backward pass
                # Try clipping gradients to help with vanishing gradients
                # nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()                        # perform optimization step
                if PROFILE:
                    end5 = time.time()

                # Accumulate loss for reporting
                running_train_loss += loss
                running_kl_loss += kld_loss
                running_spec_loss += spec_loss
                running_env_loss += env_loss

                accum_iter+=1
                # print("Iteration: ",accum_iter)

                # for name, param in model.named_parameters():
                #     print(name, param.grad.norm())

            # Decay the learning rate
            # lr_scheduler.step()
            # new_lr = optimizer.param_groups[0]["lr"]
                
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
            running_multi_spec_loss = 0.0


            if PROFILE:
                print("PROFILE")
                print("\t Data Pre-processing: ", end1-start1)
                print("\t Model Run: ", end2-start2)
                print("\t Data Post-procesing: ", end3-start3)
                print("\t Loss: ", end4-start4)
                print("\t Optimisation: ", end5-start5)
                print("PROFILE END")

            with torch.no_grad():
                for waveform in val_dataloader:
                    # waveform, label = data 
                    waveform = waveform.to(DEVICE)
                    
                    ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32).to(DEVICE)
                    stft_audio = torch.stft(waveform, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")

                    # ---------- Turn Waveform into grains END ----------

                    # # ---------- Get CCs, or MFCCs and invert ----------
                    # CCs
                    grain_db = 20*safe_log10(torch.abs(stft_audio))
                    # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
                    cepstral_coeff = dct.dct(grain_db.permute(0,2,1))
                    cepstral_coeff[:,:,NUM_CC:] = 0
                    inv_cep_coeffs_log_mag = (dct.idct(cepstral_coeff))
                    inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)

                    # # ---------- Get CCs, or MFCCs and invert END ----------

                    # ---------- Run Model ----------

                    x_hat, z, mu, log_variance = model(waveform)
                    # x_hat, z, mu, log_variance = model(inv_cep_coeffs[:,:-1,:])
                    mu = mu.reshape(mu.shape[0]*mu.shape[1],mu.shape[2])
                    log_variance = mu.reshape(log_variance.shape[0]*log_variance.shape[1], log_variance.shape[2])

                    # ---------- Run Model END ----------

                    spec_loss = spec_dist(x_hat, waveform)

                    # print(img)

                    if beta > 0:
                        kld_loss = compute_kld(mu, log_variance) * beta
                        # kld_loss = 0
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


            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "spec_loss": train_spec_loss, "env_loss": env_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "spec_val_loss": spec_val_loss, "env_val_loss": env_val_loss, "val_loss": val_loss, "beta": beta})

            # for i in model.named_parameters():
            #     print(i[0][-6:])
            # Checking the gradients
            # for p,n in zip(model.parameters(),model.named_parameters()):
            #     if n[0][-6:] == 'weight':
            #         print('===========\ngradient:{}\n----------\n{}'.format(n[0],p.grad.sum()))

            print('Epoch: {}'.format(epoch+1),
            '\tStep: {}'.format(accum_iter+1),
            '\t KL Loss: {:.5f}'.format(kl_loss),
            '\tTraining Loss: {:.4f}'.format(train_loss),
            '\tValidation Loss: {:.4f}'.format(val_loss),
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
                        }, f"{SAVE_DIR}/waveform_vae_latest_latentTest.pt")
                    

            if SAVE_RECONSTRUCTIONS:
                if (epoch+1) % RECON_REGULAIRTY == 0:

                    # Get data using test dataset
                    with torch.no_grad():

                        # Lets get batch of test images
                        dataiter = iter(val_dataloader)
                        # dataiter = iter(test_dataloader)
                        waveform = next(dataiter)
                        waveform = waveform.to(DEVICE)

                        ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32).to(DEVICE)
                        stft_audio = torch.stft(waveform, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")

                        # ---------- Turn Waveform into grains END ----------

                        # ---------- Get CCs, or MFCCs and invert ----------
                        # CCs
                        grain_db = 20*safe_log10(torch.abs(stft_audio))
                        # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
                        cepstral_coeff = dct.dct(grain_db.permute(0,2,1))
                        cepstral_coeff[:,:,NUM_CC:] = 0
                        inv_cep_coeffs_log_mag = (dct.idct(cepstral_coeff))
                        inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)


                        # # ---------- Get CCs, or MFCCs and invert END ----------

                        # ---------- Run Model ----------

                        x_hat, z, mu, log_variance = model(waveform)
                        # x_hat, z, mu, log_variance = model(inv_cep_coeffs[:,:-1,:])
                        mu = mu.reshape(mu.shape[0]*mu.shape[1],mu.shape[2])
                        log_variance = mu.reshape(log_variance.shape[0]*log_variance.shape[1], log_variance.shape[2])

                        # ---------- Run Model END ----------

                        spec_loss = spec_dist(x_hat, waveform)

                        for i, recon_signal in enumerate(x_hat):
                            # spec_loss = spec_dist(x_hat[i], waveforms[i])
                            # Check the energy differences
                            print("Saving ", i)
                            torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/CC_recon_{i}_{spec_loss}_{epoch+1}_0.1.wav', recon_signal.unsqueeze(0).cpu(), SAMPLE_RATE)
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


    elif SEQ_TEST:

        ###########
        # Sequential test
        # - Check the models ability to generate a sequence.
        ########### 
        seed = 0
        torch.manual_seed(seed)


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
            # ---------- Turn Waveform into grains ----------
            ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32).to(DEVICE)
            stft_audio = torch.stft(waveforms, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")

            # ---------- Turn Waveform into grains END ----------

            # ----------- RI Spec ---------
            # The compression factor essentially increases the granularity of the 
            compressionFactor = COMPRESSION_FACTOR
            real_spec = stft_audio.real
            imag_spec = stft_audio.imag
            compress_real_spec = 2.0 * torch.sigmoid(compressionFactor*real_spec) - 1.0
            compress_imag_spec = 2.0 * torch.sigmoid(compressionFactor*imag_spec) - 1.0

            ri_spec = torch.cat((compress_real_spec, compress_imag_spec), dim=1)

            ri_spec = ri_spec.permute(0, 2, 1)

            # ---------- RI Spec END ----------

            # # ---------- Get CCs, or MFCCs and invert ----------
            # CCs
            grain_db = 20*safe_log10(torch.abs(stft_audio))
            # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
            cepstral_coeff = dct.dct(grain_db.permute(0,2,1))
            cepstral_coeff[:,:,NUM_CC:] = 0
            inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)

            # # MFCCs  - use librosa function as they are more reliable
            # # grain_fft = grain_fft.permute(0,2,1)
            # # grain_mel= safe_log10(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft.cpu().numpy())**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
            # # mfccs = dct.dct(grain_mel)
            # # inv_mfccs = dct.idct(mfccs).cpu().numpy()       
            # # inv_mfccs = torch.from_numpy(librosa.feature.inverse.mel_to_stft(M=10**inv_mfccs, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH)).to(DEVICE)
            # # inv_mfccs = inv_mfccs.permute(0,2,1)

            # # ---------- Get CCs, or MFCCs and invert END ----------

            # --------- Extract the prev ri spec -----------
            first_grain = ri_spec[:, 0, :]
            #current_ri_spec = ri_spec[:, 1:, :]
            current_ri_spec = ri_spec[:, 1:, :]
            prev_ri_spec = ri_spec[:,:-1,:]
            inv_cep_coeffs = inv_cep_coeffs[:, 1:, :]
            print(current_ri_spec.shape)

            # Reshape to [bs*n_grains, l_grain] 
            # inv_cep_coeffs = inv_cep_coeffs.reshape(inv_cep_coeffs.shape[0]*(n_grains-1), (int((l_grain//2)+1)))
            prev_ri_spec = prev_ri_spec.reshape(prev_ri_spec.shape[0]*(n_grains-1), (int((l_grain//2)+1))*2)
            current_ri_spec = current_ri_spec.reshape(current_ri_spec.shape[0]*(n_grains-1), (int((l_grain//2)+1))*2)


            # ---------- Run Model ----------
            # std = 0.00001
            # mean = 0
            # prev_ri_spec_noise = prev_ri_spec + torch.randn(prev_ri_spec.size()) * std + mean

            x_hat, z, mu, log_variance = model(inv_cep_coeffs[:,:-1,:])

            recon_audio = x_hat.reshape(x_hat.shape[0], x_hat.shape[2])
            # x_hat, z, mu, log_variance = model(inv_mfccs)   
            # mse_test = calc_reconstruction_loss(current_ri_spec, x_hat)


            # print("Biggest Flux: ", torch.max(torch.abs(x_hat[2:245] - current_ri_spec[2:245])))


            prev_recon_ri_spec = first_grain
            # prev_recon_ri_spec = prev_ri_spec[20,:].unsqueeze(0)
            recon_ri_spec = first_grain
            recon_ri_spec_tmp = first_grain
            for i in range(z.shape[0]):
            # for i in range(1,10):
            # for i in range(20,21):
                prev_recon_ri_spec = model.decode(z[i,:].unsqueeze(0), prev_recon_ri_spec)["audio"]
                recon_ri_spec = torch.cat((recon_ri_spec, prev_recon_ri_spec), dim=0)
                mse = calc_reconstruction_loss(current_ri_spec[i,:].unsqueeze(0), prev_recon_ri_spec)

                # std = 0.01
                # std = 0.001
                # mean = 0
                # prev_ri_spec_noise = prev_ri_spec + torch.randn(prev_ri_spec.size()) * std + mean

                tmp = model.decode(z[i,:].unsqueeze(0), prev_ri_spec[i,:].unsqueeze(0))["audio"]
                # print("Predicted")
                # print(tmp.sum())
                # print(((torch.logit((1.0+tmp) * 0.5, eps=1e-7)) / COMPRESSION_FACTOR).sum())
                # plt.plot(((torch.logit((1.0+tmp[0]) * 0.5, eps=1e-7)) / COMPRESSION_FACTOR))
                # plt.plot(tmp[0])
                # plt.savefig("test.png")
                recon_ri_spec_tmp = torch.cat((recon_ri_spec_tmp, tmp), dim=0)
                # print("Actual")
                # print(current_ri_spec[i,:].sum())
                # print(real_spec[0,i+1,:].sum()+imag_spec[0,i+1,:].sum())
                # plt.plot(torch.cat((real_spec[0,i+1,:], imag_spec[0,i+1,:])))
                # print("Biggest Flux: ", torch.max(torch.abs(tmp - current_ri_spec[i,:])))
                # plt.plot(current_ri_spec[i,:])
                # plt.savefig("test.png")
                mse_tmp = calc_reconstruction_loss(current_ri_spec[i,:].unsqueeze(0), tmp)
                mse_full = calc_reconstruction_loss(current_ri_spec[i,:].unsqueeze(0), x_hat[i,:].unsqueeze(0))
                # print("MSE: ", mse_tmp.item(), mse.item())
                print('MSE {:.7f}'.format(mse),
                '\t, {:.7f}'.format(mse_tmp),
                '\t, {:.7f}'.format(mse_full))
            # print(img)

            # Reshape back to [bs, n_grains, l_grain]
            x_hat = recon_ri_spec.reshape(-1, n_grains, (int((l_grain//2)+1))*2)
            # x_hat = recon_ri_spec_tmp.reshape(-1, n_grains, (int((l_grain//2)+1))*2)

            # ---------- Run Model END ----------

            # decompress x_hat
            x_hat = (torch.logit((1.0+x_hat) * 0.5, eps=1e-7)) / COMPRESSION_FACTOR

            recon_real = x_hat[:, :, :l_grain//2+1]
            recon_imag = x_hat[:, :, l_grain//2+1:]
            recon_complex = torch.complex(recon_real, recon_imag)
            recon_audio = torch.istft(recon_complex.permute(0, 2, 1), n_fft=l_grain, hop_length=hop_size, window=ola_window)

            spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
            spec_loss = spec_dist(recon_audio, waveforms)

            print(x_hat.shape)
            print(waveforms.shape)

            print("Spectral Loss: ", spec_loss)

            if SAVE_RECONSTRUCTIONS:
                for i, signal in enumerate(recon_audio):
                    # torchaudio.save(f"./audio_tests/usd_vae_{classes[labels[i]]}_{i}.wav", signal, SAMPLE_RATE)
                    spec_loss = spec_dist(recon_audio[i], waveforms[i])
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


    else:

        print("-------- Inference Mode --------")

        ###########
        # Inference
        ########### 

        seed = 0
        torch.manual_seed(seed)


        # with torch.no_grad():
        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

        # Put model in eval mode
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():

            # Lets get batch of test images
            # dataiter = iter(test_dataloader)
            dataiter = iter(val_dataloader)
            waveforms = next(dataiter)

            waveforms = waveforms.to(DEVICE)
            # ---------- Turn Waveform into grains ----------
            ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32).to(DEVICE)
            stft_audio = torch.stft(waveforms, n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
            # ---------- Turn Waveform into grains END ----------

            # ----------- RI Spec ---------
            # The compression factor essentially increases the granularity of the 
            compressionFactor = COMPRESSION_FACTOR
            real_spec = stft_audio.real
            imag_spec = stft_audio.imag
            compress_real_spec = 2.0 * torch.sigmoid(compressionFactor*real_spec) - 1.0
            compress_imag_spec = 2.0 * torch.sigmoid(compressionFactor*imag_spec) - 1.0

            ri_spec = torch.cat((compress_real_spec, compress_imag_spec), dim=1)

            ri_spec = ri_spec.permute(0, 2, 1)

            # print("RI spec shape: ", ri_spec.shape)

            # plt.figure()
            # librosa.display.specshow(real_spec[0].cpu().numpy(), n_fft=l_grain, hop_length=hop_size, sr=SAMPLE_RATE, x_axis='time', y_axis='log')
            # plt.colorbar()
            # plt.savefig("test.png")

            # ---------- RI Spec END ----------

            # # ---------- Get CCs, or MFCCs and invert ----------
            # CCs
            grain_db = 20*safe_log10(torch.abs(stft_audio))
            # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
            cepstral_coeff = dct.dct(grain_db.permute(0,2,1))
            cepstral_coeff[:,:,NUM_CC:] = 0
            inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)
            inv_cep_coeffs_test = inv_cep_coeffs


            # ---------- Run Model ----------

            print(waveforms.shape)
            x_hat, z, mu, log_variance = model(waveforms)
            recon_audio = x_hat
            print(recon_audio.shape)
            print(img)

            # ---------- Run Model END ----------

            spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
            spec_loss = spec_dist(recon_audio, waveforms)

            print(x_hat.shape)
            print(waveforms.shape)

            print("Spectral Loss: ", spec_loss)

            if SAVE_RECONSTRUCTIONS:
                for i, signal in enumerate(recon_audio):
                    # torchaudio.save(f"./audio_tests/usd_vae_{classes[labels[i]]}_{i}.wav", signal, SAMPLE_RATE)
                    spec_loss = spec_dist(recon_audio[i], waveforms[i])
                    # Check the energy differences
                    # print("Saving ", labels[i][:-4])
                    print("Saving ", i)
                    print("Loss: ", spec_loss)
                    # torchaudio.save(f"./audio_tests/reconstructions/2048/recon_{labels[i][:-4]}_{spec_loss}.wav", signal, SAMPLE_RATE)
                    # torchaudio.save(f"./audio_tests/reconstructions/2048/{labels[i][:-4]}.wav", waveforms[i], SAMPLE_RATE)
                    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/CC_recon_{i}_{spec_loss}.wav', signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/fake_audio/CC_recon_{i}.wav', signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                    torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/real_audio/CC_{i}.wav", waveforms[i].unsqueeze(0).cpu(), SAMPLE_RATE)
                    # print(f'{classes[labels[i]]} saved')

                    fad_score = frechet.score(f'{RECONSTRUCTION_SAVE_DIR}/real_audio', f'{RECONSTRUCTION_SAVE_DIR}/fake_audio', dtype="float32")
                    print("FAD Score: ", fad_score)
            
            # Export random samples



