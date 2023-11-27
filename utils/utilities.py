import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
import soundfile as sf
from sklearn.decomposition import PCA
import numpy as np
from torch.nn import functional as F
import torchaudio
import librosa
from scipy import fft, signal
import torch_dct as dct
# from dsp_components import amp_to_impulse_response_w_phase
import utils.dsp_components as dsp
from scripts.configs.hyper_parameters_waveform import NORMALIZE_OLA, RECONSTRUCTION_SAVE_DIR, SAMPLE_RATE


# Sample from a gaussian distribution
def sample_from_distribution(mu, log_variance):

    # point = mu + sigma*sample(N(0,1))
    
    std = torch.exp(log_variance * 0.5)
    # epsilon = torch.normal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
    epsilon = torch.randn_like(std)
    sampled_point = mu + std * epsilon

    return sampled_point

# Show the latent space
def show_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10,10))
    plt.scatter(latent_representations[:, 0],
        latent_representations[:, 1],
        cmap="rainbow",
        c = sample_labels,
        alpha = 0.5,
        s = 2)
    plt.colorbar
    plt.savefig("laetnt_rep.png") 

def show_image_comparisons(images, x_hat):

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
            
    # input images on top row, reconstructions on bottom
    for images, row in zip([images, x_hat], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("comparisons.png")

def plot_latents(train_latents,train_labels, classes,export_dir):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    n_grains = train_latents.shape[1]
    z_dim = train_latents.shape[2]
    train_latents = train_latents.view(-1,z_dim).numpy()
    train_labels = train_labels.unsqueeze(-1).repeat(1,n_grains).view(-1).numpy().astype(str)
    for i,c in enumerate(classes):
        train_labels[np.where(train_labels==str(i))] = c
    pca = PCA(n_components=2)
    pca.fit(train_latents)
    train_latents = pca.transform(train_latents)
    print(f'PCA Shape: {train_latents.shape}')
    # TODO: shuffle samples for better plotting
    sns.scatterplot(x=train_latents[:,0], y=train_latents[:,1], hue=train_labels, s=1)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(export_dir,"latent_scatter_trainset.pdf"))
    plt.close("all")

# Compute the latens
def compute_latents(w_model,dataloader,device):
    tmploader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=5, shuffle=False, drop_last=False)
    dataset_latents = []
    dataset_labels = []
    for i,batch in enumerate(tmploader):
        with torch.no_grad():
            audio,labels = batch
            bs = audio.shape[0]
            mu = w_model.encode(audio.to(device))["mu"].cpu()
            # mu of shape [bs*n_grains,z_dim]
            mu = mu.reshape(bs,w_model.n_grains,w_model.z_dim)
            dataset_latents.append(mu)
            dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    dataset_labels = torch.cat(dataset_labels,0)
    # labels not so important now, but will be in future
    # print("--- Exported dataset sizes:\t",dataset_latents.shape,dataset_labels.shape)
    print("--- Exported dataset sizes:\t",dataset_latents.shape)
    return dataset_latents,dataset_labels

# Export the latents
def export_latents(w_model,train_dataloader,test_dataloader, device):
    train_latents,train_labels = compute_latents(w_model,train_dataloader, device)
    test_latents,test_labels = compute_latents(w_model,test_dataloader, device)
    return train_latents,train_labels,test_latents,test_labels

# Safe log for cases where x is very close to zero
def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

def init_beta(max_steps,tar_beta,beta_steps=1000, warmup_perc=0.1):
    # if continue_training:
    #     beta = tar_beta
    #     print("\n*** setting fixed beta of ",beta)
    # else:
    # warmup wihtout increasing beta
    warmup_start = int(warmup_perc*max_steps)
    # set beta steps to only increase of half of max steps
    beta_step_size = int(max_steps/2/beta_steps)
    beta_step_val = tar_beta/beta_steps
    beta = 0
    print("--- Initialising Beta, from 0 to ", tar_beta)
    print("")
    print('--- Beta: {}'.format(beta),
            '\tWarmup Start: {}'.format(warmup_start),
            '\tStep Size: {}'.format(beta_step_size),
            '\tStep Val: {:.5f}'.format(beta_step_val))
        
    return beta, beta_step_val, beta_step_size, warmup_start

def export_embedding_to_audio_reconstructions(l_model,w_model,batch, export_dir, sr, device,trainset=False):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)

    with torch.no_grad():
        z,conds = batch
        z,conds = z.to(device),conds.to(device)
        # forward through latent embedding
        z_hat, e, mu, log_variance = l_model(z,conds, sampling=False)
        # reshape as minibatch of individual grains of shape [bs*n_grains,z_dim]
        z,z_hat = z.reshape(-1,w_model.z_dim),z_hat.reshape(-1,w_model.z_dim)
        # export reconstruction by pretrained waveform model and by embedding + waveform models
        audio,audio_hat = w_model.decode(z),w_model.decode(z_hat)
        audio = audio.cpu().numpy()
        audio_hat = audio_hat.cpu().numpy()
        #audio_export = torch.cat((audio,audio_hat),-1).cpu().numpy()
        for i in range(audio_hat.shape[0]):
            if trainset:
                sf.write(os.path.join(export_dir,"embedding_to_audio_train_reconstruction_orig_"+str(i)+".wav"),audio[i,:], sr)
                sf.write(os.path.join(export_dir,"embedding_to_audio_train_reconstruction_hat_"+str(i)+".wav"),audio_hat[i,:], sr)
            else:
                sf.write(os.path.join(export_dir,"embedding_to_audio_test_reconstruction_orig_"+str(i)+".wav"),audio[i,:], sr)
                sf.write(os.path.join(export_dir,"embedding_to_audio_test_reconstruction_hat_"+str(i)+".wav"),audio_hat[i,:], sr)

def export_random_samples(l_model,w_model,export_dir, z_dim, e_dim, sr, classes, device, n_samples=10,temperature=1.):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    with torch.no_grad():
        for i,cl in enumerate(classes):
            rand_e = torch.randn((n_samples, e_dim)).to(device)
            rand_e = rand_e*temperature
            conds = torch.zeros(n_samples).to(device).long()+i
            z_hat = l_model.decode(rand_e,conds).reshape(-1, z_dim)
            audio_hat = w_model.decode(z_hat).view(-1).cpu().numpy()
            sf.write(os.path.join(export_dir,"random_samples_"+cl+".wav"),audio_hat, sr)

def generate_noise_grains(batch_size, n_grains, l_grain, dtype, device, hop_ratio=0.25):

    tar_l = int(((n_grains+3)/4)*l_grain)

    noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*2-1

    hop_size = int(hop_ratio*l_grain)

    new_noise = noise[:, 0:l_grain].unsqueeze(1)
    for i in range(1, n_grains):  
        starting_point = i*hop_size
        ending_point = starting_point+l_grain
        tmp_noise = noise[:, starting_point:ending_point].unsqueeze(1)
        new_noise = torch.cat((new_noise, tmp_noise), dim = 1)

    return new_noise

def generate_noise_grains_stft(batch_size, tar_l, dtype, device, hop_size):

    noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*2-1

    noise_stft = librosa.stft(noise.cpu().numpy(), hop_length=hop_size)
    noise_stft = torch.from_numpy(noise_stft)

    return noise_stft

def print_spectral_shape(waveform, learnt_spec_shape, hop_size, l_grain):

    print("-----Saving Spectral Shape-----")

    slice_kernel = torch.eye(l_grain).unsqueeze(1)
    mb_grains = F.conv1d(waveform.unsqueeze(0).unsqueeze(0).cpu(), slice_kernel,stride=hop_size,groups=1,bias=None)
    mb_grains = mb_grains.permute(0,2,1).squeeze()

    grain_fft = fft.rfft(mb_grains.cpu().numpy())

    grain_db = 20*np.log10(np.abs(grain_fft))

    plt.plot(grain_db[0])

    # Note transposing for librosa
    cepstral_coeff = fft.dct(grain_db)

    cepstral_coeff[:, 128:] = 0

    inv_cepstral_coeff = fft.idct(cepstral_coeff)
    plt.plot(inv_cepstral_coeff[0])

    plt.plot(learnt_spec_shape[0])

    plt.savefig("spectral_shape.png")

def filter_spectral_shape(waveform, hop_size, l_grain, n_grains, tar_l):

    print("-----Noise filtering Spectral Shape-----")
    # Set BS equal to 1
    bs = 1

    slice_kernel = torch.eye(l_grain).unsqueeze(1)
    mb_grains = F.conv1d(waveform.unsqueeze(0).unsqueeze(0).cpu(), slice_kernel,stride=hop_size,groups=1,bias=None)
    mb_grains = mb_grains.permute(0,2,1)

    ola_window = signal.hann(l_grain,sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
    ola_windows = torch.nn.Parameter(ola_windows,requires_grad=False)
    mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))


    grain_fft = fft.rfft(mb_grains.cpu().numpy())
    grain_fft_torch = torch.fft.rfft(mb_grains)
    # grain_db = 20*np.log10(np.abs(grain_fft))
    grain_db = 20*np.log10(np.abs(grain_fft))
    grain_db_torch = 20*torch.log10(torch.abs(grain_fft_torch))

    # Note transposing for librosa
    cepstral_coeff = fft.dct(grain_db)
    cepstral_coeff_torch = dct.dct(grain_db_torch)

    cepstral_coeff[:, :, 128:] = 0
    cepstral_coeff_torch[:, :, 128:] = 0

    # Use torch and dct for now
    cepstral_coeff = cepstral_coeff_torch

    # inv_cepstral_coeff = 10**(fft.idct(cepstral_coeff) / 20)
    inv_cepstral_coeff = 10**(dct.idct(cepstral_coeff) / 20)

    # filter_ir = dsp.amp_to_impulse_response_w_phase(torch.from_numpy(inv_cepstral_coeff), l_grain)
    filter_ir = dsp.amp_to_impulse_response_w_phase(inv_cepstral_coeff, l_grain)

    noise = generate_noise_grains(bs, n_grains, l_grain, filter_ir.dtype, filter_ir.device, hop_ratio=0.25)
    noise = noise.reshape(bs*n_grains, l_grain)

    audio = dsp.fft_convolve(noise, filter_ir)

    # Check if number of grains wanted is entered, else use the original
    audio = audio.reshape(-1,n_grains,l_grain)
    # audio = inv_grain_fft.reshape(-1,n_grains,l_grain)

    # Check if an overlapp add window has been passed, if not use that used in encoding.

    ola_window = signal.hann(l_grain,sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
    ola_windows = ola_windows
    audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

    audio = audio/torch.max(audio)


    # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
    # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
    # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
    # since kernel size is l_grain, this is needed in the second dimension.
    ola_folder = torch.nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
    audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

    # Normalise the energy values across the audio samples
    if NORMALIZE_OLA:
        # Normalises based on number of overlapping grains used in folding per point in time.
        unfolder = torch.nn.Unfold((l_grain,1),stride=(hop_size,1))
        input_ones = torch.ones(1,1,tar_l,1)
        ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
        ola_divisor = ola_divisor
        audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

    #for testing
    for i in range(audio_sum.shape[0]):
        torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/cc_filtering_{i}.wav', audio_sum[i].unsqueeze(0).cpu(), SAMPLE_RATE)
        
    
