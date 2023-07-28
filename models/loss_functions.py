import torch 
from torchaudio.transforms import Spectrogram,MelSpectrogram
import torch.nn as nn

from utils.utilities import safe_log

#################
# Loss Functions
#################

#################
# Waveform VAEs losses
#################

# Note the below calculates the kl divergence per waveform.
def compute_kld(mu, logvar):

    mu = torch.flatten(mu, start_dim=1)
    logvar = torch.flatten(logvar, start_dim=1)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

    return kld_loss

class spectral_distances(nn.Module):
    def __init__(self,stft_scales=[2048, 1024, 512, 256, 128], mel_scales=[2048, 1024], spec_power=1, mel_dist=True, log_dist=0, sr=16000, device="cpu"):
        super(spectral_distances, self).__init__()
        self.stft_scales = stft_scales
        self.mel_scales = mel_scales
        self.mel_dist = mel_dist
        self.log_dist = log_dist
        T_spec = []
        for scale in stft_scales:
            T_spec.append(Spectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,power=spec_power).to(device))
        self.T_spec = T_spec
        if mel_dist:
            # print("\n*** training with MelSpectrogram distance")
            T_mel = []
            for scale in mel_scales:
                T_mel.append(MelSpectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,sample_rate=sr,f_min=50.,n_mels=scale//4,power=spec_power).to(device))
            self.T_mel = T_mel
    
    def forward(self,x_inp,x_tar):
        loss = 0
        n_scales = 0
        for i,scale in enumerate(self.stft_scales):
            S_inp,S_tar = self.T_spec[i](x_inp),self.T_spec[i](x_tar)
            stft_dist = (S_inp-S_tar).abs().mean()
            loss = loss+stft_dist
            n_scales += 1
            if self.log_dist>0:
                loss = loss+(safe_log(S_inp)-safe_log(S_tar)).abs().mean()*self.log_dist
                n_scales += self.log_dist
        if self.mel_dist:
            for i,scale in enumerate(self.mel_scales):
                M_inp,M_tar = self.T_mel[i](x_inp),self.T_mel[i](x_tar)
                mel_dist = (M_inp-M_tar).abs().mean()
                loss = loss+mel_dist
                n_scales += 1
                if self.log_dist>0:
                    loss = loss+(safe_log(M_inp)-safe_log(M_tar)).abs().mean()*self.log_dist
                    n_scales += self.log_dist
        return loss/n_scales

def envelope_distance(x_inp,x_tar,n_fft=1024,log=True):

    # Reshapes, but are these really needed [bs, n_grains, num_samples] --> [bs*n_grains, num_samples]
    x_inp = x_inp.reshape(x_inp.shape[0]*x_inp.shape[1], x_inp.shape[2])
    x_tar = x_tar.reshape(x_tar.shape[0]*x_tar.shape[1], x_tar.shape[2])

    env_inp = torch.stft(x_inp, n_fft, hop_length=n_fft//4, onesided=True, return_complex=False)
    env_inp = torch.mean(env_inp[:,:,:,0]**2+env_inp[:,:,:,1]**2,1)
    env_tar = torch.stft(x_tar, n_fft, hop_length=n_fft//4, onesided=True, return_complex=False)
    env_tar = torch.mean(env_tar[:,:,:,0]**2+env_tar[:,:,:,1]**2,1)
    if log:
        env_inp,env_tar = safe_log(env_inp),safe_log(env_tar)
    return (env_inp-env_tar).abs().mean()

############
# Others
############

def calc_reconstruction_loss(target, prediction):

    # MSE
    error = target - prediction
    reconstruction_loss = torch.mean(error**2)

    return reconstruction_loss

# This calculates the kl divergence 
def calc_kl_loss(mu, log_variance):

    # KL Divergence between predicted gaussian distribution and standard guassian distribution N(0,1)
    kl_loss = - 0.5 * torch.sum(1 + log_variance - torch.square(mu) - torch.exp(log_variance))

    return kl_loss



def calc_combined_loss(target, prediction, mu, log_variance, reconstruction_loss_weight):

    reconstruction_loss = calc_reconstruction_loss(target, prediction)
    kl_loss = calc_kl_loss(mu, log_variance)
    combined_loss = (reconstruction_loss_weight * reconstruction_loss) + kl_loss

    return combined_loss, kl_loss, reconstruction_loss

def compute_losses(self, batch, beta):
    audio,labels = batch
    audio = audio.to(self.device)
    # forward
    audio_output,encoder_outputs = self.forward(audio, sampling=True)
    # compute losses
    spec_loss = self.spec_dist(audio_output,audio)
    if beta>0:
        kld_loss = compute_kld(encoder_outputs["mu"],encoder_outputs["logvar"])*beta
    else:
        kld_loss = 0
    if self.env_dist>0:
        env_loss = envelope_distance(audio_output,audio,n_fft=1024,log=True)*self.env_dist
    else:
        env_loss = 0
    return {"spec_loss":spec_loss,"kld_loss":kld_loss,"env_loss":env_loss}