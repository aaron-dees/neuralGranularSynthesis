import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution
from scripts.configs.hyper_parameters_waveform import BATCH_SIZE, DEVICE, LATENT_SIZE
from utils.dsp_components import noise_filtering, mod_sigmoid


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from scipy import signal

class linear_block(nn.Module):
    def __init__(self, in_size,out_size,norm="BN"):
        super(linear_block, self).__init__()
        if norm=="BN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.BatchNorm1d(out_size),nn.LeakyReLU(0.2))
        if norm=="LN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.LayerNorm(out_size),nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.block(x)

# ------------
# LATENT AUTO-ENCODER v1
# ------------

class LatentEncoder_v1(nn.Module):

    def __init__(self,
                 e_dim,
                 z_dim,
                 h_dim,
                 n_linears,
                 rnn_type,
                 n_RNN,
                 n_grains,
                #  classes,
                #  conditional,
                #  lr
                    ):
        super(LatentEncoder_v1, self).__init__()

        self.flatten_size = n_grains * z_dim

        encoder_z = [linear_block(self.flatten_size, h_dim, norm="LN")]
        encoder_z += [linear_block(h_dim, h_dim, norm="LN") for i in range(1, n_linears)]
        self.encoder_z = nn.Sequential(*encoder_z)

        self.mu = nn.Linear(h_dim, e_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim, e_dim), nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipped to avoid numerical instabilities



    def encode(self, z):

        h = z.reshape(z.shape[0], self.flatten_size)

        h = self.encoder_z(h)

        mu = self.mu(h)
        logvar = self.logvar(h)

        e = sample_from_distribution(mu, logvar)

        return e, mu, logvar

    def forward(self, latents):

        e, mu, logvar = self.encode(latents)

        return e, mu, logvar
    

class LatentDecoder_v1(nn.Module):

    def __init__(self,
                 e_dim,
                 z_dim,
                 h_dim,
                 n_linears,
                 rnn_type,
                 n_RNN,
                 n_grains,
                 classes,
                 conditional,
                #  lr
                    ):
        super(LatentDecoder_v1, self).__init__()

        self.n_grains = n_grains
        self.flatten_size = n_grains * z_dim
        self.z_dim = z_dim

        decoder_z = [linear_block(e_dim, h_dim, norm="LN")] 
        decoder_z += [linear_block(h_dim, h_dim, norm="LN") for i in range(1,n_linears)]
        decoder_z += [nn.Linear(h_dim, self.flatten_size)]
        self.decoder_z = nn.Sequential(*decoder_z)


    def decode(self, e, conds=None):

        # In theory here I could create the final output of the dense layer be a larger seq? 
        z = self.decoder_z(e)

        z = z.reshape(z.shape[0], self.n_grains, self.z_dim)

        return z

    def forward(self, latents, conds=None):

        z = self.decode(latents, conds)

        return z
    

class LatentVAE_v1(nn.Module):

    def __init__(self,
                e_dim,
                z_dim,
                h_dim,
                n_linears,
                rnn_type,
                n_RNN,
                n_grains,
                classes,
                conditional,
                    ):
        super(LatentVAE_v1, self).__init__()

        self.Encoder = LatentEncoder_v1(
            e_dim=e_dim,
            z_dim=z_dim,
            h_dim=h_dim,
            n_linears=n_linears,
            rnn_type=rnn_type,
            n_RNN=n_RNN,
            n_grains=n_grains,
        )
        self.Decoder = LatentDecoder_v1(
            e_dim=e_dim,
            z_dim=z_dim,
            h_dim=h_dim,
            n_linears=n_linears,
            n_grains=n_grains,
            rnn_type=rnn_type,
            n_RNN=n_RNN,
            classes=classes,
            conditional=conditional
        )

        # Number of convolutional layers
    def encode(self, z):

         # z ---> e
        e, mu, log_variance = self.Encoder(z);
    
        return {"e":e,"mu":mu,"logvar":log_variance} 

    def decode(self, e, mu, sampling=True):

        if sampling:
            z_hat = self.Decoder(e)
        else:
            z_hat = self.Decoder(mu) 
        
        return z_hat

    def forward(self, z, conds=None, sampling=True):

        # z ---> e
        
        e, mu, log_variance = self.Encoder(z);

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            z_hat = self.Decoder(e, conds)
        else:
            z_hat = self.Decoder(mu, conds)

        return z_hat, e, mu, log_variance
    
# ------------
# LATENT AUTO-ENCODER (temporal embedding)
# ------------

class LatentEncoder(nn.Module):

    def __init__(self,
                 e_dim,
                 z_dim,
                 h_dim,
                 n_linears,
                 rnn_type,
                 n_RNN,
                #  n_grains,
                #  classes,
                #  conditional,
                #  lr
                    ):
        super(LatentEncoder, self).__init__()

        encoder_z = [linear_block(z_dim, h_dim, norm="LN")]
        encoder_z += [linear_block(h_dim, h_dim, norm="LN") for i in range(1, n_linears)]
        self.encoder_z = nn.Sequential(*encoder_z)
        self.rnn_type = rnn_type

        if rnn_type=="GRU":
            self.encoder_rnn = nn.GRU(h_dim,h_dim,num_layers=n_RNN,batch_first=True)
        if rnn_type=="LSTM":
            self.encoder_rnn = nn.LSTM(h_dim,h_dim,num_layers=n_RNN,batch_first=True)

        encoder_e = [linear_block(h_dim, h_dim) for i in range(1, n_linears)]
        encoder_e += [linear_block(h_dim, e_dim)]
        self.encoder_e = nn.Sequential(*encoder_e)

        self.mu = nn.Linear(e_dim, e_dim)
        self.logvar = nn.Sequential(nn.Linear(e_dim, e_dim), nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipped to avoid numerical instabilities



    def encode(self, z):

        h = self.encoder_z(z)

        h,h_n = self.encoder_rnn(h)
        # h contains the output features from the last layer of the LSTM, for each t.
        # h_n contains final cell state and final hidden state 

        if self.rnn_type == "LSTM":
            # discard the LSTM cell state
            h_n = h_n[0]

        h = self.encoder_e(h_n[-1,:,:])

        mu = self.mu(h)
        logvar = self.logvar(h)

        e = sample_from_distribution(mu, logvar)

        return e, mu, logvar

    def forward(self, latents):

        e, mu, logvar = self.encode(latents)

        return e, mu, logvar
    

class LatentDecoder(nn.Module):

    def __init__(self,
                 e_dim,
                 z_dim,
                 h_dim,
                 n_linears,
                 rnn_type,
                 n_RNN,
                 n_grains,
                 classes,
                 conditional,
                #  lr
                    ):
        super(LatentDecoder, self).__init__()

        # when len(classes)>1 and conditional=True then the decoder receives a one-hot vector of size len(classes)
        # TODO: replace one-hot condition with FiLM modulation ? or add Fader-Network regularization to the encoder
        if conditional is True and len(classes)>1:
            self.n_conds = len(classes)
            print("\n--- training latent VAE with class conditioning over",classes)
        else:
            self.n_conds = 0

        decoder_e = [linear_block(e_dim+self.n_conds, h_dim)]
        decoder_e += [linear_block(h_dim, h_dim) for i in range(1, n_linears)]
        self.decoder_e = nn.Sequential(*decoder_e)

        self.n_grains = n_grains

        if rnn_type=="GRU":
            self.decoder_rnn = nn.GRU(h_dim,h_dim,num_layers=n_RNN,batch_first=True)
        if rnn_type=="LSTM":
            self.decoder_rnn = nn.LSTM(h_dim,h_dim,num_layers=n_RNN,batch_first=True)

        decoder_z = [linear_block(h_dim+self.n_conds, h_dim, norm="LN")] # Granular conditioning after the RNN
        decoder_z += [linear_block(h_dim, h_dim, norm="LN") for i in range(1,n_linears)]
        decoder_z += [nn.Linear(h_dim, z_dim)]
        self.decoder_z = nn.Sequential(*decoder_z)


    def decode(self, e, conds=None):

        # input embedding of shape [N,e_dim] and conds of shape [N] (long)
        if self.n_conds>0:
            conds = F.one_hot(conds, num_classes=self.n_conds)
            e = torch.cat((e,conds),1)

        h = self.decoder_e(e)
        h = h.unsqueeze(1).repeat(1,self.n_grains,1).contiguous()

        # otherwise could use an auto-regressive approach if mean seaking
        # e.g. https://stackoverflow.com/questions/65205506/lstm-autoencoder-problems
        h,_ = self.decoder_rnn(h)

        if self.n_conds>0:
            conds = conds.unsqueeze(1).repeat(1,self.n_grains,1).contiguous()
            h = torch.cat((h,conds),2)

        # In theory here I could create the final output of the dense layer be a larger seq? 
        z = self.decoder_z(h)

        return z

    def forward(self, latents, conds=None):

        z = self.decode(latents, conds)

        return z
    

class LatentVAE(nn.Module):

    def __init__(self,
                e_dim,
                z_dim,
                h_dim,
                n_linears,
                rnn_type,
                n_RNN,
                n_grains,
                classes,
                conditional,
                    ):
        super(LatentVAE, self).__init__()

        self.Encoder = LatentEncoder(
            e_dim=e_dim,
            z_dim=z_dim,
            h_dim=h_dim,
            n_linears=n_linears,
            rnn_type=rnn_type,
            n_RNN=n_RNN,
        )
        self.Decoder = LatentDecoder(
            e_dim=e_dim,
            z_dim=z_dim,
            h_dim=h_dim,
            n_linears=n_linears,
            n_grains=n_grains,
            rnn_type=rnn_type,
            n_RNN=n_RNN,
            classes=classes,
            conditional=conditional
        )

        # Number of convolutional layers
    def encode(self, z):

         # z ---> e
        e, mu, log_variance = self.Encoder(z);
    
        return {"e":e,"mu":mu,"logvar":log_variance} 

    def decode(self, e, mu, sampling=True):

        if sampling:
            z_hat = self.Decoder(e)
        else:
            z_hat = self.Decoder(mu) 
        
        return z_hat

    def forward(self, z, conds=None, sampling=True):

        # z ---> e
        
        e, mu, log_variance = self.Encoder(z);

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            z_hat = self.Decoder(e, conds)
        else:
            z_hat = self.Decoder(mu, conds)

        return z_hat, e, mu, log_variance

