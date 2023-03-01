import torch 

#################
# Loss Functions
#################

def calc_reconstruction_loss(target, prediction):

    # MSE
    error = target - prediction
    reconstruction_loss = torch.mean(error**2)

    return reconstruction_loss

def calc_kl_loss(mu, log_variance):

    # KL Divergence
    kl_loss = - 0.5 * torch.sum(1 + log_variance - torch.square(mu) - torch.exp(log_variance))

    return kl_loss

def calc_combined_loss(target, prediction, mu, log_variance, reconstruction_loss_weight):

    reconstruction_loss = calc_reconstruction_loss(target, prediction)
    kl_loss = calc_kl_loss(mu, log_variance)
    combined_loss = (reconstruction_loss_weight * reconstruction_loss) + kl_loss

    return combined_loss, kl_loss, reconstruction_loss