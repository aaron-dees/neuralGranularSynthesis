import torch

# Sample from a gaussian distribution
def sample_from_distribution(mu, log_variance, device, shape):

    # point = mu + sigma*sample(N(0,1))
    epsilon = torch.normal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
    sampled_point = mu + torch.exp(log_variance / 2) * epsilon

    return sampled_point