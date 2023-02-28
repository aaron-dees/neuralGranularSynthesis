import torch
import matplotlib.pyplot as plt

# Sample from a gaussian distribution
def sample_from_distribution(mu, log_variance, device, shape):

    # point = mu + sigma*sample(N(0,1))
    epsilon = torch.normal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
    sampled_point = mu + torch.exp(log_variance / 2) * epsilon

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