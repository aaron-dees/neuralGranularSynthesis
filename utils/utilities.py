import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

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

# Safe log for cases where x is very close to zero
def safe_log(x, eps=1e-7):
    return torch.log(x + eps)