import sys
sys.path.append('../')

from models.vae import VAE

if __name__ == "__main__":

    print("Training Script")

    vae = VAE()

    print(vae)