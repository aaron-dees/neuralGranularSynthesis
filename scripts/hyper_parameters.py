import torch

# Define all constants

DEVICE = torch.device('mps:0')
# DEVICE = torch.device('cpu')
TRAIN = True
BATCH_SIZE = 10
EPOCHS = 100
LEARNING_RATE = 0.0005
SAVE_MODEL = True
LOAD_MODEL = False
MODEL_PATH = '/Users/adees/Code/neural_granular_synthesis/models/saved_models/fsdd_vae_gpu_100epochs_10batch.pt'
HOP_LENGTH = 256
MIN_MAX_VALUES_PATH = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/min_max_values.pkl"
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions"
LATENT_SIZE = 128