import torch

# if torch.backends.mps.is_available():
#    DEVICE = torch.device("mps:0")
# else:
DEVICE = torch.device("cpu") 

print("DEVICE: ", DEVICE)
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda:0")
# else:
#     DEVICE = torch.device("cpu") 

# having issues with padding on Apple GPU
DATALOADER_DEVICE = torch.device('cpu')

# Hyper Parameters
BATCH_SIZE = 1
TEST_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.001
LATENT_SIZE = 128
ENV_DIST = 0
KERNEL_SIZE = 3
# BETA_PARAMS
# Number of warmup iterations before increasing beta
BETA_WARMUP_START_PERC = 0.1
TARGET_BETA = 0.01
# number of warmup steps over half max_steps
# BETA_STEPS = 500
BETA_STEPS = 1

# Audio Processing Parameters
# ANNOTATIONS_FILE = "/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
# AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/audio"
ANNOTATIONS_FILE = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/esc50_seaWaves.csv"
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/5secs/small"

# FRAME_SIZE = 1024
# HOP_LENGTH = 512
NUM_MELS = 64
SAMPLE_RATE = 44100
NUM_CC = 128
# DURATION = NUM_SAMPLES / SAMPLE_RATE # SECONDS
# MONO = True
NORMALIZE_OLA = True
POSTPROC_KER_SIZE = 65
POSTPROC_CHANNELS = 5

# Grain Params
# Ratio of hop size to grain length
HOP_SIZE_RATIO = 0.25
# GRAIN_LENGTH = 2048
GRAIN_LENGTH = 1024
# Target length in seconds
TARGET_LENGTH = 5.0
# High pass frequencey for bi-quad filtering
HIGH_PASS_FREQ = 50

# Mode and directories
WANDB = False
TRAIN = False
EXPORT_LATENTS = False
SAVE_CHECKPOINT = False
CHECKPOINT_REGULAIRTY = 20
LOAD_CHECKPOINT = True
VIEW_LATENT = False
SAVE_RECONSTRUCTIONS = True
COMPARE_ENERGY = False
SAVE_DIR = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints"
# SAVE_DIR = f"/home/ICTDOMAIN/d22127229/code/github/neuralGranularSynthesis/models/saved_models"
# RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions/one_sec"
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/scripts"
CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/spectrogram_models/spec_vae_100epochs.pt"
# CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_cpu_{EPOCHS}epochs_{64}batch_{BETA}beta_{ENV_DIST}envdist_latest.pt"
