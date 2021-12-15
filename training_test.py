import torch.cuda
import lstm.music_lstm as lstm
import lstm.music_training as training
import sys
import wandb
import torch
import random
import numpy as np
import logging
import util.coloredLogging as cl


def train(num_songs,
          hidden_dimension,
          learning_rate,
          cuda_device,
          wandb_enable):
    make_deterministic()

    if wandb_enable:
        wandb.init(project='nes-generator', entity='cs4100final')
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": num_songs, "batch_size": 1
        }

    model = lstm.MusicLSTM(hidden_dimension)
    if cuda_device is not None:
        print('Cuda device set')
        model.cuda(cuda_device)

    if wandb_enable:
        wandb.watch(model)

    model = training.trainModel(
        model,
        num_songs,
        learning_rate,
        cuda_device,
        wandb_enable
    )

    model.zero_grad()

    return model

def generate_songs(model,
                   out_folder,
                   num_songs,
                   cuda_device):
    return training.runModel(model, num_songs, out_folder, cuda_device)


def make_deterministic():
    # torch.use_deterministic_algorithms(True)  # Improves reproducibility but possibly reduces performance
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

def init_cuda(cuda_enable):
    cuda_device = None
    if cuda_enable:
        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            logging.info('Using CUDA device: ' + torch.cuda.get_device_name(cuda_device))
        else:
            logging.warning('No CUDA device detected; falling back to CPU.')
    return cuda_device


"""
logging.getLogger().setLevel(logging.INFO)
cl.init_colors()

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

HIDDEN_DIMENSION = 512
DEFAULT_SONGS = 25
DEFAULT_LEARNING_RATE = 0.01
NUM_OUTPUT_SONGS = 1

if '--help' in sys.argv:
    print('Tests the training code with way too few epochs')
    print('Usage: python ./training_test.py [num_songs] [...args]')
    print('args:')
    print('  --no-save')
    print('  --play')
    print('  --cpu')
    exit(0)

num_songs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SONGS

wandb.init(project='nes-generator', entity='cs4100final')
wandb.config = {
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": num_songs, "batch_size": 1
}
model = lstm.MusicLSTM(HIDDEN_DIMENSION)

cuda_device = None
if '--cpu' not in sys.argv:
    if torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
        logging.info('Using CUDA device: ' + torch.cuda.get_device_name(cuda_device))
        model.cuda(cuda_device)
    else:
        logging.warning('No CUDA device detected; falling back to CPU.')

wandb.watch(model)
model = training.trainModel(
    model,
    num_songs,
    DEFAULT_LEARNING_RATE,
    cuda_device
)
model.zero_grad()

print('\nwriting songs...')
outputs = training.runModel(model, NUM_OUTPUT_SONGS,
                            '--no-save' not in sys.argv,
                            cuda_device)

if '--play' in sys.argv:
    import expressive.player as player
    for song in outputs:
        player.play(song)
        
"""
