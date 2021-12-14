import lstm.music_lstm as lstm
import lstm.music_training as training
import sys
import wandb
import torch
import random
import numpy as np
import logging
import util.coloredLogging as cl

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
    print('  --cuda')
    exit(0)

num_songs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SONGS

wandb.init(project='nes-generator', entity='cs4100final')
wandb.config = {
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": num_songs, "batch_size": 1
}
model = lstm.MusicLSTM(HIDDEN_DIMENSION)
wandb.watch(model)
model = training.trainModel(
    model,
    num_songs,
    DEFAULT_LEARNING_RATE,
    'cuda' if '--cuda' in sys.argv else 'cpu'
)
model.zero_grad()

print('\nwriting songs...')
outputs = training.runModel(model, NUM_OUTPUT_SONGS,
                            '--no-save' not in sys.argv)

if '--play' in sys.argv:
    import expressive.player as player
    for song in outputs:
        player.play(song)
