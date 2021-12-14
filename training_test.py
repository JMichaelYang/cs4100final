import torch.cuda

import lstm.music_training as training
import sys
import logging
import util.coloredLogging as cl

logging.getLogger().setLevel(logging.INFO)
cl.init_colors()

DEFAULT_SONGS = 25

if '--help' in sys.argv:
    print('Tests the training code with way too few epochs')
    print('Usage: python ./training_test.py [num_songs] [...args]')
    print('args:')
    print('  --no-save')
    print('  --play')
    print('  --cuda    ' + ('(available)' if torch.cuda.is_available() else '(not available)'))
    exit(0)

cuda_device = None
if '--cuda' in sys.argv:
    if torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
        logging.info('Using CUDA device:', torch.cuda.get_device_name(cuda_device))
    else:
        logging.warning('No CUDA device detected; falling back to CPU.')

model = training.trainModel(
    int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SONGS,
    True if (('--cuda' in sys.argv) and torch.cuda.is_available()) else False
)
model.zero_grad()

print('\nwriting songs...')
outputs = training.runModel(model, 1, '--no-save' not in sys.argv, cuda_device)

if '--play' in sys.argv:
    import expressive.player as player
    for song in outputs:
        player.play(song)
