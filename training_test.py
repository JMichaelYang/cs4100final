import lstm.music_training as training
import sys
import logging
import util.coloredLogging as cl

logging.getLogger().setLevel(logging.INFO)
cl.init_colors()

if '--help' in sys.argv:
    print('Tests the training code with way too few epochs')
    print('Arguments:')
    print('  --no-save')
    print('  --play')
    exit(0)

model = training.trainModel(1)
model.zero_grad()

print('\nwriting songs...')
outputs = training.runModel(model, 3, '--no-save' not in sys.argv)

if '--play' in sys.argv:
    import expressive.player as player
    for song in outputs:
        player.play(song)
