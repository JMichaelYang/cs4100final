import lstm.music_training as training
import sys
import logging
import util.coloredLogging as cl

logging.getLogger().setLevel(logging.INFO)
cl.init_colors()

DEFAULT_SONGS = 25

if '--help' in sys.argv:
    print('Tests the training code with way too few epochs')
    print('Arguments:')
    print('  --no-save')
    print('  --play')
    exit(0)

model = training.trainModel(
    int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SONGS)
model.zero_grad()

print('\nwriting songs...')
outputs = training.runModel(model, 1, '--no-save' not in sys.argv)

if '--play' in sys.argv:
    import expressive.player as player
    for song in outputs:
        player.play(song)
