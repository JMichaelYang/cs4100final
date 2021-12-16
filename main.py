import argparse
import logging
import util.coloredLogging as cl

from expressive.expressiveCodec import loadFile
from expressive.player import play
from training_test import train, init_cuda, generate_songs


argparser = argparse.ArgumentParser()

argparser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug printing')

subparsers = argparser.add_subparsers(help='sub-command help', dest='program')


parser_train = subparsers.add_parser('train', help="Train a new model")

parser_train.add_argument('--input-songs', type=int, default=25,
                          help='How many songs to train the model with')

parser_train.add_argument('--epochs', type=int, default=25,
                          help='How many times to run over the input data')

parser_train.add_argument('--cpu', action='store_true', default=False)

parser_train.add_argument('--no-save', action='store_true', default=False)

parser_train.add_argument('--output-songs', type=int, default=1,
                          help='How many songs to generate after training')

parser_train.add_argument(
    '--disable-wandb', action='store_true', default=False)

parser_train.add_argument('--hidden-size', type=int, default=512,
                          help='How large should the hidden layer be?')

parser_train.add_argument('--learning-rate', type=int, default=0,
                          help='Controls how fast (and unstable) the learning is')

# parser_train.add_argument('--input-folder', type=str, default='./nesmdb24_exprsco/train')

parser_train.add_argument('--output-folder', type=str, default='./generated')

parser_play = subparsers.add_parser('play', help="Play a sound file")

parser_play.add_argument('location', type=str, nargs='?', default=None,
                         help='The file to play')

parser_play = subparsers.add_parser(
    'convert', help="Convert a pkl file to exprsco")

parser_play.add_argument('location', type=str, nargs='?', default=None,
                         help='The file to convert')

subparsers.default = None


def parse_subcommands(extra=None):
    args, extra = argparser.parse_known_args(extra)
    if len(extra):
        return [args] + parse_subcommands(extra)
    return [args]


def main():
    logging.getLogger().setLevel(logging.INFO)
    cl.init_colors()

    namespaces = parse_subcommands()
    training_output = None
    for namespace in namespaces:
        if namespace.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        if namespace.program == 'train':
            cl.printHeader('    Training model')
            logging.debug(f'epochs: {namespace.epochs}')
            logging.debug(f'input songs: {namespace.input_songs}')
            logging.debug(f'hidden dimension: {namespace.hidden_size}')
            logging.debug(
                f'learning rate: {namespace.learning_rate if namespace.learning_rate else "default"}')
            logging.debug(f'output songs: {namespace.output_songs}')
            if namespace.no_save:
                logging.debug('Not saving output')
            else:
                logging.debug(f'Output folder: {namespace.output_folder}')
            if namespace.disable_wandb:
                logging.debug('Wandb disabled')
            if namespace.cpu:
                logging.debug('CPU only')

            cuda_device = init_cuda(not namespace.cpu)

            model = train(namespace.input_songs,
                          namespace.epochs,
                          namespace.hidden_size,
                          namespace.learning_rate,
                          cuda_device,
                          not namespace.disable_wandb)

            if namespace.output_songs:
                cl.printHeader(
                    f'Generating {namespace.output_songs} song{"s" if namespace.output_songs > 1 else ""}')
                training_output = generate_songs(model,
                                                 namespace.output_folder if (
                                                     not namespace.no_save) else None,
                                                 namespace.output_songs,
                                                 cuda_device)

        if namespace.program == 'play':
            cl.printHeader('    Playing file')
            if training_output and not namespace.location:
                play(training_output)
            else:
                if not namespace.location:
                    namespace.location = './test/resources/295_SilverSurfer_02_03SectionStart.exprsco.pkl'
                data = loadFile(namespace.location)
                play(data)

        if namespace.program == 'convert':
            cl.printHeader('    Converting file')
            if not namespace.location:
                namespace.location = './test/resources/295_SilverSurfer_02_03SectionStart.exprsco.pkl'
            data = loadFile(namespace.location)
            print(data)

    print('Done.')


if __name__ == '__main__':
    main()
