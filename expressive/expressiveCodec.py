import pickle
import logging
import os


def loadFile(path):
    filename = os.path.basename(path)
    if not path.endswith('.pkl'):
        message = f'Expected a Pickle file (.pkl); got: {filename}'
        logging.error(message)
        exit(IOError(message))
    try:
        with open(path, 'rb') as f:
            rate, nsamps, exprsco = pickle.load(f)
            
    except pickle.PickleError as e:
        logging.error(f'Failed to unpickle {filename}')
        logging.error(e)
        exit(1)
    except OSError as e:
        logging.error(f'Failed to open {filename}')
        logging.error(e)
        exit(1)

    logging.info(f'Opened {filename}')
    logging.debug(f'Temporal discretization rate: {rate}')
    logging.debug(f'Length of original VGM: {round(nsamps / 44100, 2)}s')
    
    return exprsco


def saveFile(path, exprsco):
    filename = os.path.basename(path)
    nsamps = int(44100 * (len(exprsco) / 24) + 10)
    if not path.endswith('.pkl'):
        path += '.pkl'
    
    try:
        with open(path, 'wb') as f:
            pickle.dump((24.0, nsamps, exprsco), f)
            logging.info(f'Saved {filename}')
    except pickle.PickleError as e:
        logging.error(f'Failed to pickle {filename}:')
        logging.error(e)
        exit(1)
    except OSError as e:
        logging.error(f'Failed to write to {filename}:')
        logging.error(e)
