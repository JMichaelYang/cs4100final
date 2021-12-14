import lstm.music_lstm as lstm
import expressive.expressiveCodec as codec
import expressive.internalCodec as int_codec
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
import logging
from util.coloredLogging import printHeader

SONG_TIME_SECONDS = 30
HIDDEN_DIMENSION = 512
BASE_FILE_PATH = r'./nesmdb24_exprsco/train'
BASE_WRITE_PATH = r'./generated'


def getRandomFile():
    files = Path(BASE_FILE_PATH).rglob('*.exprsco.pkl')
    return random.choice(list(files))


def convertFile(path):
    exprsco = codec.loadFile(path)
    return int_codec.expressiveToInternal(exprsco)


def prepareData(data, cuda_device):
    data = torch.unsqueeze(data, dim=0)
    data = torch.unsqueeze(data, dim=0)
    if cuda_device is not None:
        return data.cuda(cuda_device)
    return data


def trainSong(model, path, loss_function, optimizer, cuda_device):
    internal = convertFile(path)

    total_loss = 0

    # Clear the gradients
    model.zero_grad()
    next_data = prepareData(internal[0], cuda_device)

    for i, data in enumerate(internal[:-1]):
        # Run forward pass and calculate loss
        data = next_data
        predicted_data = model(data)
        next_data = prepareData(internal[i + 1], cuda_device)
        loss = loss_function(predicted_data, next_data)
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss / (len(internal) - 1)


def trainModel(num_songs, cuda_device = None):
    printHeader('Training model')

    model = lstm.MusicLSTM(HIDDEN_DIMENSION)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    queue = []

    if cuda_device is not None:
        model.cuda(cuda_device)

    for _ in range(num_songs):
        if len(queue) == 0:
            queue = list(Path(BASE_FILE_PATH).rglob('*.exprsco.pkl'))
            random.shuffle(queue)
        filepath = queue.pop()
        logging.debug(f'Training on: {filepath}')
        avg_loss = trainSong(model, filepath, loss_function, optimizer, cuda_device)
        logging.debug(f'Average loss: {avg_loss}')

    return model


def makeSong(model, path, cuda_device):
    song = numpy.empty((0, 15))
    data = convertFile(path)
    data = data[0]
    data = prepareData(data, cuda_device)

    for _ in range(SONG_TIME_SECONDS * 24):
        data = model(data)
        next_step = data.detach().numpy()[0][0]
        song = numpy.append(song, numpy.array([next_step]), axis=0)

    return song


def runModel(model, num_songs, save=True):
    basepath = Path(BASE_FILE_PATH)
    outs = []
    for _ in range(num_songs):
        filepath = getRandomFile()
        song = makeSong(model, filepath)
        exprsco = int_codec.internalToExpressive(song)
        print(exprsco)
        outs.append(exprsco)
        if save:
            outpath = Path(BASE_WRITE_PATH).joinpath(
                Path(filepath).relative_to(basepath))
            codec.saveFile(outpath, exprsco)
    return outs
