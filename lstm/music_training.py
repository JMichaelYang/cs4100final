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
import wandb

SONG_TIME_SECONDS = 30
BASE_FILE_PATH = r'./nesmdb24_exprsco/train'
BASE_WRITE_PATH = r'./generated'


def getRandomFile():
    files = Path(BASE_FILE_PATH).rglob('*.exprsco.pkl')
    return random.choice(list(files))


def convertFile(path):
    exprsco = codec.loadFile(path)
    return int_codec.expressiveToInternal(exprsco)


def prepareData(data):
    data = torch.unsqueeze(data, dim=0)
    return torch.unsqueeze(data, dim=0)


def trainSong(model, path, loss_function, optimizer):
    internal = convertFile(path)
    total_loss = 0

    # Clear the gradients
    model.zero_grad()

    for i, data in enumerate(internal[:-1]):
        # Run forward pass and calculate loss
        data = prepareData(data)
        predicted_data = model(data)
        next_data = prepareData(internal[i + 1])
        loss = loss_function(predicted_data, next_data)
        wandb.log({'loss': loss})
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss / (len(internal) - 1)


def trainModel(model, num_songs, learning_rate, cuda_enabled=False):
    printHeader('Training model')
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    queue = []

    for _ in range(num_songs):
        if len(queue) == 0:
            queue = list(Path(BASE_FILE_PATH).rglob('*.exprsco.pkl'))
            random.shuffle(queue)
        filepath = queue.pop()
        avg_loss = trainSong(model, filepath, loss_function, optimizer)
        logging.debug(f'Trained On: {filepath} / Average loss: {avg_loss}')

    return model


def makeSong(model, path):
    song = numpy.empty((0, 15))
    data = convertFile(path)
    data = data[0]
    data = torch.unsqueeze(data, dim=0)
    data = torch.unsqueeze(data, dim=0)

    for _ in range(SONG_TIME_SECONDS * 24):
        data = model(data)
        next_step = data.detach().numpy()[0][0]
        song = numpy.append(song, numpy.array([next_step]), axis=0)

    return song


def runModel(model, num_songs, learning_rate, save=True):
    basepath = Path(BASE_FILE_PATH)
    outs = []
    for _ in range(num_songs):
        filepath = getRandomFile()
        song = makeSong(model, filepath)
        exprsco = int_codec.internalToExpressive(song)
        outs.append(exprsco)
        if save:
            outpath = Path(BASE_WRITE_PATH).joinpath(
                Path(filepath).relative_to(basepath))
            codec.saveFile(outpath, exprsco)
    return outs
