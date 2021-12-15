import expressive.expressiveCodec as ExpressiveCodec
import expressive.internalCodec as InternalCodec
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


def getRandomFile():
    files = Path(BASE_FILE_PATH).rglob('*.exprsco.pkl')
    return random.choice(list(files))


def convertFile(path):
    exprsco = ExpressiveCodec.loadFile(path)
    return InternalCodec.expressiveToInternal(exprsco)


def prepareData(data, cuda_device):
    data = torch.unsqueeze(data, dim=0)
    data = torch.unsqueeze(data, dim=0)
    if cuda_device is not None:
        return data.cuda(cuda_device)
    return data


def trainSong(model, path, loss_function, optimizer, cuda_device, wandb_enable):
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
        if wandb_enable:
            wandb.log({'loss': loss})
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss / (len(internal) - 1)


def trainModel(model, num_songs, learning_rate, cuda_device=None, wandb_enable=True):
    printHeader('Training model')
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    if learning_rate:
        optimizer.learning_rate = learning_rate
    queue = []

    if cuda_device is not None:
        model.cuda(cuda_device)

    for index in range(num_songs):
        if len(queue) == 0:
            queue = list(Path(BASE_FILE_PATH).rglob('*.exprsco.pkl'))
            random.shuffle(queue)
        filepath = queue.pop()
        avg_loss = trainSong(model, filepath, loss_function, optimizer, cuda_device, wandb_enable)
        logging.debug(f'Trained on: {filepath} / Average loss: {avg_loss}')
        logging.info(f'{100 * index // num_songs}% ({index} / {num_songs})')

    return model


def makeSong(model, path, cuda_device):
    song = numpy.empty((0, 15))
    seedSong = convertFile(path)
    data = prepareData(seedSong[0], cuda_device)

    for _ in range(SONG_TIME_SECONDS * 24):
        data = model(data)

        next_step = data.detach().cpu().numpy()[0][0]
        song = numpy.append(song, numpy.array([next_step]), axis=0)

    return song


def runModel(model, num_songs, write_to=None, cuda_device=None):
    basepath = Path(BASE_FILE_PATH)
    outs = []
    for _ in range(num_songs):
        filepath = getRandomFile()
        song = makeSong(model, filepath, cuda_device)
        exprsco = InternalCodec.internalToExpressive(song)
        outs.append(exprsco)
        if write_to:
            outpath = Path(write_to).joinpath(
                Path(filepath).relative_to(basepath))
            ExpressiveCodec.saveFile(outpath, exprsco)
            logging.info(f'Saved {outpath}')
    return outs