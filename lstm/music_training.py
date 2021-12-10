import lstm.music_lstm as lstm
import expressive.codec as codec
import expressive.internalCodec as int_codec
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random

SONG_TIME_SECONDS = 30
HIDDEN_DIMENSION = 15
BASE_FILE_PATH = "D:\\development\\source\\school\\cs4100\\nesmdb\\train"
BASE_WRITE_PATH = "D:\\development\\source\\school\\cs4100\\generated"

def getRandomFile():
  files = os.listdir(BASE_FILE_PATH)
  return files[random.randrange(len(files))]

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
    total_loss += loss
    loss.backward()
    optimizer.step()

  return total_loss / (len(internal) - 1)

def trainModel(num_songs):
  model = lstm.MusicLSTM(HIDDEN_DIMENSION)
  loss_function = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)

  for _ in range(num_songs):
    filepath = getRandomFile()
    print(f'Training on: {filepath}')
    avg_loss = trainSong(model, f'{BASE_FILE_PATH}\\{filepath}', loss_function, optimizer)
    print(f'Average loss: {avg_loss}')

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

def runModel(model, num_songs):
  for _ in range(num_songs):
    filepath = getRandomFile()
    song = makeSong(model, f'{BASE_FILE_PATH}\\{filepath}')
    exprsco = int_codec.internalToExpressive(song)
    codec.saveFile(f'{BASE_WRITE_PATH}\\{filepath}', exprsco)