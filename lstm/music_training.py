import music_lstm as lstm
import expressive.codec as codec
import expressive.internalCodec as int_codec
import torch.nn as nn
import torch.optim as optim
import os
import random

HIDDEN_DIMENSION = 15
BASE_FILE_PATH = "../test/resources/training"

def getRandomFile():
  files = os.listdir(BASE_FILE_PATH)
  return files[random.randrange(len(files))]

def convertFile(path):
  exprsco = codec.loadFile(path)
  return int_codec.expressiveToInternal(exprsco)

def trainSong(model, path, loss_function, optimizer):
  internal = convertFile(path)

  # Clear the gradients
  model.zero_grad()

  for i, data in enumerate(internal[:-1]):
    # Run forward pass and calculate loss
    predicted_data = model(data)
    next_data = internal[i + 1]
    loss = loss_function(predicted_data, next_data)
    loss.backward()
    optimizer.step()

def trainModel(num_songs):
  model = lstm.MusicLSTM(HIDDEN_DIMENSION)
  loss_function = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)

  for _ in range(num_songs):
    filepath = getRandomFile()
    trainSong(model, filepath, loss_function, optimizer)

  return model
