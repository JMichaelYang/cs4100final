import lstm.music_training as training

model = training.trainModel(1)
model.zero_grad()

print('\nwriting songs...')
training.runModel(model, 3)