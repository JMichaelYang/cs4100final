import torch
import numpy as np

def expressiveToInternal(expressive):
	# Normalizes our values
	internal = []
	for frame in expressive:
		internal.append([
			float(frame[0][1] > 0), mapRange(frame[0][0], 32, 108), mapRange(frame[0][1], 0, 15), mapRange(frame[0][2], 0, 3),
			float(frame[1][1] > 0), mapRange(frame[1][0], 32, 108), mapRange(frame[1][1], 0, 15), mapRange(frame[1][2], 0, 3),
			float(frame[2][0] > 0), mapRange(frame[2][0], 21, 108),
			float(frame[3][1] > 0), mapRange(frame[3][0], 1, 16), mapRange(frame[3][1], 0, 15), frame[3][2],
			0])

	# Sets the end to 1
	internal[-1][-1] = 1

	# Convert to tensor
	return torch.tensor(internal)


def internalToExpressive(internal, threshold=0.5):
	expressive = []
	for frame in internal:
		expressive.append([
			[mapRange(frame[1], 0, 1, 32, 108), mapRange((frame[2] * float(frame[0] > threshold)), 0, 1, 0, 15), mapRange(frame[3], 0, 1, 0, 3)],
			[mapRange(frame[5], 0, 1, 32, 108), mapRange((frame[6] * float(frame[4] > threshold)), 0, 1, 0, 15), mapRange(frame[7], 0, 1, 0, 3)],
			[mapRange(frame[9], 0, 1, 21, 108) * float(frame[8] > threshold), 0, 0],
			[mapRange(frame[11], 0, 1, 1, 16), mapRange((frame[12] * float(frame[10] > threshold)), 0, 1, 0, 15), frame[13]]])

	expressive = [[[round(float(param)) for param in channel] for channel in frame] for frame in expressive]
	return np.array(expressive)


def mapRange(value, minInput, maxInput, minOutput=0, maxOutput=1):
	normalized = max((value - minInput) / (maxInput - minInput), 0)
	return normalized * (maxOutput - minOutput) + minOutput
