import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	X = np.array(features)
	y = np.array(labels)
	w = np.array(weights)

	pred = X @ w + bias
	probabilities = 1 / (1 + np.exp(-pred))
	mse = np.mean(( probabilities - labels)**2)
	# Your code here
	return np.round(probabilities, 4).tolist(), np.round(mse, 4)