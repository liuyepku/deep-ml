import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	X = np.array(features)
	y = np.array(labels)
	w = np.array(initial_weights)
	b = float(initial_bias)
	mse_values = []
	m = len(labels)

	for i in range(epochs):
		pred = X @ w + b 
		prob = 1 / (1 + np.exp(-pred))

		mse = np.mean((y - prob)**2)
		mse_values.append(np.round(mse, 4))

		delta = (2/m) * (prob - y) * prob * (1 - prob)
		dw = X.T @ delta
		db = np.sum(delta)

		w -= learning_rate*dw
		b -= learning_rate*db
	updated_weights = np.round(w, 4).tolist() 
	updated_bias = np.round(b, 4).tolist()	
	return updated_weights, updated_bias, mse_values