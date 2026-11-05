import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	x = np.array(scores)
	shift = x - np.max(x)

	log_sum = np.log(np.sum(np.exp(shift)))
	return (shift - log_sum).tolist()