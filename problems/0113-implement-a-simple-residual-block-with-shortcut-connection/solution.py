import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	h = np.maximum(0, w1 @ x)
	h = w2 @ h
	y = np.maximum(0, h+x)
	return np.round(y, 4).tolist()

