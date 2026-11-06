import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	y = X @ w 
	mse = np.mean((y - y_true)**2)
	reg = np.sum(w**2)
	return float(mse + alpha*reg)
