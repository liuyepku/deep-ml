import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	
	
	mean = np.mean(data, axis=0, keepdims = True)
	std = np.std(data, axis=0, keepdims = True)
	std = np.where(std == 0, 1, std)
	standardized_data = (data - mean)/std

	minV = np.min(data, axis=0, keepdims = True)
	maxV = np.max(data, axis=0, keepdims = True)
	scale = maxV - minV
	scale = np.where(scale == 0, 1, scale)
	normalized_data = (data - minV)/scale
	return standardized_data, normalized_data