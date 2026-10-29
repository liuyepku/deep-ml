def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	import numpy as np
	mat = np.array(vectors,dtype = float)
	mMean = np.mean(mat, axis=1, keepdims = True)
	# print(mMean)

	center = mat-mMean
	n = mat.shape[1]
	# print(np.matmul(center, center.T))
	return (np.matmul(center, center.T)/(n-1)).tolist()