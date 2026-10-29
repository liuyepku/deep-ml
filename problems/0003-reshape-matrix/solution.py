import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	matA = np.array(a)
	if matA.size!=np.prod(new_shape):
		return []
	matB = np.reshape(matA, new_shape)
	return matB.tolist()