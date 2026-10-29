import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	a = np.array(A)
	t = np.array(T)
	s = np.array(S)
	if np.isclose(np.linalg.det(t), 0) or np.isclose(np.linalg.det(s), 0):
		return -1
	res = np.dot(np.linalg.inv(t), a)
	return np.dot(res, s)