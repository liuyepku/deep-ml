def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np
	mat = np.array(matrix)
	if mode == "row":
		return np.mean(mat, axis=1).tolist()
	elif mode == "column":
		return np.mean(mat, axis=0).tolist()
