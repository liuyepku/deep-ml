def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np
	mat = np.array(matrix)
	eigenvalues = np.linalg.eigvals(mat).real.tolist()
	eigenvalues.sort(reverse = True)
	return eigenvalues