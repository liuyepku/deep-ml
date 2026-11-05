import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	h = np.array(initial_hidden_state)
	wx = np.array(Wx)
	wh = np.array(Wh)
	b = np.array(b)

	for x in input_sequence:
		xt = np.array(x)
		h = np.tanh( wx @ xt + wh @ h + b)

	return h.tolist()