import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		h = np.array(initial_hidden_state)
		c = np.array(initial_cell_state)
		outputs = []
		outputs.append(h)

		for xt in x:
			xt = xt.reshape(-1, 1)

			zt = np.vstack([h, xt])
			ft = self.sigmoid(self.Wf @ zt + self.bf)
			it = self.sigmoid(self.Wi @ zt + self.bi)
			ct = np.tanh(self.Wc @ zt + self.bc)
			ot = self.sigmoid(self.Wo @ zt + self.bo)

			c = ft * c + it *ct
			h = ot * np.tanh(c)
			outputs.append(h)
		return outputs, h, c 

	
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
