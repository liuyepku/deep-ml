
from collections import Counter
import numpy as np

def confusion_matrix(data):
	# Implement the function here
	data = np.array(data)
	y_true = data[:,0]
	y_pred = data[:,1]
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	tn = np.sum((y_true == 0) & (y_pred == 0))
	return [[int(tp), int(fn)], [int(fp), int(tn)]]
