import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.array(scores)
    scores = scores  - np.max(scores)
    sumV = np.sum(np.exp(scores))
    scores = np.exp(scores)/sumV
    return scores