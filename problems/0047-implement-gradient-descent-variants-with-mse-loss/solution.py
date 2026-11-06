import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    X = np.array(X)
    y = np.array(y)
    w = np.array(weights)

    m = len(X)

    if method == 'batch':
        batch_size = m
    elif method == 'stochastic':
        batch_size = 1
    
    for i in range(n_epochs):
        for k in range(0, m, batch_size):
            x_batch = X[k: k+batch_size]
            y_batch = y[k: k+batch_size]
            y_pred = x_batch @ w 
            error = y_pred - y_batch
            grad = (2/len(y_batch)) * x_batch.T @ error
            w -= learning_rate * grad
    return w 





