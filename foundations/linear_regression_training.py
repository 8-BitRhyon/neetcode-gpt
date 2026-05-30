import numpy as np
from numpy.typing import NDArray

class Solution:
    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        weights = initial_weights.copy()
        N = len(X)

        for _ in range(num_iterations):
            predictions = X @ weights 
            
            gradients = (2 / N) * (X.T @ (predictions - Y))
            
            weights -= self.learning_rate * gradients

        return np.round(weights, 5)