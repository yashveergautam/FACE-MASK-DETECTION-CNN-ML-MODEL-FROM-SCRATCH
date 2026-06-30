import numpy as np


class Softmax:

    def forward(self, x):

        shifted = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        exp = np.exp(shifted)

        probabilities = exp / np.sum(
            exp,
            axis=1,
            keepdims=True
        )

        return probabilities