import numpy as np


class Dense:

    def __init__(self,
                 input_features,
                 output_features,
                 learning_rate=0.01):

        self.learning_rate = learning_rate

        self.weights = np.random.randn(
            input_features,
            output_features
        ) * np.sqrt(2 / input_features)

        self.biases = np.zeros(
            (1, output_features)
        )

    def forward(self, x):

        self.original_shape = x.shape

        if x.ndim > 2:
            self.input = x.reshape(x.shape[0], -1)
        else:
            self.input = x

        output = np.dot(
            self.input,
            self.weights
        ) + self.biases

        return output

    def backward(self, grad_output):

        batch_size = self.input.shape[0]

        grad_weights = np.dot(
            self.input.T,
            grad_output
        )

        grad_biases = np.sum(
            grad_output,
            axis=0,
            keepdims=True
        )

        grad_input = np.dot(
            grad_output,
            self.weights.T
        )

        self.weights -= (
            self.learning_rate *
            grad_weights /
            batch_size
        )

        self.biases -= (
            self.learning_rate *
            grad_biases /
            batch_size
        )

        return grad_input.reshape(
            self.original_shape
        )