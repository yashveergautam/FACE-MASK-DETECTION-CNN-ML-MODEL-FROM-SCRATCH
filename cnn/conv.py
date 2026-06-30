import numpy as np


class Conv2D:
    """
    2D Convolution Layer implemented completely from scratch.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        learning_rate=0.01
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        limit = np.sqrt(
            2 / (in_channels * kernel_size * kernel_size)
        )

        self.weights = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * limit

        self.biases = np.zeros(
            (out_channels, 1)
        )

    def forward(self, x):

        self.input = x

        batch_size, channels, height, width = x.shape

        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        output = np.zeros(
            (
                batch_size,
                self.out_channels,
                output_height,
                output_width
            )
        )

        for i in range(output_height):

            for j in range(output_width):

                region = x[
                    :,
                    :,
                    i:i+self.kernel_size,
                    j:j+self.kernel_size
                ]

                output[:, :, i, j] = (

                    np.sum(

                        region[:, np.newaxis] *
                        self.weights,

                        axis=(2, 3, 4)

                    )

                    + self.biases.T

                )

        return output

    def backward(self, grad_output):

        batch_size = self.input.shape[0]

        grad_input = np.zeros_like(self.input)

        grad_weights = np.zeros_like(self.weights)

        grad_biases = np.sum(
            grad_output,
            axis=(0, 2, 3)
        ).reshape(self.out_channels, 1)

        output_height = grad_output.shape[2]
        output_width = grad_output.shape[3]

        for i in range(output_height):

            for j in range(output_width):

                region = self.input[
                    :,
                    :,
                    i:i+self.kernel_size,
                    j:j+self.kernel_size
                ]

                for k in range(self.out_channels):

                    grad_weights[k] += np.sum(

                        region *

                        grad_output[
                            :,
                            k,
                            i,
                            j
                        ][:, None, None, None],

                        axis=0

                    )

                    grad_input[
                        :,
                        :,
                        i:i+self.kernel_size,
                        j:j+self.kernel_size
                    ] += (

                        self.weights[k]

                        *

                        grad_output[
                            :,
                            k,
                            i,
                            j
                        ][:, None, None, None]

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

        return grad_input