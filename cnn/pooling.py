import numpy as np


class MaxPooling2D:

    def __init__(self, pool_size=2):

        self.pool_size = pool_size

        self.input = None

    def forward(self, x):

        self.input = x

        batch_size, channels, height, width = x.shape

        out_height = height // self.pool_size

        out_width = width // self.pool_size

        output = np.zeros(

            (
                batch_size,
                channels,
                out_height,
                out_width
            )

        )

        for i in range(out_height):

            for j in range(out_width):

                region = x[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ]

                output[:, :, i, j] = np.max(
                    region,
                    axis=(2,3)
                )

        return output

    def backward(self, grad_output):

        batch_size, channels, height, width = self.input.shape

        grad_input = np.zeros_like(self.input)

        out_height = grad_output.shape[2]

        out_width = grad_output.shape[3]

        for i in range(out_height):

            for j in range(out_width):

                region = self.input[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ]

                max_mask = (

                    region

                    ==

                    np.max(
                        region,
                        axis=(2,3),
                        keepdims=True
                    )

                )

                grad_input[
                    :,
                    :,
                    i*self.pool_size:(i+1)*self.pool_size,
                    j*self.pool_size:(j+1)*self.pool_size
                ] += (

                    max_mask

                    *

                    grad_output[
                        :,
                        :,
                        i,
                        j
                    ][:,:,None,None]

                )

        return grad_input