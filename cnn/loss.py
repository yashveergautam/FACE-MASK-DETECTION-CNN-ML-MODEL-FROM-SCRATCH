import numpy as np


class CrossEntropyLoss:

    def forward(
        self,
        predictions,
        targets
    ):

        self.predictions = predictions
        self.targets = targets

        loss = -np.sum(

            targets *

            np.log(
                predictions + 1e-15
            )

        )

        return loss / predictions.shape[0]

    def backward(self):

        return (

            self.predictions

            -

            self.targets

        )