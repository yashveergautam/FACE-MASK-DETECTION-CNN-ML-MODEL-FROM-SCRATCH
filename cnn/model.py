import numpy as np

from cnn.conv import Conv2D
from cnn.relu import ReLU
from cnn.pooling import MaxPooling2D
from cnn.dense import Dense
from cnn.softmax import Softmax
from cnn.loss import CrossEntropyLoss


class ScratchCNN:

    def __init__(self,
                 learning_rate=0.01,
                 num_classes=2):

        self.conv = Conv2D(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            learning_rate=learning_rate
        )

        self.relu = ReLU()

        self.pool = MaxPooling2D(
            pool_size=2
        )

        self.dense = Dense(
            input_features=4 * 13 * 13,
            output_features=num_classes,
            learning_rate=learning_rate
        )

        self.softmax = Softmax()

        self.loss = CrossEntropyLoss()

    def forward(self, x, y):

        out = self.conv.forward(x)

        out = self.relu.forward(out)

        out = self.pool.forward(out)

        out = self.dense.forward(out)

        probabilities = self.softmax.forward(out)

        loss = self.loss.forward(
            probabilities,
            y
        )

        return loss, probabilities

    def backward(self):

        grad = self.loss.backward()

        grad = self.dense.backward(grad)

        grad = self.pool.backward(grad)

        grad = self.relu.backward(grad)

        self.conv.backward(grad)

    def predict(self, x):

        out = self.conv.forward(x)

        out = self.relu.forward(out)

        out = self.pool.forward(out)

        out = self.dense.forward(out)

        probabilities = self.softmax.forward(out)

        prediction = np.argmax(
            probabilities,
            axis=1
        )

        confidence = np.max(
            probabilities,
            axis=1
        )

        return prediction, confidence

    def save(self, path):

        np.savez(

            path,

            conv_weights=self.conv.weights,
            conv_biases=self.conv.biases,

            dense_weights=self.dense.weights,
            dense_biases=self.dense.biases

        )

    def load(self, path):

        data = np.load(path)

        self.conv.weights = data["conv_weights"]
        self.conv.biases = data["conv_biases"]

        self.dense.weights = data["dense_weights"]
        self.dense.biases = data["dense_biases"]