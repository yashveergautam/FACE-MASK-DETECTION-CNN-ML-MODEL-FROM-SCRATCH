import numpy as np

from cnn.model import ScratchCNN
from training.preprocess import ImagePreprocessor


class FaceMaskPredictor:

    def __init__(self):

        self.class_names = [
            "With Mask",
            "Without Mask"
        ]

        self.preprocessor = ImagePreprocessor()

        self.model = ScratchCNN(
            num_classes=2
        )

        self.model.load(
            "models/weights.npz"
        )

    def predict(self, image_path):

        print("STEP 1 : Reading Image")

        image = self.preprocessor.process(image_path)

        print("STEP 2 : Image Processed")

        image = image.reshape(1,1,28,28)

        print("STEP 3 : Running CNN")

        prediction, confidence = self.model.predict(image)

        print("STEP 4 : CNN Finished")

        return {

            "prediction": self.class_names[prediction[0]],

            "confidence": round(float(confidence[0])*100,2)

        }