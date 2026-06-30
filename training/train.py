from cnn.model import ScratchCNN
from training.dataloader import DataLoader
import numpy as np

print("Loading Dataset...")

loader = DataLoader("dataset")
X, y = loader.load()

print("Dataset Loaded Successfully!")
print("Images Shape :", X.shape)
print("Labels Shape :", y.shape)

print("\nCreating CNN...")
model = ScratchCNN()

print("CNN Created Successfully!")

epochs = 20

for epoch in range(epochs):

    loss, probabilities = model.forward(X, y)

    model.backward()

    predictions = np.argmax(probabilities, axis=1)

    labels = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == labels) * 100

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

print("\nTraining Finished!")

import os

os.makedirs("models", exist_ok=True)

model.save("models/weights.npz")

print("\nModel Saved Successfully!")