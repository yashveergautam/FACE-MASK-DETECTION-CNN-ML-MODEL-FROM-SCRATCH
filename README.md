# 🧠 Scratch CNN Face Mask Detection

A complete **Face Mask Detection System** built using a **Convolutional Neural Network (CNN) implemented entirely from scratch with NumPy**. This project demonstrates the complete workflow of deep learning, from image preprocessing and CNN implementation to model deployment through a web application.

> **Note:** The CNN model is implemented without using TensorFlow, Keras, PyTorch, or any deep learning framework. Only NumPy is used for all neural network computations.

---

# 📌 Project Overview

The goal of this project is to understand how a Convolutional Neural Network works internally instead of relying on high-level deep learning libraries.

The project performs the following tasks:

* Load and preprocess face mask images
* Convert RGB images to grayscale
* Resize images to **28 × 28**
* Normalize pixel values
* Train a CNN implemented completely from scratch
* Save trained model weights
* Load the trained model for inference
* Deploy the model using Flask
* Provide a web interface for image upload and prediction

---

# 🚀 Features

* CNN implemented from scratch using NumPy
* Manual implementation of forward propagation
* Manual implementation of backpropagation
* Custom Convolution Layer
* ReLU Activation
* Max Pooling Layer
* Dense (Fully Connected) Layer
* Softmax Classifier
* Cross Entropy Loss
* Image preprocessing pipeline
* Flask REST API
* HTML/CSS/JavaScript frontend
* Real-time image prediction
* Confidence score display

---

# 🛠 Technologies Used

### Programming Language

* Python 3

### Machine Learning

* NumPy

### Image Processing

* Matplotlib (Image Loading)

### Backend

* Flask
* Flask-CORS

### Frontend

* HTML
* CSS
* JavaScript

---

# ❌ Deep Learning Libraries Not Used

This project intentionally avoids using any deep learning framework.

* TensorFlow ❌
* Keras ❌
* PyTorch ❌
* OpenCV DNN ❌
* Scikit-Learn Neural Networks ❌

---

# 📂 Project Structure

```text
Scratch-CNN-FaceMask/

│
├── backend/
│   ├── predictor.py
│   └── server.py
│
├── cnn/
│   ├── conv.py
│   ├── dense.py
│   ├── pooling.py
│   ├── relu.py
│   ├── softmax.py
│   ├── loss.py
│   └── model.py
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   └── weights.npz
│
├── training/
│   ├── preprocess.py
│   ├── dataloader.py
│   └── train.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 🧠 CNN Architecture

```
Input Image (28×28×1)
        │
        ▼
Convolution Layer
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling
        │
        ▼
Flatten
        │
        ▼
Dense Layer
        │
        ▼
Softmax
        │
        ▼
Prediction
```

---

# 📊 Data Preprocessing

The preprocessing pipeline includes:

* Reading the input image
* RGB to Grayscale conversion
* Image resizing to **28 × 28**
* Pixel normalization (0–255 → 0–1)
* Conversion into CNN input tensor

---

# ⚙️ Training Process

The model is trained using the following pipeline:

1. Load Dataset
2. Preprocess Images
3. Forward Propagation
4. Compute Cross Entropy Loss
5. Backpropagation
6. Gradient Descent Weight Update
7. Repeat for Multiple Epochs
8. Save Trained Weights

---

# 🌐 Deployment Pipeline

```
User Uploads Image
        │
        ▼
Frontend (HTML/CSS/JavaScript)
        │
        ▼
Flask Backend
        │
        ▼
Image Preprocessing
        │
        ▼
Scratch CNN
        │
        ▼
Prediction
        │
        ▼
JSON Response
        │
        ▼
Browser Displays Result
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/scratch-cnn-face-mask.git

cd scratch-cnn-face-mask
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

```bash
python -m training.train
```

The trained weights will be saved in:

```
models/weights.npz
```

---

# 🚀 Run the Application

```bash
python -m backend.server
```

Open your browser:

```
http://127.0.0.1:5000
```

Upload an image and click **Predict**.

---

# 📈 Current Model

* Binary Classification
* Classes:

  * With Mask
  * Without Mask
* Image Size: **28 × 28**
* Activation: ReLU
* Loss Function: Cross Entropy
* Optimizer: Gradient Descent

---

# 🔮 Future Improvements

* Increase dataset size
* Improve model accuracy
* Mini-batch Gradient Descent
* Data Augmentation
* Multiple Convolution Layers
* Batch Normalization
* Dropout
* Real-Time Webcam Detection
* Docker Deployment
* Cloud Deployment (AWS/Azure/GCP)

---

# 🎯 Learning Outcomes

This project demonstrates practical understanding of:

* Computer Vision
* Deep Learning
* Convolutional Neural Networks
* Image Processing
* Matrix Operations
* Gradient Descent
* Backpropagation
* REST API Development
* Model Deployment
* Full-Stack Machine Learning Applications

---

# 📷 Demo

1. Upload an image.
2. Click **Predict**.
3. The application displays:

   * Predicted Class
   * Confidence Score

---

# 👨‍💻 Author

**Yashveer**

Bachelor of Technology (Computer Science & Engineering)

Machine Learning Enthusiast | Python Developer | Deep Learning Learner

---

# 📄 License

This project is created for educational and learning purposes. Feel free to use, modify, and extend it for academic projects while providing appropriate credit.
