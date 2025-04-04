
# Handwritten Digit Recognition

This project allows you to draw handwritten digits and predicts the digit using a Convolutional Neural Network (CNN) model trained on the MNIST dataset.

## Prerequisites

Before running the project, ensure you have the following installed:
cle
- Python 3.6 or higher
- `pip` package manager

## Required Libraries

To install the necessary libraries, run:

```bash
pip install tensorflow numpy opencv-python Pillow tk
```

## Project Files

- **`mnist_model.py`**: Trains and saves a CNN model using the MNIST dataset. If the model already exists, it will load the pre-trained model.
- **`drawing_app.py`**: Provides a GUI to draw digits using Tkinter. After drawing, it predicts the digit using the pre-trained model.

## Running the Project

### 1. Train the Model (if needed)
Run `mnist_model.py` to train the model using the MNIST dataset and save it as `mnist_cnn_model.h5`. If the model already exists, this step is skipped.

```bash
python mnist_model.py
```

### 2. Run the Drawing Application
After training or if the model is already trained, run the `drawing_app.py` to open the drawing interface where you can draw a digit and get predictions.

```bash
python drawing_app.py
```

### 3. Using the Drawing App
- **Draw** a digit on the white canvas using your mouse.
- **Click "Predict"** to see the model's prediction for the drawn digit.
- **Click "Clear"** to reset the canvas.

## Model Architecture

The model used for prediction is a simple **Convolutional Neural Network (CNN)** with the following layers:
- Conv2D (32 filters, 3x3 kernel)
- MaxPooling2D (2x2 pool size)
- Conv2D (64 filters, 3x3 kernel)
- MaxPooling2D (2x2 pool size)
- Flatten
- Dense (128 units, ReLU activation)
- Output layer (10 units, softmax activation)

The model is trained on the **MNIST dataset**, which contains 28x28 grayscale images of handwritten digits (0-9).

