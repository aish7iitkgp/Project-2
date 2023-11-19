# Project 2 - CNN Image Classification

## Overview
This project focuses on building and training a Convolutional Neural Network (CNN) for the classification of images into two categories: benign and malignant. The main objective is to utilize deep learning techniques for accurate image classification.

## File Structure
For details on the required file structure for running this code, please refer to the specific instructions provided in the accompanying README file.

## Project Workflow

1. **Loading Images**: Functions to load images from benign and malignant directories.
2. **Train-Test Split**: Utilizing scikit-learn's `train_test_split` to divide the dataset into training and validation sets.
3. **Data Augmentation**: Applying `ImageDataGenerator` from Keras for real-time data augmentation and normalization.
4. **Building the CNN Model**: The model includes convolutional, max pooling, batch normalization, flattening, dense, and dropout layers.
5. **Model Compilation**: Compiling the model using the Adam optimizer and binary cross-entropy as the loss function.
6. **Training the Model**: Conducting the training over 10 epochs using the prepared data generators.
7. **Post-Processing**: Including functions for converting model output probabilities to labels and for loading and predicting on test images.

## Improvements and Troubleshooting

- **Irregular Loss/Accuracy Curves**:
  - Augment the dataset or acquire more data.
  - Experiment with different batch sizes.
  - Adjust the learning rate.
  - Implement early stopping.
  - Add L1/L2 regularization to dense layers.


## Prerequisites
- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- OpenCV (for image processing)
- NumPy

## Setup Instructions
1. **Data Preparation**: Unzip the `FNA` folder that contains the benign and malignant data.
2. **Install Dependencies**: Execute the following command to install necessary Python packages:
   ```bash
   pip install tensorflow keras scikit-learn opencv-python numpy
