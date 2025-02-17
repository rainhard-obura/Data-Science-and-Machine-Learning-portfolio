# Mapping Floods in South Africa

This is my solution to the mapping floods in South Africa challenge, hosted by Zindi.
the public score achieved was 0.0049

## Overview
This project aims to predict the occurrence of floods using a multimodal deep learning approach that combines time-series precipitation data and satellite imagery. The goal is to develop a robust model capable of determining whether a flood event occurred based on historical precipitation and composite cloudless images.

## Dataset
The dataset consists of the following files:
- **Train.csv**: Contains `event_id`, `precipitation`, and `label` (1 for flood occurrence, 0 otherwise).
- **Test.csv**: Contains `event_id` and `precipitation` (labels are unknown).
- **composite_images.npz**: Contains composite satellite images corresponding to different event locations.
- **sample_submission.csv**: A template for submitting predictions, with `event_id` and `label` columns.

## Approach
This is a multimodal task where we process two different data types:
1. **Time-Series Precipitation Data**: Processed using a **1D ResNet model** to capture temporal dependencies.
2. **Satellite Images**: Processed using a **ResNet50 model** to extract spatial features.

The extracted features from both models are concatenated and passed through a dense network for final classification.

## Data Preprocessing
- **Precipitation Data**: Normalized using `MinMaxScaler`.
- **Image Data**:
  - Normalized to approximate a mean of 0 and standard deviation of 1.
  - Slope data decoded from 16-bit unsigned integer representation to radians.
  - Stored in separate train, validation, and test sets to optimize memory usage.

## Model Architecture
- **1D ResNet for Time-Series Data**
  - Multiple convolutional layers with ReLU activation
  - Max-pooling layers to downsample features
  - Fully connected layers for feature extraction

- **ResNet50 for Image Data**
  - Pretrained model (initialized without weights)
  - Global Average Pooling layer to extract meaningful representations
  - Dense layers for final feature extraction

- **Fusion and Classification**
  - Concatenation of extracted features from both models
  - Fully connected layers with dropout for regularization
  - Final output layer using sigmoid activation for binary classification

## Training
- Loss function: **Binary Crossentropy**
- Optimizer: **Adam**
- Metrics: **Accuracy**
- Validation split: 10% of the training data
- Number of epochs: 20
- Batch size: 32

## Evaluation
The model is evaluated using:
- **Log Loss** for performance measurement
- **Accuracy** as an additional metric
- Training and validation loss/accuracy plots

## Predictions
- Predictions are made on the test set, producing probability scores for flood occurrence.
- The final output is saved as `submission.csv` in the required format for submission.

## Authors
This project was developed as part of a machine learning challenge hosted on Zindi. Contributions are welcome!

## License
This project is open-source and available under the MIT License.

