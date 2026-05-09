# CTC-CRNN License Plate Recognition using Optimized AdamW

A deep learning-based Automatic License Plate Recognition (ALPR) system built using a CTC-CRNN architecture with a custom optimized AdamW_XT optimizer.
The model performs end-to-end license plate text recognition directly from cropped plate images without character segmentation.

## Project Overview

This project implements a complete ALPR pipeline consisting of:

* License plate extraction from annotated images
* Image preprocessing and augmentation
* CRNN (CNN + BiLSTM) architecture
* CTC Loss for sequence prediction
*  Custom AdamW_XT optimizer
* Model evaluation using OCR metrics
* Comparison with EasyOCR

The system is designed to recognize alphanumeric license plate sequences efficiently even under varying lighting, angle, and noise conditions.

## Model Architecture

The recognition model follows a CTC-CRNN pipeline:

### CNN Feature Extractor
* Multiple Conv2D layers
* Batch Normalization
* MaxPooling
* Spatial Dropout
### Sequence Modeling
* Feature maps reshaped into sequences
* Bidirectional LSTM for temporal learning
### Prediction Layer
* Dense Softmax output
* CTC decoding for sequence transcription

## Custom Optimizer — AdamW_XT

This project introduces a modified optimizer named AdamW_XT featuring:

* Dynamic Weight Decay
* Trust Factor Gradient Scaling
* Gradient Clipping
* Stable convergence for OCR training

The optimizer was designed to improve:

* Training stability
* Generalization
* Character prediction accuracy

## Technologies Used
* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* EasyOCR
* Levenshtein Distance
* Matplotlib

## Evaluation Metrics

The model is evaluated using:

* Character Error Rate (CER)
* Exact Match Accuracy
* Levenshtein Distance
* OCR comparison against EasyOCR
