# CloudExify Project 4: Disease Prediction

Binary classification project predicting patient disease status based on clinical indicators using Logistic Regression and Decision Tree models.

## Overview

This project implements an end-to-end machine learning pipeline for medical risk prediction. It covers exploratory analysis, handling class imbalance, model training, evaluation metrics, and inference on new patient data.

## Features & Implementation

* **Data Preprocessing:** Cleans missing values and applies standard feature scaling (`StandardScaler`).
* **Class Imbalance Handling:** Uses `SMOTE` (Synthetic Minority Over-sampling Technique) for Logistic Regression and `class_weight='balanced'` for the Decision Tree.
* **Models Evaluated:**
  * Logistic Regression (with optimized solver iterations and class weighting)
  * Decision Tree Classifier (`max_depth=10`)
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC curves.
* **Feature Importance:** Visualizes top predictive features (led by Glucose and Age).
* **Inference:** Includes a script section to predict risk probabilities for new patient records.

## Project Structure

```text
├── CloudExify-Project4.py    # Main training and evaluation script
├── disease_data.csv  # Processed dataset
└── README.md                 # Project documentation
