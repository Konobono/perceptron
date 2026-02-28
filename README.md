# Perceptron Binary Classifier

##  Project Overview

This repository contains a Python implementation of a **Perceptron-based binary classifier**, a project that I made for a Computer Science class — one of the fundamental algorithms in machine learning.  
The purpose of the project is to demonstrate how a simple neural classifier can be implemented from scratch using only basic Python libraries (`numpy`, `pandas`, and `csv`), and applied to a real-world binary classification problem.

The classifier learns to distinguish between two classes within a dataset (e.g., authentic vs. forged banknotes) by computing a linear decision boundary. It does not rely on external machine learning frameworks such as scikit-learn; instead, all logic is implemented manually to show understanding of the core algorithm.

##  Dataset Description

The dataset used in this project is `data_banknote_authentication.csv`, which contains numerical features extracted from banknote images. Each row corresponds to one example of a banknote, with the final column representing the ground truth class label:

- `0` or `1`: Binary class indicators (e.g., authentic vs. forged).
- Numerical feature columns are used for model input.

The dataset is used as both training and evaluation data for the Perceptron algorithm.

##  Requirements

Before running the project, make sure you have the necessary Python packages installed:

```bash
pip install numpy pandas
