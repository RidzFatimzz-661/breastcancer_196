# breastcancer_196
AI Health Advisor – Neural Network-Based Multi-Disease Prediction System

# Neural Network-Based Multi-Disease Prediction System

#Overview

This project implements a **Neural Network-Based Multi-Disease Prediction System** to predict the likelihood of multiple diseases using clinical and demographic data. The primary objective is to develop a robust prediction model utilizing various Artificial Neural Network (ANN) techniques, including perceptrons, multi-layer perceptrons (MLPs), backpropagation, activation functions, optimization techniques (SGD, Adam), regularization (L1, L2), dropout, and batch normalization.

#Dataset

The dataset used is the **Breast Cancer Dataset** from Kaggle, containing 30 clinical features from breast cancer cell nuclei and a binary diagnosis label (Malignant/Benign).

#Data Preprocessing

* Removed unnecessary columns (e.g., 'id').
* Split data into features (X) and labels (y).
* Scaled the features using `StandardScaler`.
* Split the dataset into training (80%) and testing (20%) sets.

#Model Description

#Implemented Models:

1. **Perceptron:**

   * Basic binary classification model with a single layer using a step activation function.
   * Uses the delta rule to update weights.

2. **MLP with SGD:**

   * Multi-layer Perceptron using **Stochastic Gradient Descent**.
   * Activation: ReLU.
   * Regularization: L1/L2 and dropout (20%).

3. **MLP with Adam:**

   * Multi-layer Perceptron using the **Adam optimizer**.
   * Activation: Tanh.
   * Adaptive learning rate and stable convergence.

# Key ANN Concepts:

Activation Functions: ReLU, sigmoid, tanh.
Backpropagation:Gradient descent for minimizing loss.
Optimization Techniques:SGD (with momentum) and Adam.
Regularization Methods:L1, L2, and dropout (20%).
Batch Normalization: Stabilizes training by normalizing activations.

#Architecture
#Perceptron Architecture:

Input layer: Scaled features.
Output layer:Binary classification (0 or 1).
Activation Function:Step function.

#MLP Architecture:

Input Layer:Scaled features.
Hidden Layer: 30 neurons with ReLU (SGD) or tanh (Adam).
Output Layer:Sigmoid activation for binary classification.
Regularization: L1, L2, dropout.
Optimization: SGD (momentum) and Adam.
Batch Normalization:Improves convergence stability.

#Results

#Accuracy Comparison:

Perceptron Accuracy: Displayed during training.
MLP with SGD (ReLU, L1/L2, Dropout) Accuracy:Improved due to dropout and regularization.
MLP with Adam (Tanh) Accuracy: Highest accuracy due to adaptive learning rate.

# Performance Analysis:

* The Adam optimizer significantly improves convergence speed and model accuracy.
* Regularization (L2, dropout) effectively reduces overfitting, especially in the SGD model.

# Future Enhancements:

* Extend to multi-class disease prediction.
* Experiment with more complex architectures, like Convolutional Neural Networks (CNNs).
* Integrate additional patient data for enhanced prediction accuracy.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/multi-disease-prediction.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:

   ```bash
   python train_model.py
   ```
4. Evaluate the model accuracy and compare the results.

## License

This project is licensed under the MIT License.

