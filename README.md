# Breast Cancer Data Analysis and Prediction

This project aims to predict whether breast cancer is **malignant** or **benign** using machine learning techniques. The analysis includes feature selection, hyperparameter tuning, and model training with an **Artificial Neural Network (ANN)**. Additionally, the project includes a **Streamlit app** for user interaction, where users can input data and view predictions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Breast cancer is one of the most common types of cancer, and early detection can significantly improve patient outcomes. This project uses the **Breast Cancer dataset** from `scikit-learn` to train an Artificial Neural Network (ANN) model that classifies tumors as malignant or benign. The project also includes a **Streamlit web app** to make predictions interactively.

---


## Features

- **Data Preprocessing**:
  - Used `SelectKBest` to select the most relevant features.
  - Scaled the data using `StandardScaler`.

- **Model Training**:
  - Tuned hyperparameters using `GridSearchCV`.
  - Trained an ANN using `MLPClassifier`.

- **Streamlit Web App**:
  - Interactive UI for entering feature values.
  - Displays predictions (malignant or benign) with probabilities.

---

## Setup Instructions

To set up and run the project locally, follow these steps:

### **1. Clone the Repository**

git clone https://github.com/jeffalmazan/Breast_Cancer_Data_Analysis.git
cd Breast_Cancer_Data_Analysis

### **2. Create venv**
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

### **3. Install Dependencies**

pip install -r requirements.txt

### **4. Running Streamlit App**
streamlit run app_st.py
```bash