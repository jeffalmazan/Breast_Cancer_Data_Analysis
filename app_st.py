import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
import joblib


# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Load the pre-trained ANN model
model = joblib.load("best_ann_model.pkl")  # Save your model as 'best_ann_model.pkl'

# Streamlit App
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether breast cancer is malignant or benign based on user inputs.")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.sidebar.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    return pd.DataFrame([input_data])

user_data = user_input_features()
st.write("### User Input Features", user_data)

# Predict and Display Results
if st.button("Predict"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)
    st.write("### Prediction:", "Malignant" if prediction[0] == 0 else "Benign")
    st.write("### Prediction Probability:", prediction_proba)
