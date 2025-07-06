import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

# Load the trained model and feature names
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Title and description
st.title("Breast Cancer Prediction App")
st.markdown("""
This app uses a trained Random Forest model to predict whether a breast tumor is benign or malignant based on input features.
Enter the feature values using the sliders below, and the model will provide a prediction. Explore visualizations to understand the model and dataset.
""")

# Input section
st.header("Input Features")
st.write("Adjust the sliders to input feature values for the prediction.")

# Load dataset once for slider ranges
data = load_breast_cancer()

# Create sliders for each feature
input_data = []
for i, feature in enumerate(feature_names):
    min_val = float(np.min(data.data[:, i]))
    max_val = float(np.max(data.data[:, i]))
    mean_val = float(np.mean(data.data[:, i]))
    input_data.append(st.slider(f"{feature}", min_val, max_val, mean_val))

# Convert input to numpy array
input_data = np.array(input_data).reshape(1, -1)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    result = "Malignant" if prediction == 0 else "Benign"
    st.header("Prediction Result")
    st.write(f"The model predicts the tumor is **{result}**.")
    st.write(f"Probability of Benign: {prob[1]:.2f}")
    st.write(f"Probability of Malignant: {prob[0]:.2f}")

# Visualizations
st.header("Model and Data Insights")
st.subheader("Feature Importance")
st.write("This plot shows the top 10 most important features used by the model.")
feature_img = Image.open("feature_importance.png")
st.image(feature_img, caption="Feature Importance Plot")

st.subheader("Confusion Matrix")
st.write("This confusion matrix shows the model's performance on the test data.")
cm_img = Image.open("confusion_matrix.png")
st.image(cm_img, caption="Confusion Matrix")

st.subheader("Correlation Heatmap")
st.write("This heatmap shows correlations between features in the dataset.")
heatmap_img = Image.open("correlation_heatmap.png")
st.image(heatmap_img, caption="Correlation Heatmap")

st.subheader("Pair Plot of Top Features")
st.write("This pair plot shows relationships between the top 4 important features, colored by class (0: Malignant, 1: Benign).")
pair_img = Image.open("pair_plot.png")
st.image(pair_img, caption="Pair Plot of Top 4 Features")