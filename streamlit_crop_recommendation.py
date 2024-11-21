
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("Crop_recommendation.csv")
    return data

# Load data
data = load_data()

# Streamlit app title
st.title("Crop Recommendation System")

# Data Overview
st.header("Dataset Overview")
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Splitting the data into features and target
X = data.iloc[:, :-1]  # Features (e.g., soil and environmental conditions)
y = data.iloc[:, -1]   # Target (Crop label)

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# User Input for prediction
st.header("Predict the Suitable Crop")
N = st.number_input("Nitrogen Content", min_value=0.0, max_value=100.0, step=1.0)
P = st.number_input("Phosphorus Content", min_value=0.0, max_value=100.0, step=1.0)
K = st.number_input("Potassium Content", min_value=0.0, max_value=100.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, step=1.0)

# Make predictions
if st.button("Recommend Crop"):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_features)
    st.success(f"The recommended crop is: {prediction[0]}")

# Model Evaluation
st.header("Model Accuracy")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy of the model: {accuracy * 100:.2f}%")
