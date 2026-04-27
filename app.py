import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction using ML")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("clean_dataset.csv")
    return data

data = load_data()

st.subheader("Dataset Preview")
st.write(data.head())

# Split features & target
X = data.drop("price", axis=1)   # change 'price' if your target column name is different
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter House Details")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=0.0)

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Estimated House Price: ₹ {round(prediction[0], 2)}")
