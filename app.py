import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("clean_dataset.csv")

# Train model
X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# Title
st.title("🏠 House Price Prediction using ML")

# User Inputs
area = st.number_input("Area (sqft)", 500, 10000, 1500)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 5.0, 2.0)
floors = st.number_input("Floors", 1.0, 3.0, 1.0)
condition = st.slider("Condition (1-5)", 1, 5, 3)
yr_built = st.number_input("Year Built", 1900, 2025, 2000)
yr_renovated = st.number_input("Year Renovated", 0, 2025, 0)

sqft_living = area
sqft_lot = st.number_input("Lot Size", 1000, 50000, 5000)
sqft_above = st.number_input("Sqft Above", 500, 5000, 1500)
sqft_basement = st.number_input("Basement Sqft", 0, 3000, 0)
view = st.slider("View (0-4)", 0, 4, 0)
waterfront = st.selectbox("Waterfront", [0, 1])
year = st.number_input("Year Sold", 2000, 2025, 2014)
month = st.slider("Month Sold", 1, 12, 5)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[
        bedrooms, bathrooms, sqft_living, sqft_lot,
        floors, waterfront, view, condition,
        sqft_above, sqft_basement, yr_built,
        yr_renovated, year, month
    ]])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Price: ₹ {int(prediction[0]):,}")