import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction using ML")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("clean_dataset.csv")
    return data

data = load_data()

# Handle categorical column (city)
data = pd.get_dummies(data, columns=["city"], drop_first=True)

# Split features & target
X = data.drop("price", axis=1)
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter House Details")

# User inputs
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0.0)
sqft_living = st.number_input("Sqft Living", min_value=0)
sqft_lot = st.number_input("Sqft Lot", min_value=0)
floors = st.number_input("Floors", min_value=0.0)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View", 0, 4)
condition = st.slider("Condition", 1, 5)
sqft_above = st.number_input("Sqft Above", min_value=0)
sqft_basement = st.number_input("Sqft Basement", min_value=0)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2025)
yr_renovated = st.number_input("Year Renovated", min_value=0)
city = st.selectbox("City", data.filter(like="city_").columns)

# Create input dataframe
input_dict = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated
}

input_df = pd.DataFrame([input_dict])

# Add city columns (one-hot)
for col in X.columns:
    if col.startswith("city_"):
        input_df[col] = 1 if col == city else 0

# Match column order
input_df = input_df[X.columns]

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)}")
