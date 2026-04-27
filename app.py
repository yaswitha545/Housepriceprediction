import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset safely
df = pd.read_csv("data.csv")

# Drop missing values
df = df.dropna()

# ---------------------------
# IMPORTANT FIX (convert ALL categorical columns)
# ---------------------------

# Convert city using one-hot encoding
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Ensure ALL remaining columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any rows still containing NaN after conversion
df = df.dropna()

# ---------------------------
# SPLIT DATA
# ---------------------------

y = df['price']
X = df.drop('price', axis=1)

# Convert again to be 100% safe
X = X.astype(float)
y = y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# MODEL
# ---------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("🏠 House Price Prediction")

# numeric inputs only (simple version to avoid errors)
bedrooms = st.number_input("Bedrooms", step=1)
bathrooms = st.number_input("Bathrooms", step=1)
sqft_living = st.number_input("Sqft Living")
sqft_lot = st.number_input("Sqft Lot")
floors = st.number_input("Floors")
waterfront = st.number_input("Waterfront (0 or 1)", step=1)
view = st.number_input("View", step=1)
condition = st.number_input("Condition", step=1)
sqft_above = st.number_input("Sqft Above")
sqft_basement = st.number_input("Sqft Basement")
yr_built = st.number_input("Year Built", step=1)
yr_renovated = st.number_input("Year Renovated", step=1)
year = st.number_input("Year", step=1)
month = st.number_input("Month", step=1)

if st.button("Predict Price"):

    # Build input row with correct columns
    input_data = {col: 0 for col in X.columns}

    input_data["bedrooms"] = bedrooms
    input_data["bathrooms"] = bathrooms
    input_data["sqft_living"] = sqft_living
    input_data["sqft_lot"] = sqft_lot
    input_data["floors"] = floors
    input_data["waterfront"] = waterfront
    input_data["view"] = view
    input_data["condition"] = condition
    input_data["sqft_above"] = sqft_above
    input_data["sqft_basement"] = sqft_basement
    input_data["yr_built"] = yr_built
    input_data["yr_renovated"] = yr_renovated
    input_data["year"] = year
    input_data["month"] = month

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")
