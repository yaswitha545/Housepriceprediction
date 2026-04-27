import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_data.csv")

# Drop missing values
df = df.dropna()

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# One-hot encoding for city (VERY IMPORTANT)
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Target
y = df['price']

# Features (remove target)
X = df.drop('price', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🏠 House Price Prediction")

bedrooms = st.number_input("Bedrooms", step=1)
bathrooms = st.number_input("Bathrooms", step=1)
sqft_living = st.number_input("Sqft Living")
sqft_lot = st.number_input("Sqft Lot")
floors = st.number_input("Floors")
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.number_input("View", step=1)
condition = st.number_input("Condition", step=1)
sqft_above = st.number_input("Sqft Above")
sqft_basement = st.number_input("Sqft Basement")
yr_built = st.number_input("Year Built", step=1)
yr_renovated = st.number_input("Year Renovated", step=1)
year = st.number_input("Year", step=1)
month = st.number_input("Month", step=1)

# city handling (must match training encoding)
city_options = [col.replace("city_", "") for col in X.columns if "city_" in col]
city = st.selectbox("City", city_options)

if st.button("Predict Price"):

    # Create input row with all columns
    input_dict = {col: 0 for col in X.columns}

    input_dict["bedrooms"] = bedrooms
    input_dict["bathrooms"] = bathrooms
    input_dict["sqft_living"] = sqft_living
    input_dict["sqft_lot"] = sqft_lot
    input_dict["floors"] = floors
    input_dict["waterfront"] = waterfront
    input_dict["view"] = view
    input_dict["condition"] = condition
    input_dict["sqft_above"] = sqft_above
    input_dict["sqft_basement"] = sqft_basement
    input_dict["yr_built"] = yr_built
    input_dict["yr_renovated"] = yr_renovated
    input_dict["year"] = year
    input_dict["month"] = month

    # set city encoding
    city_col = f"city_{city}"
    if city_col in input_dict:
        input_dict[city_col] = 1

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")
