import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #eef2f3, #dfe9f3);
    }
    .title {
        font-size:40px;
        font-weight:700;
        color:#2c3e50;
    }
    .card {
        background-color:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.1);
    }
    .price-box {
        background: linear-gradient(to right, #11998e, #38ef7d);
        padding:25px;
        border-radius:15px;
        color:white;
        font-size:28px;
        text-align:center;
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🏡 House Price Prediction App</div>', unsafe_allow_html=True)
st.write("Enter property details to estimate the market price")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("clean_dataset.csv")
    return data

data = load_data()

# Encode city
data = pd.get_dummies(data, columns=["city"], drop_first=True)

X = data.drop("price", axis=1)
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar input (clean UI)
st.sidebar.header("🏠 Enter Property Details")

bedrooms = st.sidebar.number_input("Bedrooms", 0, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 0.0, 10.0, 2.0)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 500, 10000, 1500)
sqft_lot = st.sidebar.number_input("Lot Size (sqft)", 500, 20000, 5000)
floors = st.sidebar.number_input("Floors", 1.0, 5.0, 1.0)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view = st.sidebar.slider("View Rating", 0, 4, 1)
condition = st.sidebar.slider("Condition", 1, 5, 3)
sqft_above = st.sidebar.number_input("Sqft Above", 500, 10000, 1200)
sqft_basement = st.sidebar.number_input("Sqft Basement", 0, 3000, 0)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
yr_renovated = st.sidebar.number_input("Year Renovated", 0, 2025, 0)

city = st.sidebar.selectbox("City", data.filter(like="city_").columns)

# Prepare input
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

# Handle city encoding
for col in X.columns:
    if col.startswith("city_"):
        input_df[col] = 1 if col == city else 0

input_df = input_df[X.columns]

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Property Overview")
    st.write(input_df)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.button("🔍 Predict Price"):
        prediction = model.predict(input_df)[0]

        st.markdown(
            f'<div class="price-box">💰 Estimated Price <br> ₹ {round(prediction, 2)}</div>',
            unsafe_allow_html=True
        )
