import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Tractor Rent Predictor", page_icon="🚜", layout="wide")

st.title("🚜 Tractor Rent Prediction System")
st.write("Predict tractor rent based on tractor type, working hours and distance.")

data = {
    "Tractor": ["Mahindra", "Swaraj", "John Deere", "Mahindra", "Swaraj", "John Deere"],
    "Working_Hours": [2, 3, 4, 5, 6, 7],
    "Distance": [5, 8, 10, 12, 15, 18],
    "Rent": [500, 700, 900, 1100, 1300, 1500]
}

df = pd.DataFrame(data)

df["Tractor_Code"] = df["Tractor"].astype("category").cat.codes

X = df[["Tractor_Code", "Working_Hours", "Distance"]]
y = df["Rent"]

model = LinearRegression()
model.fit(X, y)

st.sidebar.header("Enter Tractor Details")

tractor_name = st.sidebar.selectbox(
    "Select Tractor",
    ["Mahindra", "Swaraj", "John Deere"]
)

tractor_code = {"Mahindra":0, "Swaraj":1, "John Deere":2}[tractor_name]

working_hours = st.sidebar.slider("Working Hours", 1, 12, 3)

distance = st.sidebar.slider("Distance (km)", 1, 30, 5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset")
    st.dataframe(df)

    st.subheader("📈 Rent Chart")
    st.line_chart(df["Rent"])

with col2:
    st.subheader("🔮 Prediction")

    if st.button("Predict Rent"):
        prediction = model.predict([[tractor_code, working_hours, distance]])

        st.success(f"💰 Estimated Rent: ₹ {prediction[0]:.2f}")

        st.balloons()

st.markdown("---")