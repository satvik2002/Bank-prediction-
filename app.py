import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Set page config first
st.set_page_config(page_title="Customer Category Predictor", layout="centered")

# --- Login Setup ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# --- Logout ---
st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# --- App Title ---
st.title("🏦 Customer Category Prediction App (CatBoost)")

# Load model and scaler
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# Feature columns
numeric_features = [
    'Outstanding_Debt',
    'Monthly_Inhand_Salary',
    'Total_EMI_per_month',
    'Credit_Utilization_Ratio',
    'Credit_History_Age_Months',
    'Delay_from_due_date'
]
categorical_features = [
    'Occupation',
    'Income_Category',
    'Age_Category',
    'Spending_Level'
]
all_features = numeric_features + categorical_features

# Categorical options
occupation_map = {
    'Accountant': 0, 'Architect': 1, 'Developer': 2, 'Doctor': 3, 'Engineer': 4,
    'Entrepreneur': 5, 'Journalist': 6, 'Lawyer': 7, 'Manager': 8, 'Mechanic': 9,
    'Media_Manager': 10, 'Musician': 11, 'Scientist': 12, 'Teacher': 13, 'Writer': 14
}
income_map = {
    'High Income': 0, 'Low Income': 1, 'Lower Middle Income': 2, 'Upper Middle Income': 3
}
age_map = {
    'Adults': 0, 'Middle-Aged Adults': 1, 'Older Adults': 2, 'Teenagers': 3, 'Young Adults': 4
}
spending_map = {
    'High': 0, 'Low': 1
}
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# --- Sidebar Input Mode ---
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Entry Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        inputs = {col: st.number_input(col, value=0.0) for col in numeric_features}
        inputs['Occupation'] = st.selectbox("Occupation", list(occupation_map.keys()))
        inputs['Income_Category'] = st.selectbox("Income Category", list(income_map.keys()))
        inputs['Age_Category'] = st.selectbox("Age Category", list(age_map.keys()))
        inputs['Spending_Level'] = st.selectbox("Spending Level", list(spending_map.keys()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        df = pd.DataFrame([inputs])

        # Map categorical values
        df['Occupation'] = df['Occupation'].map(occupation_map)
        df['Income_Category'] = df['Income_Category'].map(income_map)
        df['Age_Category'] = df['Age_Category'].map(age_map)
        df['Spending_Level'] = df['Spending_Level'].map(spending_map)

        # Scale numeric values only
        df_scaled = df.copy()
        df_scaled[numeric_features] = scaler.transform(df[numeric_features])

        # Predict
        pred = model.predict(df_scaled)[0]
        label = label_map.get(int(pred), "Unknown")
        st.success(f"🎯 Predicted Category: *{label}*")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Check if all required columns exist
            missing = [col for col in all_features if col not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                # Map categorical values
                df['Occupation'] = df['Occupation'].map(occupation_map)
                df['Income_Category'] = df['Income_Category'].map(income_map)
                df['Age_Category'] = df['Age_Category'].map(age_map)
                df['Spending_Level'] = df['Spending_Level'].map(spending_map)

                # Fill NA and scale numeric columns
                df = df.fillna(0)
                df_scaled = df.copy()
                df_scaled[numeric_features] = scaler.transform(df[numeric_features])

                # Predict
                preds = model.predict(df_scaled)
                df['Predicted Category'] = [label_map.get(int(p), "Unknown") for p in preds]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠ Error: {e}")
