import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import Pool

# --- Page Config ---
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App")

# --- Load model and scaler ---
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# --- Feature Definitions ---
numeric_features = [
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
    'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Delay_from_due_date'
]

categorical_features = ['Occupation', 'Income_Category', 'Age_Category', 'Spending_Level']
all_features = numeric_features + categorical_features

# --- Label Mapping ---
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# --- Categorical Options ---
occupation_options = [
    'Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer',
    'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
    'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer'
]
income_options = ['High Income', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']
age_options = ['Adults', 'Middle-Aged Adults', 'Older Adults', 'Teenagers', 'Young Adults']
spending_options = ['High', 'Low']

# --- Authentication ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# --- Logout Button ---
st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
st.sidebar.header("📥 Input Method")
input_mode = st.sidebar.radio("Choose how to input data:", ["Manual Entry", "Upload CSV"])

# --- Manual Input ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        input_data = {
            'Outstanding_Debt': st.number_input("Outstanding Debt", value=0.0),
            'Monthly_Inhand_Salary': st.number_input("Monthly Inhand Salary", value=0.0),
            'Total_EMI_per_month': st.number_input("Total EMI per Month", value=0.0),
            'Credit_Utilization_Ratio': st.number_input("Credit Utilization Ratio", value=0.0),
            'Credit_History_Age_Months': st.number_input("Credit History Age (Months)", value=0.0),
            'Delay_from_due_date': st.number_input("Delay from Due Date", value=0.0),
            'Occupation': st.selectbox("Occupation", occupation_options),
            'Income_Category': st.selectbox("Income Category", income_options),
            'Age_Category': st.selectbox("Age Category", age_options),
            'Spending_Level': st.selectbox("Spending Level", spending_options),
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        pool = Pool(input_df[all_features], cat_features=categorical_features)
        prediction = model.predict(pool)[0]
        label = label_map.get(int(prediction), "Unknown")
        st.success(f"🎯 Predicted Category: **{label}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if not all(col in df.columns for col in all_features):
                st.error(f"❌ CSV must contain: {', '.join(all_features)}")
            else:
                df[numeric_features] = scaler.transform(df[numeric_features])
                pool = Pool(df[all_features], cat_features=categorical_features)
                predictions = model.predict(pool)
                df["Predicted Category"] = [label_map.get(int(p), "Unknown") for p in predictions]
                st.success("✅ Predictions generated!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
