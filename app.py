import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------- Page Config -------------------
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App (CatBoost)")

# ------------------- Login Management -------------------
# Dummy credentials
AUTH_USERS = {"admin": "admin123", "user": "user123"}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Login form
if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in AUTH_USERS and AUTH_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password")
    st.stop()

# Logout button
st.sidebar.markdown(f"**👤 Logged in as:** `{st.session_state.username}`")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()

# ------------------- Load Model and Scaler -------------------
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# ------------------- Feature Columns -------------------
feature_cols = [
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
    'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Delay_from_due_date',
    'Occupation', 'Income_Category', 'Age_Category', 'Spending_Level'
]

# Mappings for categorical variables
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
spending_map = {'High': 0, 'Low': 1}

label_map = {
    0: "Established Customer", 1: "Growing Customer", 2: "Legacy Customer",
    3: "Loyal Customer", 4: "New Customer"
}

# ------------------- Input Mode -------------------
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# ------------------- Manual Entry -------------------
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        user_input = {
            'Outstanding_Debt': st.number_input("Outstanding Debt", value=0.0),
            'Monthly_Inhand_Salary': st.number_input("Monthly Inhand Salary", value=0.0),
            'Total_EMI_per_month': st.number_input("Total EMI per Month", value=0.0),
            'Credit_Utilization_Ratio': st.number_input("Credit Utilization Ratio", value=0.0),
            'Credit_History_Age_Months': st.number_input("Credit History Age (Months)", value=0.0),
            'Delay_from_due_date': st.number_input("Delay from Due Date", value=0.0),
            'Occupation': st.selectbox("Occupation", list(occupation_map.keys())),
            'Income_Category': st.selectbox("Income Category", list(income_map.keys())),
            'Age_Category': st.selectbox("Age Category", list(age_map.keys())),
            'Spending_Level': st.selectbox("Spending Level", list(spending_map.keys()))
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        input_df['Occupation'] = input_df['Occupation'].map(occupation_map)
        input_df['Income_Category'] = input_df['Income_Category'].map(income_map)
        input_df['Age_Category'] = input_df['Age_Category'].map(age_map)
        input_df['Spending_Level'] = input_df['Spending_Level'].map(spending_map)

        numeric_cols = [col for col in input_df.columns if col not in ['Occupation', 'Income_Category', 'Age_Category', 'Spending_Level']]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        pred = model.predict(input_df)[0]
        label = label_map.get(pred, "Unknown")
        st.success(f"🎯 Predicted Category: **{label}**")

# ------------------- CSV Upload -------------------
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(col in df.columns for col in feature_cols):
                st.error(f"❌ CSV must include these columns: {', '.join(feature_cols)}")
            else:
                df['Occupation'] = df['Occupation'].map(occupation_map)
                df['Income_Category'] = df['Income_Category'].map(income_map)
                df['Age_Category'] = df['Age_Category'].map(age_map)
                df['Spending_Level'] = df['Spending_Level'].map(spending_map)

                df = df.fillna(0)
                numeric_cols = [col for col in feature_cols if col not in ['Occupation', 'Income_Category', 'Age_Category', 'Spending_Level']]
                df[numeric_cols] = scaler.transform(df[numeric_cols])

                preds = model.predict(df[feature_cols])
                df['Predicted Category'] = [label_map.get(p, "Unknown") for p in preds]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error while processing the file: {e}")
