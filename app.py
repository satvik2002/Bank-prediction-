import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up Streamlit page
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App (CatBoost)")

# Load the CatBoost model and scaler
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# Define feature names (must match training)
feature_cols = [
    'Outstanding_Debt',
    'Monthly_Inhand_Salary',
    'Total_EMI_per_month',
    'Credit_Utilization_Ratio',
    'Credit_History_Age_Months',
    'Delay_from_due_date',
    'Occupation',
    'Income_Category',
    'Age_Category',
    'Spending_Level'
]

# Mapping predicted class to label
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Sidebar input method
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Entry ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        user_input = {}
        for col in feature_cols:
            user_input[col] = st.number_input(f"{col}", value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(input_df)
        pred = model.predict(scaled_input)[0]
        label = label_map.get(pred, "Unknown")
        st.success(f"🎯 Predicted Category: **{label}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if not all(col in df.columns for col in feature_cols):
                st.error(f"❌ CSV must contain the following columns: {', '.join(feature_cols)}")
            else:
                X = df[feature_cols].fillna(0)
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                df["Predicted Category"] = [label_map.get(p, "Unknown") for p in preds]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predicted_categories.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
