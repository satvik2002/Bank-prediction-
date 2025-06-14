import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import Pool

# --- Streamlit Page Config ---
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App (CatBoost)")

# --- Load Model & Scaler ---
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# --- Feature Lists ---
numeric_features = [
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
    'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Delay_from_due_date'
]

categorical_features = [
    'Occupation', 'Income_Category', 'Age_Category', 'Spending_Level'
]

all_features = numeric_features + categorical_features

# --- Category Options ---
occupation_options = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer',
                      'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
                      'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer']

income_options = ['High Income', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']
age_options = ['Adults', 'Middle-Aged Adults', 'Older Adults', 'Teenagers', 'Young Adults']
spending_options = ['High', 'Low']

label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# --- Input Method Selection ---
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

# --- Manual Input Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        user_input = {}
        for col in numeric_features:
            user_input[col] = st.number_input(f"{col}", value=0.0)

        user_input['Occupation'] = st.selectbox("Occupation", occupation_options)
        user_input['Income_Category'] = st.selectbox("Income Category", income_options)
        user_input['Age_Category'] = st.selectbox("Age Category", age_options)
        user_input['Spending_Level'] = st.selectbox("Spending Level", spending_options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])

        # Scale numeric features only
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Make prediction with Pool
        pool = Pool(data=input_df, cat_features=categorical_features)
        prediction = model.predict(pool)[0]

        label = label_map.get(int(prediction), "Unknown")
        st.success(f"🎯 Predicted Category: **{label}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if not all(col in df.columns for col in all_features):
                st.error(f"❌ CSV must contain all required columns:\n{', '.join(all_features)}")
            else:
                # Scale numeric features
                df[numeric_features] = scaler.transform(df[numeric_features])

                pool = Pool(data=df[all_features], cat_features=categorical_features)
                preds = model.predict(pool)
                df["Predicted Category"] = [label_map.get(int(p), "Unknown") for p in preds]

                st.success("✅ Prediction complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predicted_categories.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
