import streamlit as st
import pandas as pd
import joblib
from catboost import Pool

# Set up Streamlit app
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App (CatBoost)")

# Load model and scaler
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# Define feature columns
numeric_features = [
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
    'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Delay_from_due_date'
]
categorical_features = [
    'Occupation', 'Income_Category', 'Age_Category', 'Spending_Level'
]
feature_cols = numeric_features + categorical_features

# Dropdown options
occupation_options = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer',
                      'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
                      'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer']

income_options = ['High Income', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']
age_options = ['Adults', 'Middle-Aged Adults', 'Older Adults', 'Teenagers', 'Young Adults']
spending_options = ['High', 'Low']

# Label mapping
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Sidebar input mode
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Entry Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")

    with st.form("manual_input_form"):
        user_input = {}
        for feature in numeric_features:
            user_input[feature] = st.number_input(feature, value=0.0)
        
        user_input['Occupation'] = st.selectbox("Occupation", occupation_options)
        user_input['Income_Category'] = st.selectbox("Income Category", income_options)
        user_input['Age_Category'] = st.selectbox("Age Category", age_options)
        user_input['Spending_Level'] = st.selectbox("Spending Level", spending_options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])

        # Scale numeric columns
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Create Pool for CatBoost with categorical feature names
        pool = Pool(data=input_df, cat_features=categorical_features)
        prediction = model.predict(pool)[0]

        label = label_map.get(int(prediction), "Unknown")
        st.success(f"🎯 Predicted Customer Category: **{label}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                df[numeric_features] = scaler.transform(df[numeric_features])
                pool = Pool(data=df, cat_features=categorical_features)
                preds = model.predict(pool)
                df["Predicted Category"] = [label_map.get(int(p), "Unknown") for p in preds]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv, "predicted_customers.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
