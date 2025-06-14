import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App (CatBoost)")

# Load model and scaler
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# Feature columns (with direct string values for categorical variables)
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

# Output labels
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Possible dropdown values (used directly in the form)
occupation_options = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer',
                      'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
                      'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer']

income_options = ['High Income', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']
age_options = ['Adults', 'Middle-Aged Adults', 'Older Adults', 'Teenagers', 'Young Adults']
spending_options = ['High', 'Low']

# Sidebar selection
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Input Mode ---
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
            'Occupation': st.selectbox("Occupation", occupation_options),
            'Income_Category': st.selectbox("Income Category", income_options),
            'Age_Category': st.selectbox("Age Category", age_options),
            'Spending_Level': st.selectbox("Spending Level", spending_options),
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        scaled_numeric = scaler.transform(input_df.select_dtypes(include=['float64', 'int64']))
        input_df[input_df.select_dtypes(include=['float64', 'int64']).columns] = scaled_numeric
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Category: **{label_map.get(prediction, 'Unknown')}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(col in df.columns for col in feature_cols):
                st.error(f"❌ CSV must contain: {', '.join(feature_cols)}")
            else:
                df_copy = df.copy()
                numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
                df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])
                preds = model.predict(df_copy)
                df['Predicted Category'] = [label_map.get(p, "Unknown") for p in preds]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv, "predicted_categories.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
