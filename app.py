import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App")

# Load model and scaler
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# Feature columns
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

numeric_cols = [
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
    'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Delay_from_due_date'
]

categorical_cols = ['Occupation', 'Income_Category', 'Age_Category', 'Spending_Level']

# Label map
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Dropdown options
occupation_options = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur', 'Journalist',
                      'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer']
income_options = ['High Income', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']
age_options = ['Adults', 'Middle-Aged Adults', 'Older Adults', 'Teenagers', 'Young Adults']
spending_options = ['High', 'Low']

# Sidebar
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Input ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        numeric_inputs = {}
        for col in numeric_cols:
            numeric_inputs[col] = st.number_input(f"{col}", value=0.0)

        categorical_inputs = {
            'Occupation': st.selectbox("Occupation", occupation_options),
            'Income_Category': st.selectbox("Income Category", income_options),
            'Age_Category': st.selectbox("Age Category", age_options),
            'Spending_Level': st.selectbox("Spending Level", spending_options)
        }

        submitted = st.form_submit_button("Predict")

    if submitted:
        full_input = {**numeric_inputs, **categorical_inputs}
        input_df = pd.DataFrame([full_input])

        # Scale numeric columns only
        input_df_scaled = input_df.copy()
        input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Ensure correct column order
        input_df_scaled = input_df_scaled[numeric_cols + categorical_cols]

        # Predict
        prediction = model.predict(input_df_scaled)[0]
        st.success(f"🎯 Predicted Category: **{label_map.get(prediction, 'Unknown')}**")

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
                df = df[feature_cols]
                df[numeric_cols] = scaler.transform(df[numeric_cols])

                # Ensure correct order
                df = df[numeric_cols + categorical_cols]

                preds = model.predict(df)
                df["Predicted Category"] = [label_map.get(p, "Unknown") for p in preds]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predicted_categories.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
