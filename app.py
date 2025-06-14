import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App (CatBoost)")

# Load model and scaler
model = joblib.load("catboost.pkl")
scaler = joblib.load("scaler1.pkl")

# Feature columns
feature_cols = [
    'Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
    'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Delay_from_due_date',
    'Occupation', 'Income_Category', 'Age_Category', 'Spending_Level'
]

# Label mapping
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Categorical options
occupation_map = {'Accountant': 0, 'Architect': 1, 'Developer': 2, 'Doctor': 3, 'Engineer': 4,
                  'Entrepreneur': 5, 'Journalist': 6, 'Lawyer': 7, 'Manager': 8, 'Mechanic': 9,
                  'Media_Manager': 10, 'Musician': 11, 'Scientist': 12, 'Teacher': 13, 'Writer': 14}

income_map = {'High Income': 0, 'Low Income': 1, 'Lower Middle Income': 2, 'Upper Middle Income': 3}

age_map = {'Adults': 0, 'Middle-Aged Adults': 1, 'Older Adults': 2, 'Teenagers': 3, 'Young Adults': 4}

spending_map = {'High': 0, 'Low': 1}

# Streamlit Sidebar
st.sidebar.header("📥 Select Input Method")
mode = st.sidebar.radio("Input Method", ["Manual Entry", "Upload CSV"])

# --- Manual Entry ---
if mode == "Manual Entry":
    st.subheader("📝 Manual Input")
    with st.form("manual_form"):
        numeric_inputs = {
            'Outstanding_Debt': st.number_input("Outstanding Debt"),
            'Monthly_Inhand_Salary': st.number_input("Monthly Inhand Salary"),
            'Total_EMI_per_month': st.number_input("Total EMI per Month"),
            'Credit_Utilization_Ratio': st.number_input("Credit Utilization Ratio"),
            'Credit_History_Age_Months': st.number_input("Credit History Age (Months)"),
            'Delay_from_due_date': st.number_input("Delay from Due Date"),
        }

        categorical_inputs = {
            'Occupation': st.selectbox("Occupation", list(occupation_map.keys())),
            'Income_Category': st.selectbox("Income Category", list(income_map.keys())),
            'Age_Category': st.selectbox("Age Category", list(age_map.keys())),
            'Spending_Level': st.selectbox("Spending Level", list(spending_map.keys()))
        }

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Combine and encode
        all_inputs = {**numeric_inputs, **categorical_inputs}
        input_df = pd.DataFrame([all_inputs])

        # Encode categorical columns
        input_df['Occupation'] = input_df['Occupation'].map(occupation_map)
        input_df['Income_Category'] = input_df['Income_Category'].map(income_map)
        input_df['Age_Category'] = input_df['Age_Category'].map(age_map)
        input_df['Spending_Level'] = input_df['Spending_Level'].map(spending_map)

        # Scale numeric features only
        numeric_cols = input_df.select_dtypes(include='number').columns
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Category: **{label_map.get(prediction, 'Unknown')}**")

# --- CSV Upload ---
else:
    st.subheader("📁 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Validate columns
            if not all(col in df.columns for col in feature_cols):
                st.error(f"❌ CSV must contain: {', '.join(feature_cols)}")
            else:
                # Map categorical values
                df['Occupation'] = df['Occupation'].map(occupation_map)
                df['Income_Category'] = df['Income_Category'].map(income_map)
                df['Age_Category'] = df['Age_Category'].map(age_map)
                df['Spending_Level'] = df['Spending_Level'].map(spending_map)

                df = df.fillna(0)

                # Scale numeric features
                numeric_cols = df.select_dtypes(include='number').columns
                df[numeric_cols] = scaler.transform(df[numeric_cols])

                preds = model.predict(df)
                df["Predicted Category"] = [label_map.get(p, "Unknown") for p in preds]

                st.success("✅ Predictions Done!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
