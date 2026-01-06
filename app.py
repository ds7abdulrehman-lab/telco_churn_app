import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 1. REDEFINE THE CUSTOM FUNCTION (This is what the model is looking for)
def churn_engineering_feature(X):
    x_copy = X.copy()
    x_copy["IsFiber"] = (x_copy["InternetService"]=="Fiber optic").astype(int)
    cat_col = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    x_copy["Total_Services"] = (x_copy[cat_col] == "Yes").sum(axis = 1)
    x_copy["TotalCharges"] = pd.to_numeric(x_copy["TotalCharges"], errors='coerce').fillna(0)
    x_copy["ChargesPerMonth"] = x_copy["TotalCharges"]/(x_copy["tenure"] +0.1)
    
    # We drop these because the model wasn't trained on them
    return x_copy.drop(columns=["gender", "customerID", "InternetService"])

# 2. LOAD THE MODEL (Now it will find the function above and work!)
@st.cache_resource # This keeps the model in memory so it's fast
def load_my_model():
    return joblib.load('Churn_model_pipeline.pkl')

model = load_my_model()

# 3. STREAMLIT UI
st.title("📞 Telco Churn Intelligence Dashboard")

st.info("Input customer details on the left to see the churn risk score.")

# Sidebar for inputs
with st.sidebar:
    st.header("Customer Profile")
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 9000.0, 800.0)
    
    # Simple checkboxes for services (to feed into Total_Services)
    st.subheader("Services Subscribed")
    os = st.checkbox("Online Security")
    ob = st.checkbox("Online Backup")
    dp = st.checkbox("Device Protection")
    ts = st.checkbox("Tech Support")
    
# 4. PREPARE DATA FOR MODEL
# Model expects ALL original columns before the FunctionTransformer hits them
input_dict = {
    'customerID': "1234-WEB", 'gender': "Male", 'SeniorCitizen': 0, 
    'Partner': "No", 'Dependents': "No", 'tenure': tenure, 
    'PhoneService': "Yes", 'MultipleLines': "No", 'InternetService': internet,
    'OnlineSecurity': "Yes" if os else "No", 
    'OnlineBackup': "Yes" if ob else "No", 
    'DeviceProtection': "Yes" if dp else "No", 
    'TechSupport': "Yes" if ts else "No",
    'StreamingTV': "No", 'StreamingMovies': "No", 
    'Contract': contract, 'PaperlessBilling': "Yes",
    'PaymentMethod': "Electronic check", 'MonthlyCharges': monthly, 
    'TotalCharges': total
}

input_df = pd.DataFrame([input_dict])

# 5. PREDICTION
if st.button("Predict Churn Risk"):
    # Get Probability
    prob = model.predict_proba(input_df)[0][1]
    
    # Display Result
    st.subheader("Churn Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Score", f"{prob:.1%}")
    
    with col2:
        if prob > 0.7:
            st.error("HIGH RISK")
        elif prob > 0.4:
            st.warning("MEDIUM RISK")
        else:
            st.success("LOW RISK")

    # Business insight based on your IsFiber feature
    if internet == "Fiber optic" and prob > 0.5:
        st.write("💡 **Insight:** This customer is on Fiber Optic. Our analysis shows high churn in this segment—consider a tech-support check-in.")