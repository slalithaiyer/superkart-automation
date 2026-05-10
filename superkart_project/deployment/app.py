import streamlit as st
import pandas as pd
import joblib

model = joblib.load("superkart_project/model/best_random_forest_model.joblib")

st.title("Superkart Sales Prediction")

product_mrp = st.number_input("Product MRP", min_value=1.0)
product_weight = st.number_input("Product Weight", min_value=0.1)
allocated_area = st.number_input("Allocated Display Area Ratio", min_value=0.0, max_value=1.0)
sugar_content = st.selectbox("Sugar Content", ["Low", "Regular", "No Sugar"])
store_size = st.selectbox("Store Size", ["High", "Medium", "Low"])
city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])

if st.button("Predict Sales"):
    input_data = pd.DataFrame([{
        "Product_MRP": product_mrp,
        "Product_Weight": product_weight,
        "Product_Allocated_Area": allocated_area,
        "Product_Sugar_Content": sugar_content,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales: {prediction:.2f}")
