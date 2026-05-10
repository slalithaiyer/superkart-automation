import streamlit as st
import pandas as pd
import joblib
import os

# Define the relative path to the model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best_random_forest_model.joblib")

# Ensure the model file exists before loading
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}. Please ensure the model is trained and saved.")
else:
    model = joblib.load(model_path)

    st.title("Superkart Sales Prediction")

    # User input form
    st.header("Product and Store Details")
    product_mrp = st.number_input("Product MRP", min_value=1.0, value=100.0, help="Maximum Retail Price of the product.")
    product_weight = st.number_input("Product Weight (kg)", min_value=0.1, value=10.0, help="Weight of the product in kilograms.")
    allocated_area = st.number_input("Allocated Display Area Ratio", min_value=0.0, max_value=1.0, value=0.05, help="Ratio of the store's display area allocated to the product.")

    sugar_content_options = ["Low", "Regular", "No Sugar"]
    product_type_options = ["Breakfast", "Dairy", "Drinks", "Fruits and Vegetables", "Meat", "Snacks"]
    store_size_options = ["High", "Medium", "Low"]
    city_type_options = ["Tier 1", "Tier 2", "Tier 3"]
    store_type_options = ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"]

    product_sugar_content = st.selectbox("Product Sugar Content", sugar_content_options)
    product_type = st.selectbox("Product Type", product_type_options)
    store_size = st.selectbox("Store Size", store_size_options)
    store_location_city_type = st.selectbox("Store Location City Type", city_type_options)
    store_type = st.selectbox("Store Type", store_type_options)

    # Convert categorical inputs to numerical using a consistent mapping
    # This mapping must match the LabelEncoder logic used during training.
    # For simplicity, we'll use a fixed mapping here. In a real scenario, this would be saved from prep.py.
    sugar_content_map = {category: idx for idx, category in enumerate(sugar_content_options)}
    product_type_map = {category: idx for idx, category in enumerate(product_type_options)}
    store_size_map = {category: idx for idx, category in enumerate(store_size_options)}
    city_type_map = {category: idx for idx, category in enumerate(city_type_options)}
    store_type_map = {category: idx for idx, category in enumerate(store_type_options)}

    if st.button("Predict Sales"):
        if 'model' in locals(): # Only proceed if model was loaded successfully
            # Create a DataFrame for prediction
            input_data = pd.DataFrame([{
                "Product_Weight": product_weight,
                "Product_Sugar_Content": sugar_content_map.get(product_sugar_content, -1),
                "Product_Allocated_Area": allocated_area,
                "Product_MRP": product_mrp,
                "Product_Type": product_type_map.get(product_type, -1),
                "Store_Size": store_size_map.get(store_size, -1),
                "Store_Location_City_Type": city_type_map.get(store_location_city_type, -1),
                "Store_Type": store_type_map.get(store_type, -1)
            }])
            # Ensure columns are in the same order as training data (crucial for consistent predictions)
            # This assumes Xtrain.columns was saved or known.
            # For this example, we'll try to match the order from the `prep.py` script's output Xtrain.
            # A robust solution would save Xtrain.columns to a file.
            expected_columns = [
                'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 'Product_MRP',
                'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type'
            ]
            # Add any missing columns with default values (e.g., 0) if necessary, and reorder
            for col in expected_columns:
                if col not in input_data.columns:
                    input_data[col] = 0 # Or a suitable default/median value
            input_data = input_data[expected_columns]

            # Predict
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Sales: ${prediction:.2f}")
        else:
            st.warning("Model could not be loaded, please check the setup.")
