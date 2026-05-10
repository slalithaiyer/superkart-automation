import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, hf_hub_download
import os

# Get Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Define repository details and filename for the raw data
repo_id = "Lalithas/Superkart-Prediction" # Corrected for Superkart
raw_data_filename = "superkart.csv" # Corrected for Superkart

print(f"Attempting to download raw dataset from: {repo_id}/{raw_data_filename}")

# Step 1: Download raw dataset from Hugging Face Hub
local_raw_data_path = hf_hub_download(repo_id=repo_id, filename=raw_data_filename, repo_type="dataset", token=HF_TOKEN)
print(f" Raw dataset downloaded successfully to: {local_raw_data_path}")

# Step 2: Load dataset from the local path
df = pd.read_csv(local_raw_data_path)
print(" Dataset loaded successfully into DataFrame.")

# Step 3: Drop unique identifier columns if present (corrected for Superkart columns)
id_cols = ["Product_Id", "Store_Id"]
for col in id_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Dropped identifier column '{col}'.")

# Step 4: Encode categorical columns (corrected for Superkart columns)
categorical_cols = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]
for col in categorical_cols:
    if col in df.columns:
        # Fill missing string values with 'Unknown' before encoding
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        print(f"Encoded '{col}' column.")

# Step 5: Define target column (corrected for Superkart target)
target_col = "Product_Store_Sales_Total"

# Step 6: Handle missing numerical values if any before splitting
for col in df.columns:
    if df[col].dtype != 'object': # Only process numerical columns
        df[col] = df[col].fillna(df[col].median()) # Filling with median as a robust strategy

# Step 7: Split into features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 8: Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42 # Superkart is regression, so no stratify
)

# Step 9: Save processed data locally within the 'superkart_project/data' directory
output_dir = "superkart_project/data" # Relative path within the cloned repo
os.makedirs(output_dir, exist_ok=True)

Xtrain.to_csv(f"{output_dir}/Xtrain.csv", index=False)
Xtest.to_csv(f"{output_dir}/Xtest.csv", index=False)
ytrain.to_csv(f"{output_dir}/ytrain.csv", index=False)
ytest.to_csv(f"{output_dir}/ytest.csv", index=False)

print(" Processed train/test datasets saved locally.")

# Step 10: Upload processed files to Hugging Face Hub (for tracking and subsequent downloads if needed)
files_to_upload = [
    f"{output_dir}/Xtrain.csv",
    f"{output_dir}/Xtest.csv",
    f"{output_dir}/ytrain.csv",
    f"{output_dir}/ytest.csv"
]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # just the filename
        repo_id=repo_id, # Assuming this repo_id is for dataset storage
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f" Uploaded {os.path.basename(file_path)} to {repo_id}")

print("✅ Data preparation complete, files saved locally and uploaded.")
