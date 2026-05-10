from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Get Hugging Face token from environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define repository details for Superkart dataset
repo_id = "Lalithas/Superkart-Prediction"
repo_type = "dataset"

# Step 1: Check if the repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    # Step 2: Create the repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, token=os.getenv("HF_TOKEN"))
        print(f"✅ Repository '{repo_id}' created successfully.")
    except HfHubHTTPError as e:
        print(f"❌ Failed to create repository: {e}")

# Step 3: Explicitly upload the superkart.csv file
# This assumes the raw superkart.csv is present at this path locally.
# In a typical MLOps flow, this raw data might be sourced from a data lake/warehouse.
local_file_path = "superkart_project/data/superkart.csv"
# For demonstration, you might need to create a dummy superkart.csv here
# e.g., pd.DataFrame(...).to_csv(local_file_path, index=False)

if os.path.exists(local_file_path):
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo="superkart.csv",
        repo_id=repo_id,
        repo_type=repo_type,
        token=os.getenv("HF_TOKEN")
    )
    print(f"📂 File '{os.path.basename(local_file_path)}' uploaded to '{repo_id}'.")
else:
    print(f"❌ Error: Local file '{local_file_path}' not found. Please ensure 'superkart.csv' is placed in 'superkart_project/data/'. Cannot upload to Hugging Face.")
