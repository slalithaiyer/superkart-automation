import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, hf_hub_download

# Initialize Hugging Face API
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Define repository details for data and model
data_repo_id = "Lalithas/Superkart-Prediction"
model_repo_id = "Lalithas/Superkart-Prediction-Model"

# Set MLflow tracking URI and experiment name
# Use an absolute path for MLflow tracking to ensure it's writable in CI/CD environments
mlflow_tracking_dir = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}") # Explicitly use file:// URI
mlflow.set_experiment("Superkart_Sales_Prediction")

# Download processed data from Hugging Face Hub
data_files = {
    "Xtrain": "Xtrain.csv",
    "Xtest": "Xtest.csv",
    "ytrain": "ytrain.csv",
    "ytest": "ytest.csv",
}

local_data_paths = {}
for key, filename in data_files.items():
    local_data_paths[key] = hf_hub_download(
        repo_id=data_repo_id, filename=filename, repo_type="dataset", token=HF_TOKEN
    )

Xtrain = pd.read_csv(local_data_paths["Xtrain"])
Xtest = pd.read_csv(local_data_paths["Xtest"])
ytrain = pd.read_csv(local_data_paths["ytrain"]).iloc[:, 0]
ytest = pd.read_csv(local_data_paths["ytest"]).iloc[:, 0]

print("✅ Datasets loaded successfully for training.")

# Define model and hyperparameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
}

# Ensure model repo exists on Hugging Face
try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
    print(f"✅ Model repository '{model_repo_id}' already exists.")
except Exception:
    api.create_repo(repo_id=model_repo_id, repo_type="model", private=False, token=HF_TOKEN)
    print(f"✅ Model repository '{model_repo_id}' created successfully.")

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning") as parent_run:
    print(f"MLflow Parent Run ID: {parent_run.info.run_id}")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    mlflow.log_params(best_params)

    # Child run for evaluation and model logging
    with mlflow.start_run(run_name="Best_RandomForest_Model_Evaluation", nested=True) as child_run:
        print(f"MLflow Child Run ID: {child_run.info.run_id}")

        preds = best_model.predict(Xtest)
        mae = mean_absolute_error(ytest, preds)
        r2 = r2_score(ytest, preds)

        print("Test Set Metrics:")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}") # Changed f"  R²: {r2:.4f}") to print for proper output

        mlflow.log_metrics({"test_mae": mae, "test_r2": r2})

        # Log the model using MLflow
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")
        print("✅ Best model logged to MLflow.")

        # Save model locally
        output_model_dir = "superkart_project/model"
        os.makedirs(output_model_dir, exist_ok=True)
        model_path = os.path.join(output_model_dir, "best_random_forest_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"✅ Best model saved locally at: {model_path}")

        # Upload the trained model to Hugging Face Model Hub
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_random_forest_model.joblib",
            repo_id=model_repo_id,
            repo_type="model",
            token=HF_TOKEN
        )
        print(f"✅ Best model uploaded to Hugging Face Model Hub: {model_repo_id}")

print("✅ Model training, tuning, evaluation, and registration complete.")
