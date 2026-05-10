FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY superkart_project/model/best_random_forest_model.joblib ./superkart_project/model/
COPY superkart_project/deployment/app.py ./superkart_project/deployment/

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "superkart_project/deployment/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
