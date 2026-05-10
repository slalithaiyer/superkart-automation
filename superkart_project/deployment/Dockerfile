FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app.py now downloads the model dynamically, so these directories are not strictly needed inside the container
# as destinations for copied files. Keeping them would be harmless but unnecessary.
# RUN mkdir -p superkart_project/model
# RUN mkdir -p superkart_project/deployment

# app.py and requirements.txt are uploaded to the root of the Hugging Face Space
# by the GitHub Actions workflow, so copy them directly into the WORKDIR (/app).
COPY app.py .

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
