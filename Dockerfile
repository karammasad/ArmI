# Use official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt || echo "No requirements.txt found, skipping."

# Copy the entire project (including main.py)
COPY . /app

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]

