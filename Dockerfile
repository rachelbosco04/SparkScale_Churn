FROM python:3.9-slim

WORKDIR /app

# Install Java for Spark
RUN apt-get update && apt-get install -y openjdk-11-jdk

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY models/ models/
COPY api/ api/

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]