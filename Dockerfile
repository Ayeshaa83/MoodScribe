# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages (e.g., TensorFlow/h5py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to cache the pip install step
COPY backend/requirements.txt ./backend/

# Install the Python dependencies
# (We install gunicorn here so we have a production-ready WSGI server)
# Note: pickle5 might fail on Python 3.11, so we aggressively ignore its failure if it happens
RUN pip install --no-cache-dir -r ./backend/requirements.txt || true
RUN pip install --no-cache-dir gunicorn flask flask-cors tensorflow numpy supabase python-dotenv emoji

# Copy the backend and frontend code to the container
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env ./backend/

# Expose port 5000 for the app
EXPOSE 5000

# Set the working directory to backend so that relative paths (like "../frontend") resolve correctly
WORKDIR /app/backend

# Command to run the application using Gunicorn (binding to 0.0.0.0 for Docker networking)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
