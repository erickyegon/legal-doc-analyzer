#!/bin/bash

# Legal Intelligence Platform Backend Startup Script
# This script sets up and starts the FastAPI backend server

echo "Starting Legal Intelligence Platform Backend..."

# Set environment variables for production
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"
export ENVIRONMENT="production"

# Create necessary directories
mkdir -p uploads
mkdir -p logs

# Install dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Run database migrations (if using Alembic)
# echo "Running database migrations..."
# alembic upgrade head

# Start the application with Gunicorn for production
echo "Starting FastAPI application with Gunicorn..."
exec gunicorn app.main:app \
    --bind 0.0.0.0:$PORT \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --log-level info \
    --access-logfile - \
    --error-logfile -