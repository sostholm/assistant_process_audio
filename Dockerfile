# # Use the official Python base image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container
# COPY requirements.txt .

# # Install the required Python packages
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code into the container
# COPY . .

# # Expose the port the app runs on
# EXPOSE 9000

# # Command to run the application
# CMD ["uvicorn", "whisper_server:app", "--host", "0.0.0.0", "--port", "9000"]

# Use a base image with PyTorch and CUDA installed
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set 'python' command to point to Python 3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 9000

# Command to run the application
CMD ["uvicorn", "whisper_server:app", "--host", "0.0.0.0", "--port", "9000"]
