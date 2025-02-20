# Use the official Python image as a base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (including libGL for OpenCV)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
