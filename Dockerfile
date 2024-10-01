# Use a base image that includes Python
FROM python:3.12

# Install the required system packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Command to run your Streamlit app
CMD ["streamlit", "run", "dynamic-object-detection.py", "--server.port=8501", "--server.address=0.0.0.0"]
