FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY  

# Install the required packages
RUN pip install --no-cache-dir Flask numpy

# Expose the port
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
