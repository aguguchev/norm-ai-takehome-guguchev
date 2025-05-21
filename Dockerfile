# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /norm-fullstack

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install uvicorn

# API key
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Copy the content of the local src directory to the working directory
COPY ./app /norm-fullstack/app
COPY ./docs /norm-fullstack/docs
COPY ./run.py /norm-fullstack/run.py

EXPOSE 80

# Command to run on container start
CMD ["python3", "/norm-fullstack/run.py"]
