# use official python 3.9 base image
FROM python:3.9-slim

# set working directory
WORKDIR /app

# prevent Python from writing .pyc files to the container
ENV PYTHONDONTWRITEBYTECODE 1
# ensure Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# expose port
EXPOSE 8000
