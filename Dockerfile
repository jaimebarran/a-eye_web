# Base image
FROM python:3.13

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Docker CLI
RUN apt-get update && apt-get install -y docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# optimise build time by enhance caching usage
COPY static/ ./static/
COPY templates/ ./templates/
COPY jobfiles/ ./jobfiles/

COPY package/ ./package/
COPY *.py .
COPY LICENSE.txt .

# Set environment variables
ENV FLASK_DEBUG=0

# Expose port
EXPOSE 5000

# ------------------------------
# Docker utilities

# Build docker image
# docker build -t aeyeweb .

# See docker images
# docker image ls or docker images

# Remove docker image
# docker rmi aeyeweb

# Rebuild docker image
# docker build --no-cache -t aeyeweb .

# Run docker container
# docker run -p 5000:5000 aeyeweb

# See docker containers
# docker container ls or docker ps

# Remove docker container
# docker rm <container_id>
