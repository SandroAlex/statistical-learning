# Python 3.10.12 with Ubuntu 22.04.4 LTS.
FROM ubuntu:22.04

# Labels.
LABEL \
    maintainer="Alex Araujo" \
    email="alex.fate2000@gmail.com"

# Do not buffer the output. See the logs imediately.
ENV PYTHONUNBUFFERED 1

# Prevent Python from writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE 1

# Disable apt from prompting.
ENV DEBIAN_FRONTEND=noninteractive

# Default shell inside container.
ENV SHELL=/bin/bash

# Set working folder.
WORKDIR /statapp

# Copy python dependencies and initial command.
COPY requirements.txt /statapp/requirements.txt
COPY entrypoint.sh /statapp/entrypoint.sh

# Install dependencies.
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # For building Python packages.
    build-essential \
    # Python dependencies.
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Cleaning up unused files.
    apt-get autoremove --yes && \
    apt-get clean --yes && \
    rm -rf /var/lib/apt/lists/*

# Use apt in interactive mode when we are actually using docker container.
ENV DEBIAN_FRONTEND=dialog

# Apply database migrations.
ENTRYPOINT ["/bin/bash", "/statapp/entrypoint.sh"]
