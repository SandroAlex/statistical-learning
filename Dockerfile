# Python 3.10.12 with Ubuntu 22.04.4 LTS
FROM ubuntu:22.04

# It sets up a user with specific UID and GID to avoid
# permission issues when mounting volumes from the host system into the container.
# The user is created with the same UID and GID as the host user, which allows for
# seamless file sharing between the host and the container without permission conflicts
ARG USER_NAME=worker-user
ARG USER_PASSWORD=worker-user
ARG USER_UID=1000
ARG USER_GID=1000

# Avoid interactive prompts during package installation.
# This environment variable is set to noninteractive mode, which prevents any 
# interactive prompts from appearing during package installation. This is 
# particularly useful in Dockerfiles to ensure that the build process can run 
# without manual intervention.
ARG DEBIAN_FRONTEND=noninteractive

# Set working folder
WORKDIR /statapp

# Copy project files
COPY pyproject.toml /statapp/pyproject.toml
COPY entrypoint.sh /statapp/entrypoint.sh
COPY islp /statapp/islp

# Prevents Python from writing pyc files to disc (equivalent to python -B option)
ENV PYTHONDONTWRITEBYTECODE 1 \
    # Prevents Python from buffering stdout and stderr (equivalent to python -u option)
    PYTHONUNBUFFERED 1 \
    # Ignore warning message from pip when installing packages as root
    PIP_ROOT_USER_ACTION=ignore

# Install main dependencies.
RUN apt-get update --yes && \
    # Operational system dependencies.
    apt-get install --yes --no-install-recommends \
        build-essential \
        python3-dev \
        python3-pip \
        python-is-python3 && \
    # Update pip to the latest version
    pip install --upgrade pip && \
    # Install uv using pip
    pip install uv && \
    # Cleaning up unused files
    apt-get autoremove --yes && \
    apt-get clean --yes && \
    rm -rf /var/lib/apt/lists/*

# Create the worker user with sudo privilegies
RUN groupadd --gid $USER_GID $USER_NAME && \
    useradd --uid $USER_UID --gid $USER_GID --create-home $USER_NAME && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    adduser $USER_NAME sudo

# Enable prompt color in .bashrc.
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /home/$USER_NAME/.bashrc && \
    # Give worker user permissions for root folder
    chown -R $USER_NAME:$USER_NAME /statapp

# Important tools are installed here (black, isort, pytest, uv, etc)
ENV PATH="/home/${USER_NAME}/.local/bin:$PATH"

# Switch to the worker user
USER $USER_NAME

# Install project in editable mode with dev depedencies creating a virtual environment
RUN uv sync --extra dev 

# Use apt in interactive mode when we are actually using docker container
ENV DEBIAN_FRONTEND=dialog

# Apply database migrations
ENTRYPOINT ["/bin/bash", "/statapp/entrypoint.sh"]
