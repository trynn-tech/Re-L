FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Pre-install pipenv
RUN pip3 install pipenv

# We don't COPY the code here because we use the Bind Mount in Nix
# This keeps the image lightweight and focused on the "Environment"
WORKDIR /app
