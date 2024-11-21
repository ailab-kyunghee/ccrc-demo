
# Use a base image with CUDA and development libraries
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda

# Update the package list and install necessary tools
RUN apt-get update && \
    apt-get install -y curl unzip bzip2 git && \
    apt-get clean

# Download and install Anaconda
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh && \
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b && \
    rm Anaconda3-2024.10-1-Linux-x86_64.sh

# Add Anaconda to PATH
ENV PATH="/root/anaconda3/bin:$PATH"

# Copy the source code from the host machine into the container
COPY /backend /app

# Create and initialize the Conda environment
RUN conda create -n llava python=3.10 -y && \
    conda init bash

# Activate the environment and install required packages in one RUN command
RUN /bin/bash -c "\
    source /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate llava && \
    pip install flask && \
    pip install opencv-python-headless && \
    pip install flask flask_cors && \
    pip install --upgrade pip && \
    pip install -e /app/LLaVA && \
    pip install -e /app/LLaVA[train] \
"

# Set up the entry point to activate the environment and start the server
CMD ["/bin/bash", "-c", "source /root/anaconda3/etc/profile.d/conda.sh && conda activate llava && cd /app && python server_iitp_version2.py"]
