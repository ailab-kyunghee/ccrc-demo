# MedCLIP Setup Guide

This guide explains how to set up the Docker container for this project, including the installation of the NVIDIA Docker Toolkit for GPU support.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup Steps](#setup-steps)
  - [1. Install Docker](#1-install-docker)
  - [2. Install NVIDIA Drivers](#2-install-nvidia-drivers)
  - [3. Install NVIDIA Docker Toolkit](#3-install-nvidia-docker-toolkit)
  - [4. Verify NVIDIA Docker Installation](#4-verify-nvidia-docker-installation)
  - [5. Build and Run the Docker Container](#5-build-and-run-the-docker-container)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites
- **Operating System**: Ubuntu 18.04, 20.04, or later
- **NVIDIA GPU** with supported drivers installed
- **Docker** version 19.03 or later

## Setup Steps

### 1. Install Docker

To install Docker, run the following commands:

```bash
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Docker and NVIDIA Setup Guide

## 1. Add Docker’s Official GPG Key

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

## 2. Set Up the Docker Repository

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

## 3. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

## 4. Install NVIDIA Drivers

Ensure you have the NVIDIA drivers installed for your GPU. You can install the latest drivers using:

```bash
sudo apt-get install nvidia-driver-<version>
```

or pull the latest CUDA image with:

```bash
docker pull nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
```

## 5. Install NVIDIA Docker Toolkit

### Add the Package Repositories

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

### Install the NVIDIA Container Toolkit

Follow the instructions on the following link to set up Docker NVIDIA Toolkit and install it using Apt:

[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Commands are as follows:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

## 6. Verify NVIDIA Docker Installation

Run the following command to ensure NVIDIA Docker is installed correctly:

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## 7. Build and Run the Docker Container

### Build the Docker Image

```bash
docker build -t medclip-api .
```

### Run the Docker Container

```bash
docker run --gpus all -d --name medclip-container -p 8080:8080 medclip-api
```
