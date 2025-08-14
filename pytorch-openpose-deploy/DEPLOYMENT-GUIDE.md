# PyTorch OpenPose - Deployment Guide

This guide explains how to run the PyTorch OpenPose application on another PC.

## Prerequisites

### System Requirements
- Linux Ubuntu 18.04+ or similar distribution
- NVIDIA GPU with CUDA support (GTX 1050 or better recommended)
- At least 8GB RAM
- 10GB+ free disk space
- Webcam/camera device

### Required Software
1. **Docker Engine** (20.10+)
2. **Docker Compose** (1.29+)
3. **NVIDIA Container Toolkit** (for GPU support)
4. **X11 Server** (usually pre-installed on Linux desktop)

## Installation Steps

### 1. Install Docker
```bash
# Update system
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

### 2. Install Docker Compose
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Install NVIDIA Container Toolkit (for GPU support)
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Verify GPU Setup
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Deployment

### 1. Transfer Application Files
Copy the entire `pytorch-openpose-deploy` directory to the target PC:
```bash
# On source PC
tar -czf pytorch-openpose-deploy.tar.gz pytorch-openpose-deploy/

# Transfer file to target PC (via USB, scp, etc.)
# On target PC
tar -xzf pytorch-openpose-deploy.tar.gz
cd pytorch-openpose-deploy
```

### 2. Build Docker Image
```bash
# Build the application image
docker build -t pytorch-openpose .
```

### 3. Setup Camera Permissions
```bash
# Make sure camera device exists
ls /dev/video*

# Add user to video group if needed
sudo usermod -aG video $USER
```

### 4. Run the Application

#### For Development/Interactive Mode:
```bash
# Make script executable
chmod +x run-x11.sh

# Run with X11 forwarding
./run-x11.sh

# Inside container, run:
python3 main.py
```

#### For Production Mode:
```bash
# Run the main application directly
docker-compose up pytorch-openpose
```

## Troubleshooting

### Display Issues
If you get X11/display errors:
```bash
# Allow X11 connections
xhost +local:docker

# Check display variable
echo $DISPLAY

# If using SSH, enable X11 forwarding
ssh -X username@hostname
```

### GPU Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Camera Issues
```bash
# Check camera device
ls -la /dev/video*

# Test camera with system tools
cheese  # or
ffplay /dev/video0
```

### Performance Issues
- Ensure NVIDIA drivers are properly installed
- Check GPU memory usage with `nvidia-smi`
- Verify camera resolution settings in the application
- Consider reducing detection resolution in config.py

## Network Setup (Remote Access)

If running on a headless server or remote machine:

### Option 1: X11 Forwarding via SSH
```bash
# Connect with X11 forwarding
ssh -X username@remote-pc

# Run application normally
cd pytorch-openpose-deploy
./run-x11.sh
```

### Option 2: VNC Server
```bash
# Install VNC server on remote machine
sudo apt install tightvncserver

# Start VNC session
vncserver :1

# Connect via VNC client and run application
```

## File Structure
```
pytorch-openpose-deploy/
├── Dockerfile
├── docker-compose.yml
├── run-x11.sh          # X11 startup script
├── run.sh              # Standard startup script
├── main.py             # Main application
├── requirements.txt    # Python dependencies
├── model/              # Pre-trained models
├── src/                # Source code
└── data/               # Application data
```

## Security Notes
- The container runs in privileged mode for GPU access
- X11 forwarding opens display access - use carefully on shared systems
- Camera device is mounted directly into container
- Consider firewall settings if accessing remotely

## Support
- Check logs: `docker-compose logs pytorch-openpose-dev`
- Container shell: `docker exec -it pytorch-openpose-development bash`
- System resources: `htop`, `nvidia-smi`