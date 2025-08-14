#!/bin/bash

# PyTorch OpenPose - New PC Setup Script
# Run this script on a fresh Ubuntu system to install all dependencies

set -e

echo "=== PyTorch OpenPose Setup Script ==="
echo "This script will install Docker, NVIDIA Container Toolkit, and dependencies"
echo "Press Ctrl+C to cancel, or press Enter to continue..."
read

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root (don't use sudo)"
   exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install basic dependencies
print_status "Installing basic dependencies..."
sudo apt install -y curl wget gnupg lsb-release ca-certificates

# Install Docker
print_status "Installing Docker..."
if command -v docker &> /dev/null; then
    print_warning "Docker is already installed"
else
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    print_warning "You need to log out and back in for Docker group changes to take effect"
fi

# Install Docker Compose
print_status "Installing Docker Compose..."
if command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose is already installed"
else
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Check for NVIDIA GPU
print_status "Checking for NVIDIA GPU..."
if lspci | grep -i nvidia > /dev/null; then
    print_status "NVIDIA GPU detected, installing NVIDIA drivers and container toolkit..."
    
    # Install NVIDIA drivers if not present
    if ! command -v nvidia-smi &> /dev/null; then
        print_status "Installing NVIDIA drivers..."
        sudo apt install -y ubuntu-drivers-common
        sudo ubuntu-drivers autoinstall
        print_warning "NVIDIA drivers installed. System reboot required!"
    fi
    
    # Install NVIDIA Container Toolkit
    print_status "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    
    # Configure Docker for NVIDIA
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
else
    print_warning "No NVIDIA GPU detected. Application will run in CPU mode (slower performance)"
fi

# Add user to video group for camera access
print_status "Setting up camera permissions..."
sudo usermod -aG video $USER

# Install additional tools
print_status "Installing additional tools..."
sudo apt install -y htop git tree

# Create application directory
print_status "Setting up application directory..."
mkdir -p ~/pytorch-openpose

print_status "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Copy the pytorch-openpose-deploy directory to ~/pytorch-openpose/"
echo "2. If NVIDIA drivers were installed, reboot the system"
echo "3. Log out and back in for group changes to take effect"
echo "4. Navigate to ~/pytorch-openpose/pytorch-openpose-deploy/"
echo "5. Run: docker build -t pytorch-openpose ."
echo "6. Run: ./run-x11.sh"
echo ""

# Test installations
echo "=== Testing Installations ==="
print_status "Docker version:"
docker --version

print_status "Docker Compose version:"
docker-compose --version

if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA driver version:"
    nvidia-smi --version
fi

print_status "Available cameras:"
ls /dev/video* 2>/dev/null || echo "No camera devices found"

echo ""
print_status "Setup script completed successfully!"