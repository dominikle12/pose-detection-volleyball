# PyTorch OpenPose Development Guide

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Start development environment
docker-compose up -d pytorch-openpose-dev

# Enter development container
docker-compose exec pytorch-openpose-dev bash

# Inside container, run the application
python3 main.py

# Stop development environment
docker-compose down
```

### Option 2: Using Development Script
```bash
# Start development environment
./run-dev.sh

# Enter container
docker exec -it pytorch-openpose-dev bash

# Stop container
docker stop pytorch-openpose-dev
```

### Option 3: Run Application Directly
```bash
# Run application (non-interactive)
docker-compose up pytorch-openpose

# Or run in background
docker-compose up -d pytorch-openpose
docker-compose logs -f pytorch-openpose
```

## Development Workflow

1. **Edit files on host**: All changes to files in this directory are automatically reflected in the container
2. **Test in container**: Run `python3 main.py` inside the container to test changes
3. **Debug**: Use `docker-compose logs` to view application output
4. **Restart**: Use `docker-compose restart` to restart services

## Key Features

- ✅ **Volume mounting**: Edit files on host, run in container
- ✅ **GPU support**: Automatic NVIDIA GPU detection and usage
- ✅ **Camera access**: Direct access to `/dev/video0` and other cameras
- ✅ **X11 forwarding**: GUI applications work (if X11 is available)
- ✅ **Hot reload**: File changes are immediately available in container

## Fixed Issues

- ✅ **Threading**: Fixed `task_done()` called too many times error
- ✅ **Security**: Added `weights_only=True` to `torch.load()` calls
- ✅ **Stability**: Improved error handling in pose estimation thread

## Troubleshooting

### Camera not working
```bash
# Check available cameras
ls -la /dev/video*

# Update docker-compose.yml to use correct camera device
devices:
  - /dev/video1:/dev/video1  # Change to your camera device
```

### GUI not working
```bash
# Allow X11 connections
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY
```

### GPU not detected
```bash
# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Commands Reference

```bash
# Build image
docker build -t pytorch-openpose .

# Development commands
docker-compose up -d pytorch-openpose-dev    # Start dev environment
docker-compose exec pytorch-openpose-dev bash # Enter container
docker-compose logs pytorch-openpose-dev      # View logs
docker-compose restart pytorch-openpose-dev   # Restart service
docker-compose down                           # Stop all services

# Production commands
docker-compose up pytorch-openpose           # Run application
docker-compose up -d pytorch-openpose        # Run in background
docker-compose logs -f pytorch-openpose      # Follow logs
```