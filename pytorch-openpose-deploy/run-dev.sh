#!/bin/bash
echo "🚀 Starting PyTorch OpenPose Development Environment..."

# Check if model file exists
if [ ! -f "model/body_coco.pth" ]; then
    echo "❌ Error: Model file 'model/body_coco.pth' not found!"
    echo "Please download the required model files to the model/ directory."
    exit 1
fi

# Set up X11 forwarding for GUI
xhost +local:docker 2>/dev/null || echo "Note: GUI may not work on this system"

# Stop and remove existing dev container if it exists
docker stop pytorch-openpose-dev 2>/dev/null || true
docker rm pytorch-openpose-dev 2>/dev/null || true

echo "🐳 Starting development container with volume mounting..."

# Run container with development setup
docker run \
  --name pytorch-openpose-dev \
  --ipc=host \
  --gpus all \
  --runtime=runc \
  --interactive \
  --tty \
  --detach \
  --shm-size=10gb \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/app \
  --volume="/dev:/dev" \
  --privileged \
  --workdir="/app" \
  pytorch-openpose bash

echo "✅ Development container started!"
echo ""
echo "📋 Development Commands:"
echo "  docker exec -it pytorch-openpose-dev bash    # Enter container shell"
echo "  docker exec -it pytorch-openpose-dev python3 main.py    # Run main application"
echo "  docker logs pytorch-openpose-dev             # View container logs"
echo "  docker stop pytorch-openpose-dev             # Stop container"
echo ""
echo "🔧 Inside the container, you can:"
echo "  - Edit files on host (changes persist)"
echo "  - Run python3 main.py to start the application"
echo "  - Install additional packages with pip3"
echo "  - Use all development tools"