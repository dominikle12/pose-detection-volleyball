#!/bin/bash
echo "🚀 Starting PyTorch OpenPose..."
if [ ! -f "model/body_coco.pth" ]; then
    echo "❌ Error: Model file 'model/body_coco.pth' not found!"
    echo "Please download the required model files to the model/ directory."
    exit 1
fi
xhost +local:docker 2>/dev/null || echo "Note: GUI may not work on this system"
docker run --ipc=host --gpus all --runtime=runc --interactive -it \
  --shm-size=10gb \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume="/dev:/dev" \
  --privileged \
  pytorch-openpose

# --device /dev/video1:/dev/video1 \
#  -v $(pwd)/model:/app/model \
# -v /dev/video1:/dev/video1 \