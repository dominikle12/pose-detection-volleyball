#!/bin/bash
echo "🚀 Starting PyTorch OpenPose..."
if [ ! -f "model/body_coco.pth" ]; then
    echo "❌ Error: Model file 'model/body_coco.pth' not found!"
    echo "Please download the required model files to the model/ directory."
    exit 1
fi
xhost +local:docker 2>/dev/null || echo "Note: GUI may not work on this system"
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/video0:/dev/video0 \
  --device /dev/video0:/dev/video0 \
  --privileged \
  -v $(pwd)/model:/app/model \
  pytorch-openpose
