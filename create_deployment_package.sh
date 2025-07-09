#!/bin/bash

echo "📦 Creating deployment package for PyTorch OpenPose..."

DEPLOY_DIR="pytorch-openpose-deploy"
mkdir -p $DEPLOY_DIR

echo "📋 Copying project files..."
cp -r src/ $DEPLOY_DIR/
cp main.py $DEPLOY_DIR/
cp camera_test.py $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/
cp Dockerfile $DEPLOY_DIR/
cp .dockerignore $DEPLOY_DIR/

mkdir -p $DEPLOY_DIR/model

if [ -f "model/body_coco.pth" ]; then
    echo "📁 Copying model files..."
    cp model/*.pth $DEPLOY_DIR/model/ 2>/dev/null || true
else
    echo "⚠️  Warning: No model files found in model/ directory"
fi

cat > $DEPLOY_DIR/install.sh << 'INNER_EOF'
#!/bin/bash
echo "🐳 Installing PyTorch OpenPose..."
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and log back in, then run this script again."
    exit 1
fi
echo "🔨 Building Docker image..."
docker build -t pytorch-openpose .
if [ $? -eq 0 ]; then
    echo "✅ Installation complete!"
    echo "📋 To run: ./run.sh"
else
    echo "❌ Build failed."
    exit 1
fi
INNER_EOF

cat > $DEPLOY_DIR/run.sh << 'INNER_EOF'
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
INNER_EOF

chmod +x $DEPLOY_DIR/*.sh

echo "📦 Creating archive..."
tar -czf pytorch-openpose-deploy.tar.gz $DEPLOY_DIR

echo "✅ Deployment package created: pytorch-openpose-deploy.tar.gz"
echo "📤 Copy this file to another computer and extract it to run your project!"
