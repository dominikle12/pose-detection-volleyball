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
