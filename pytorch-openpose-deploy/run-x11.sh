#!/bin/bash

# Script to run Docker with proper X11 forwarding for any display setup

# Allow X11 connections from localhost (works with both laptop and external monitors)
xhost +local:docker

# Export DISPLAY if not already set
export DISPLAY=${DISPLAY:-:0}

# Create a home .Xauthority file if it doesn't exist or copy the current one
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
    cp "$XAUTHORITY" "$HOME/.Xauthority" 2>/dev/null || true
fi

# Use home directory .Xauthority for Docker
export XAUTHORITY="$HOME/.Xauthority"

# Make sure we have X11 auth (fallback method)
touch "$HOME/.Xauthority"
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f "$HOME/.Xauthority" nmerge - 2>/dev/null || true

# Run docker-compose
echo "Starting Docker container with X11 forwarding..."
echo "Display: $DISPLAY"
echo "XAuth: $XAUTHORITY"

# Run the development container
docker-compose up -d pytorch-openpose-dev

# Optional: Exec into the container
docker-compose exec pytorch-openpose-dev bash

# Cleanup X11 permissions when done (optional)
# xhost -local:docker