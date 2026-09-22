#!/bin/bash
set -e
echo "=== System Updates and Dependencies ==="
export DEBIAN_FRONTEND=noninteractive
rm -rf /var/lib/apt/lists/*
apt-get clean
apt-get update -y
apt-get install -y python3-pip python3-venv wget curl bc file dc
echo "Root setup complete."
