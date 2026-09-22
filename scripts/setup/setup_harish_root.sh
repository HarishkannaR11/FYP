#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sed -i 's/us.us.archive.ubuntu.com/archive.ubuntu.com/g' /etc/apt/sources.list.d/ubuntu.sources
sed -i 's/us.archive.ubuntu.com/archive.ubuntu.com/g' /etc/apt/sources.list.d/ubuntu.sources
echo 'Acquire::By-Hash "no";' > /etc/apt/apt.conf.d/01byhash
echo 'Acquire::http::Pipeline-Depth 0;' > /etc/apt/apt.conf.d/99fixbadproxy
rm -rf /var/lib/apt/lists/*
apt-get clean
apt-get update -y || true
apt-get install -y python3-pip python3-venv wget curl bc file dc tar
echo "Root dependencies installed."
