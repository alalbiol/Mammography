#!/bin/bash

#!/bin/bash

set -e

# Function to install Go if not installed
install_go() {
    if command -v go >/dev/null 2>&1; then
        echo "Go is already installed: $(go version)"
    else
        echo "Installing Go..."
        GO_VERSION=1.22.3
        wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
        sudo rm -rf /usr/local/go
        sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
        rm go${GO_VERSION}.linux-amd64.tar.gz

        # Set up Go paths for current session
        export PATH=$PATH:/usr/local/go/bin
        echo "Go installed: $(go version)"
    fi
}

# Step 1: Update and install dependencies
echo "Installing system dependencies..."
sudo apt update && sudo apt install -y \
    build-essential \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup \
    curl

# Step 2: Ensure Go is installed
install_go

# Step 3: Install Apptainer
APPTAINER_VERSION=1.4.1
echo "Downloading Apptainer v${APPTAINER_VERSION}..."
wget https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer-${APPTAINER_VERSION}.tar.gz
tar -xzf apptainer-${APPTAINER_VERSION}.tar.gz
cd apptainer-${APPTAINER_VERSION}

echo "Building Apptainer..."
./mconfig
make -C builddir
sudo make -C builddir install

# Step 4: Check installation
echo "Apptainer installed successfully:"
apptainer version
