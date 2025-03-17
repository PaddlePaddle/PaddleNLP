#!/bin/bash

# Determine if glog is installed by checking for lib and include files
check_glog_installed() {
    if [ -f "/usr/lib/x86_64-linux-gnu/libglog.so" ] && [ -d "/usr/include/glog" ]; then
        return 0
    else
        return 1
    fi
}

# Install glog based on package manager
install_glog() {
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu system
        echo "libgoogle-glog0v5 is not installed. Installing..."
        apt-get update
        apt-get install -y libgoogle-glog0v5 libgoogle-glog-dev
    elif command -v rpm &> /dev/null || command -v dnf &> /dev/null || command -v yum &> /dev/null; then
        # CentOS/RHEL system
        echo "glog is not installed. Please install it using your package manager."
        exit 1
    else
        echo "Unsupported package manager. Please install glog manually."
        exit 1
    fi
}

# Check if glog is installed
if check_glog_installed; then
    echo "glog is already installed."
else
    install_glog
fi

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt