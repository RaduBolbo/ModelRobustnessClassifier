#!/bin/bash

# Function to install Python, pip, and Kaggle CLI
install_dependencies() {
    echo "Checking for Python..."
    if ! command -v python3 &> /dev/null; then
        echo "Python not found. Installing Python..."
        sudo apt update && sudo apt install -y python3
    else
        echo "Python is already installed."
    fi

    echo "Checking for pip..."
    if ! command -v pip &> /dev/null; then
        echo "pip not found. Installing pip..."
        sudo apt install -y python3-pip
    else
        echo "Pip is already installed."
    fi
}

create_venv(){
    # Check if python3-venv is installed
    if ! dpkg -l | grep -q python3.10-venv; then
        echo "python3.10-venv is not installed. Installing python3.10-venv..."
        sudo apt update && sudo apt install -y python3.10-venv || { echo "Failed to install python3.10-venv. Exiting."; exit 1; }
    fi

    # Create a new virtual environment
    python3 -m venv "$VENV_NAME"

    # Check if the virtual environment was created successfully
    if [ $? -eq 0 ]; then
        echo "Virtual environment created successfully in '$VENV_NAME'."
    else
        echo "Failed to create the virtual environment."
        exit 1
    fi

    source "$VENV_NAME/bin/activate"
    
    ehco "Installing requirements..."
    pip install -r requirements.txt
}

install_dependencies

# Check if libgl1 is installed
if dpkg -l | grep -q "libgl1"; then
    echo "libgl1 is already installed."
else
    echo "libgl1 is not installed, updating package list and installing libgl1..."
    sudo apt-get update && sudo apt-get install -y libgl1
fi

create_venv

