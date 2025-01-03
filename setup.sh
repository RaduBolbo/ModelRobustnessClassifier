#!/bin/bash

# Kaggle credentials
KAGGLE_USERNAME="username"
KAGGLE_API_KEY="key"

# Define directories
DATASET_DIR="dataset/"
DATASET_ID="alessiocorrado99/animals10"
DATASET_ZIP="$DATASET_DIR/animals10.zip"

VENV_NAME="myenv"
mkdir -p "$DATASET_DIR"

CHECKPOINTS_PATH="checkpoints"
mkdir -p "$CHECKPOINTS_PATH"

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

    echo "Checking for Kaggle CLI..."
    if ! command -v kaggle &> /dev/null; then
        echo "Kaggle CLI not found. Installing Kaggle CLI..."
        pip install kaggle
    else
        echo "Kaggle is already installed."
    fi
}

setup_kaggle_credentials() {
    # Ensure the Kaggle credentials directory exists
    mkdir -p ~/.kaggle

    # Create the kaggle.json file
    cat > ~/.kaggle/kaggle.json <<EOF
{
  "username": "$KAGGLE_USERNAME",
  "key": "$KAGGLE_API_KEY"
}
EOF

    # Set appropriate permissions
    chmod 600 ~/.kaggle/kaggle.json

    echo "Kaggle credentials have been set up successfully."
}

download_dataset(){
    echo "Downloading the dataset..."
    kaggle datasets download -d "$DATASET_ID" -p "$DATASET_DIR"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Dataset downloaded successfully to $DATASET_DIR."
    else
        echo "Failed to download the dataset. Please check the dataset identifier and Kaggle API setup."
    fi

    # Unzip dataset
    echo "Unzipping dataset..."
    # Check if 'unzip' is installed
    if ! command -v unzip &> /dev/null; then
        echo "Error: 'unzip' is not installed. Installing it now..."
        sudo apt update && sudo apt install -y unzip || { echo "Failed to install 'unzip'. Exiting."; exit 1; }
    fi
    unzip -o "$DATASET_ZIP" -d "$DATASET_DIR"
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
setup_kaggle_credentials
download_dataset

# Check if libgl1 is installed
if dpkg -l | grep -q "libgl1"; then
    echo "libgl1 is already installed."
else
    echo "libgl1 is not installed, updating package list and installing libgl1..."
    sudo apt-get update && sudo apt-get install -y libgl1
fi

create_venv

