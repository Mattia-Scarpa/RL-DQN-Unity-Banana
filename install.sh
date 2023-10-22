#!/bin/bash

# Name of the environment
ENV_NAME="P1_BananaNavigation_Env"

# Create a new conda environment with Python 3.6
conda create --name $ENV_NAME python=3.6 -y

# Activate the environment
conda activate $ENV_NAME

# Install the libraries with specified versions
conda install numpy=1.19.5 -y
conda install pytorch=1.10.2 -c pytorch -y
conda install matplotlib=3.3.4 -y
pip install unityagents==0.4.0  # Using pip for this as unityagents might not be available through standard conda channels

echo "All packages have been installed in the $ENV_NAME environment."
