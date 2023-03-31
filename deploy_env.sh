#!/bin/bash

set -eu

conda="/opt/homebrew/Caskroom/miniconda/base/bin/conda"
project_name="ct"
python_version="3.10"
conda_env="${project_name}-${python_version}"

# Create environment
"${conda}" create --name "${conda_env}" --yes python="${python_version}"

# Install Pydicom
"${conda}" install --name "${conda_env}" --yes --channel conda-forge pydicom

# Install OpenCV
"${conda}" install --name "${conda_env}" --yes --channel conda-forge opencv=4.5.5

# Install pandas
"${conda}" install --name "${conda_env}" --yes --channel anaconda pandas

# Install Plotly
"${conda}" install --name "${conda_env}" --yes --channel plotly plotly

# Install Colorama
"${conda}" install --name "${conda_env}" --yes --channel anaconda colorama

# Install Nptyping
"${conda}" install --name "${conda_env}" --yes --channel conda-forge nptyping

# Install imageio
"${conda}" install --name "${conda_env}" --yes --channel conda-forge imageio
