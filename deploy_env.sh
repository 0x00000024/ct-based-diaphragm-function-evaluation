#!/bin/bash

set -eu

conda="/opt/homebrew/Caskroom/miniconda/base/bin/conda"
project_name="ct"
python_version="3.10"
conda_env="${project_name}-${python_version}"

# Create environment
"${conda}" create --name "${conda_env}" --yes python="${python_version}"

# Install Pydicom
"${conda}" install --name "${conda_env}" --yes --channel conda-forge pydicom=2.3.0

# Install OpenCV
"${conda}" install --name "${conda_env}" --yes --channel conda-forge opencv=4.5.5

# Install pandas
"${conda}" install --name "${conda_env}" --yes --channel anaconda pandas=1.4.2

# Install Matplotlib
"${conda}" install --name "${conda_env}" --yes --channel conda-forge matplotlib=3.5.2

# Install Plotly
"${conda}" install --name "${conda_env}" --yes --channel plotly plotly=5.7.0

# Install Colorama
"${conda}" install --name "${conda_env}" --yes --channel anaconda colorama=0.4.4

# Install Nptyping
"${conda}" install --name "${conda_env}" --yes --channel conda-forge nptyping=2.0.1

# Install imageio
"${conda}" install --name "${conda_env}" --yes --channel conda-forge imageio=2.19.0

# Install pyntcloud
"${conda}" install --name "${conda_env}" --yes --channel conda-forge pyntcloud=0.3.1
