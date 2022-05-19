#!/bin/bash

set -eu

conda_base_path="/opt/homebrew/Caskroom/miniconda/base"
conda="${conda_base_path}/bin/conda"
project_name="open3d"
python_version="3.9"
conda_env="${project_name}-${python_version}"
pip="${conda_base_path}/envs/"${project_name}-${python_version}"/bin/pip3"

# Create environment
"${conda}" create --name "${conda_env}" --yes python="${python_version}"

# Install Open3D
"${pip}" install open3d==0.15.1
