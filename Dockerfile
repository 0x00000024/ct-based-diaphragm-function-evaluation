FROM ubuntu:22.04

WORKDIR /root

# No interactive frontend during docker build
ARG DEBIAN_FRONTEND=noninteractive

# Set password for superuser
ARG ROOT_PASSWORD
RUN printf "%s\n%s\n\n" "${ROOT_PASSWORD}" "${ROOT_PASSWORD}" | \
    passwd

ADD sshd_config /etc/ssh/sshd_config

# Configure authorized_keys for OpenSSH
ARG PUBLIC_KEY
RUN mkdir --parents "${HOME}/.ssh" && \
    echo "${PUBLIC_KEY}" >"${HOME}/.ssh/authorized_keys"

# Update
RUN apt-get update

########################################################################################################################
# Version Control Systems
# Version control systems and related utilities.
########################################################################################################################

# Git - fast, scalable, distributed revision control system
RUN apt-get install --assume-yes git

########################################################################################################################
# Web Software
# Web servers, browsers, proxies, download tools etc.
########################################################################################################################

# Install cURL - command line tool for transferring data with URL syntax
RUN apt-get install --assume-yes curl

# Install wget - Internet file retriever
RUN apt-get install --assume-yes wget

# Install Google Chrome - Web Browser
RUN chrome_file="$(mktemp --suffix=.deb)" && \
    chrome_url="https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" && \
    wget --progress=dot:giga --output-document="${chrome_file}" "${chrome_url}" && \
    apt install --assume-yes "${chrome_file}" && \
    rm "${chrome_file}"

########################################################################################################################
# Shells
# Command shells. Friendly user interfaces for beginners.
########################################################################################################################

# Z shell - Shell with lots of features
RUN apt-get install --assume-yes zsh

# Change login shell for the superuser
RUN chsh --shell "$(which zsh)" root

# Install Oh My Zsh
ARG OH_MY_ZSH_URL="https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh"
RUN sh -c "$(curl --location ${OH_MY_ZSH_URL}) --unattended"
SHELL ["zsh", "-c"]

########################################################################################################################
# Network
# Daemons and clients to connect your system to the world.
########################################################################################################################

# Install OpenSSH server - secure shell (SSH) server, for secure access from remote machines
RUN apt-get install --assume-yes openssh-server

# Install rsync - fast, versatile, remote (and local) file-copying tool
RUN apt-get install --assume-yes rsync

# Install Xrdp - Remote Desktop Protocol (RDP) server
RUN apt-get install --assume-yes xrdp
RUN sed -i.bak '/fi/a #xrdp multiple users configuration \nstartxfce4\n' /etc/xrdp/startwm.sh

########################################################################################################################
# Editors
# Software to edit files. Programming environments.
########################################################################################################################

# Install NeoVim
RUN apt-get install --assume-yes neovim

########################################################################################################################
# Development
# Development utilities, compilers, development environments, libraries, etc.
########################################################################################################################

# Install g++ - GNU C++ compiler
RUN apt-get install --assume-yes g++

## Install Ninja - a small build system with a focus on speed
#RUN apt-get install --assume-yes ninja-build

# Install CMake - Cross-platform make
RUN apt-get install --assume-yes cmake

# Install GNU Debugger
RUN gdb_file="$(mktemp --suffix=.deb)" && \
    gdb_url="http://launchpadlibrarian.net/580107051/gdb_11.1-0ubuntu3_amd64.deb" && \
    wget --progress=dot:giga --output-document="${gdb_file}" "${gdb_url}" && \
    apt install --assume-yes "${gdb_file}" && \
    rm "${gdb_file}"

## Install pkg-config - Manage compile and link flags for libraries
#RUN apt-get install --assume-yes pkg-config

## Install Autoconf - Automatic configure script builder
#RUN apt-get install --assume-yes autoconf

########################################################################################################################
# Library development
# Libraries necessary for developers to write programs that use them.
########################################################################################################################

# Install libcgal-dev - C++ library for computational geometry (development files)
RUN apt-get install --assume-yes libcgal-dev

# Install libcgal-qt5-dev - C++ library for computational geometry (development files, support for Qt5)
RUN apt-get install --assume-yes libcgal-qt5-dev

########################################################################################################################
# Mathematics
# Math software.
########################################################################################################################

# Install gmsh - A 3D mesh generator with a high quality mesh generation algorithm
RUN apt-get install --assume-yes gmsh

# Install glpk-utils - Utilities for the GLPK linear programming solver
RUN apt-get install --assume-yes glpk-utils

########################################################################################################################
# Utilities
# Utilities for file/disk manipulation, backup and archive tools, system monitoring, input systems, etc.
########################################################################################################################

# Install jq - Lightweight and flexible command-line JSON processor
RUN apt-get install --assume-yes jq

# Install XZ Utils - XZ-format compression utilities
RUN apt-get install --assume-yes xz-utils

# Install unzip - De-archiver for .zip files
RUN apt-get install --assume-yes unzip

# Install Glances - Curses-based monitoring tool
RUN apt-get install --assume-yes glances

########################################################################################################################
# X Window System software
# X servers, libraries, fonts, window managers, terminal emulators and many related applications.
########################################################################################################################

# Install Xfce - Meta-package for the Xfce Lightweight Desktop Environment
RUN apt-get install --assume-yes xfce4

# Install Xfce Terminal - Xfce terminal emulator
RUN apt-get install --assume-yes xfce4-terminal && \
    update-alternatives --set x-terminal-emulator /usr/bin/xfce4-terminal.wrapper





########################################################################################################################
# Install CLion - A Cross-Platform IDE for C and C++ by JetBrains
ARG clion_version="2022.1.2"
RUN clion_file="$(mktemp --suffix=.tar.gz)" && \
    clion_url="https://download-cdn.jetbrains.com/cpp/CLion-${clion_version}.tar.gz" && \
    wget --progress=dot:giga --output-document="${clion_file}" "${clion_url}" && \
    tar --extract --ungzip --file "${clion_file}" --directory=/opt && \
    rm "${clion_file}"

# Install CLion plugins
ADD jetbrains_plugin_downloader.sh /jetbrains_plugin_downloader.sh
RUN bash /jetbrains_plugin_downloader.sh "CLion${clion_version:0:6}"

## Install vcpkg - Open source C/C++ dependency manager
#RUN vcpkg_path="/opt/vcpkg" && \
#    git clone --depth=1 https://github.com/microsoft/vcpkg "${vcpkg_path}" && \
#    cd "${vcpkg_path}" && \
#	./bootstrap-vcpkg.sh && \
#	export VCPKG_FORCE_SYSTEM_BINARIES=1 && \
#	./vcpkg integrate install && \
#	./vcpkg integrate zsh && \
#	echo 'export VCPKG_FORCE_SYSTEM_BINARIES=1' >>"${HOME}/.zshrc" && \
#	ln --symbolic "${vcpkg_path}/vcpkg" /usr/local/bin/vcpkg

#RUN apt-get install --assume-yes build-essential
#RUN apt-get install --assume-yes autoconf-archive
#RUN apt-get install --assume-yes texinfo
#RUN apt-get install --assume-yes yasm
#RUN vcpkg update

## Install CGAL - C++ library for geometric algorithms
#RUN vcpkg install cgal


ARG cgal_path="/usr/local/lib/cgal"
ARG boost_path="/usr/local/lib/boost"
ARG qt_path="/usr/local/lib/qt"
RUN cgal_file="$(mktemp --suffix=.tar.xz)" && \
    cgal_url="https://github.com/CGAL/cgal/releases/download/v5.4/CGAL-5.4.tar.xz" && \
    wget --progress=dot:giga --output-document="${cgal_file}" "${cgal_url}" && \
    mkdir --parents "${cgal_path}" && \
    tar --extract --directory="${cgal_path}" --strip-components=1 --file="${cgal_file}" && \
    rm "${cgal_file}"

RUN boost_file="$(mktemp --suffix=.tar.gz)" && \
    boost_url="https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz" && \
    wget --progress=dot:giga --output-document="${boost_file}" "${boost_url}" && \
    mkdir --parents "${boost_path}" && \
    tar --extract --ungzip --directory="${boost_path}" --strip-components=1 --file="${boost_file}" && \
    rm "${boost_file}"

RUN cgal_example_path="${HOME}/cgal-example" && \
    mkdir --parents "${cgal_example_path}" && \
    cp "${cgal_path}/examples/Mesh_3/mesh_implicit_sphere.cpp" "${cgal_example_path}/sphere.cpp" && \
    cd "${cgal_example_path}" && \
    bash "${cgal_path}/scripts/cgal_create_CMakeLists" && \
    cmake -DCGAL_DIR="${cgal_path}" -DBOOST_ROOT="${boost_path}" -DCMAKE_BUILD_TYPE=Release .





RUN ssh-keygen -N "" -f ~/.ssh/id_rsa

EXPOSE 22
EXPOSE 5900

ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT ["zsh", "/entrypoint.sh"]
