#!/usr/bin/env zsh

# The shell exits when a command fails or when the shell tries to expand an unset variable
set -eu

service ssh restart

/etc/init.d/xrdp restart

zsh
