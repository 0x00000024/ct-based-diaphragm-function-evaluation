#!/bin/bash

# The shell exits when a command fails or when the shell tries to expand an unset variable
set -eu

# Key: Plugin name
# Value: Plugin ID
declare -A plugin_hash_table=(
  ["IdeaVim"]="164"
  ["Node.js"]="6098"
  ["ESLint"]="7494"
  [".ignore"]="7495"
  ["Markdown"]="7793"
  ["Toml"]="8195"
  ["google-java-format"]="8527"
  ["TeXiFy IDEA"]="9473"
  [".env files support"]="9525"
  ["CSV"]="10037"
  ["Atom Material Icons"]="10044"
  ["Rainbow Brackets"]="10080"
  ["Kubernetes"]="10485"
  ["Cyan Light Theme"]="12102"
  ["BashSupport Pro"]="13841"
  ["Makefile Language"]="9333"
  ["GitToolBox"]="7499"
  ["Requirements"]="10837"
  ["Selenium UI Testing"]="13691"
  ["Json Parser"]="10650"
  ["Jenkins Control"]="6110"
  ["Indent Rainbow"]="13308"
  ["nginx Support"]="4415"
  ["React snippets"]="10113"
  ["GitHub Copilot"]="17718"
  ["Jira Integration"]="11169"
  ["UUID Generator"]="8320"
  ["Run Configuration for TypeScript"]="10841"
  ["scss-lint"]="7530"
  ["ZooKeeper"]="7364"
  ["React CSS Modules"]="9275"
  ["macOS Keymap"]="13258"
  ["AppleScript Support"]="8149"
)

plugin_tmp_file="plugin_tmp_file.zip"
product_name="$1"
if [[ "${OSTYPE}" == "linux-gnu"* ]]; then
  plugin_path="${HOME}/.local/share/JetBrains/${product_name}"
elif [[ "${OSTYPE}" == "darwin"* ]]; then
  plugin_path="${HOME}/Library/Application Support/JetBrains/${product_name}/plugins"
else
  exit 1
fi

mkdir -p "${plugin_path}"
for name in "${!plugin_hash_table[@]}"; do
  echo "$name - ${plugin_hash_table[$name]}"
  file_path_url=$(curl https://plugins.jetbrains.com/api/plugins/"${plugin_hash_table[$name]}"/updates \
    | jq -r '.[0].file')
  echo "${file_path_url}"
  wget --output-document="${plugin_tmp_file}" "https://plugins.jetbrains.com/files/${file_path_url}"
  unzip -o -d "${plugin_path}" "${plugin_tmp_file}"
done
rm "${plugin_tmp_file}"
