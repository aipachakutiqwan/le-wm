#!/usr/bin/env bash
# Bootstrap script for GCP L4 instance.
# Run this once after SSH-ing into the VM (see cloud/GCP.md for VM setup steps).
#
# Usage:
#   bash bootstrap_gcp.sh
#
# The script will prompt for any credentials not already set as env vars.
# Set them in advance to run non-interactively:
#   export GITHUB_USERNAME=...
#   export GITHUB_PAT=...
#   export WANDB_API_KEY=...
#   bash bootstrap_gcp.sh

set -euo pipefail

REPO_URL="https://github.com/aipachakutiqwan/le-wm.git"
REPO_DIR="$HOME/le-wm"
STABLEWM_HOME="$HOME/stablewm-home"

# ── helpers ───────────────────────────────────────────────────────────────────
_prompt() {
    local var="$1" prompt="$2" secret="${3:-false}"
    if [[ -z "${!var:-}" ]]; then
        if [[ "$secret" == "true" ]]; then
            while true; do
                read -rsp "$prompt: " "$var"; echo
                local val="${!var}"
                local masked="${val:0:4}****${val: -4}"
                echo "    Preview: $masked"
                read -rp "    Looks correct? [Y/n]: " confirm
                [[ "${confirm:-Y}" =~ ^[Yy]$ ]] && break
                echo "    Re-enter..."
            done
        else
            read -rp "$prompt: " "$var"
        fi
        export "$var"
    fi
}

_add_env() {
    local key="$1" val="$2"
    if ! grep -q "export $key=" ~/.bashrc; then
        echo "export $key=\"$val\"" >> ~/.bashrc
    fi
}

# ── 1. verify nvidia-smi ──────────────────────────────────────────────────────
echo "==> Checking nvidia-smi..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "    WARNING: nvidia-smi not found. Is the NVIDIA driver installed?"
    echo "    Fix: sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot"
elif ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null; then
    echo "    WARNING: nvidia-smi found but driver not communicating with kernel."
    echo "    Fix: sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot"
else
    echo "    GPU OK."
fi

# ── 2. install docker if missing ──────────────────────────────────────────────
echo "==> Checking Docker..."
if ! command -v docker &>/dev/null; then
    echo "    Docker not found, installing..."
    sudo apt-get update -qq
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    sudo systemctl restart docker
    echo "    Docker installed. NOTE: log out and back in for group membership to take effect."
else
    echo "    Docker already installed: $(docker --version)"
fi

# ── 3. verify gpu access inside docker ───────────────────────────────────────
echo "==> Verifying GPU access inside Docker..."
if docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi; then
    echo "    GPU inside Docker OK."
else
    echo "    WARNING: GPU not accessible inside Docker."
    echo "    Try: sudo systemctl restart docker"
    echo "    If that fails, check the NVIDIA Container Toolkit installation."
fi

# ── 4. prompt for credentials ─────────────────────────────────────────────────
_prompt GITHUB_USERNAME   "Your GitHub username (for auth)"
_prompt GITHUB_PAT        "GitHub PAT (read:packages scope)" true
_prompt WANDB_API_KEY     "W&B API key" true

# ── 5. clone repo ─────────────────────────────────────────────────────────────
echo "==> Cloning repo..."
if [[ -d "$REPO_DIR" ]]; then
    echo "    $REPO_DIR already exists, pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

# ── 6. create dataset directory ───────────────────────────────────────────────
echo "==> Creating STABLEWM_HOME at $STABLEWM_HOME..."
mkdir -p "$STABLEWM_HOME"

# ── 7. persist env vars to ~/.bashrc ──────────────────────────────────────────
echo "==> Persisting env vars to ~/.bashrc..."
if ! grep -q "# le-wm environment" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# le-wm environment — added by cloud/bootstrap_gcp.sh" >> ~/.bashrc
fi
_add_env STABLEWM_HOME    "$STABLEWM_HOME"    # path to dataset storage
_add_env GITHUB_USERNAME  "$GITHUB_USERNAME"  # your GitHub username for auth
_add_env GITHUB_PAT       "$GITHUB_PAT"       # GitHub PAT with read:packages scope
_add_env WANDB_API_KEY    "$WANDB_API_KEY"    # Weights & Biases API key

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Bootstrap complete."
echo "    Repo:          $REPO_DIR"
echo "    STABLEWM_HOME: $STABLEWM_HOME"
echo ""
echo "    Next steps:"
echo "      1. source ~/.bashrc"
echo "      2. Pull Docker image: cd $REPO_DIR && ./devtools.py login && ./devtools.py pull_docker <tag>"
echo "      3. Upload dataset:    gsutil cp gs://<bucket>/<file>.h5 $STABLEWM_HOME/"
echo "      4. Start training:    cd $REPO_DIR && ./devtools.py run_local <tag> --data tworoom --setup cloud_a10g"
