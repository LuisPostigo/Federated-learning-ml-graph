#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RUN=0
DOCKER=0
MODEL_NAME="${MODEL_NAME:-cnn}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --run)
      RUN=1
      shift
      ;;
    --docker|-d)
      DOCKER=1
      RUN=1
      shift
      ;;
    --model|-m)
      MODEL_NAME="$2"
      shift 2
      ;;
    --help|-h)
      cat << EOF
Usage: $0 [OPTIONS]

Setup script for Federated Learning ML Graph.

Options:
  --run, -r              Run the federated learning after setup
  --docker, -d           Run using Docker Compose (requires --run)
  --model MODEL, -m MODEL  Select model to use: cnn, cifar10_cnn, or resnet (default: cnn)
  --help, -h             Show this help message

Examples:
  $0 --run                          # Setup venv and run locally with default CNN model
  $0 --run --model cifar10_cnn       # Run locally with CIFAR10 CNN model
  $0 --docker --model resnet         # Run with Docker using ResNet model
  $0 --model cnn --docker            # Run with Docker using CNN model

Environment Variables:
  MODEL_NAME            Model to use (overridden by --model flag)
EOF
      exit 0
      ;;
    *)
      # Unknown option, might be arguments for main.py
      break
      ;;
  esac
done

# Validate model name
case "$MODEL_NAME" in
  cnn|cifar10_cnn|cifar10cnn|resnet)
    ;;
  *)
    echo "[WARN] Unknown model name: $MODEL_NAME. Using default 'cnn'."
    echo "[INFO] Available models: cnn, cifar10_cnn, resnet"
    MODEL_NAME="cnn"
    ;;
esac

export MODEL_NAME

if [[ $DOCKER -eq 1 ]]; then
  # Docker execution
  echo "[OK] Using Docker Compose with model: $MODEL_NAME"
  echo "[OK] Changing to docker directory..."
  cd docker || { echo "[ERR] docker/ directory not found."; exit 1; }
  
  echo "[OK] Starting Docker Compose services..."
  docker-compose up --build
else
  # Local execution
  [ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  python -m pip install --upgrade pip
  pip install -r requirements.txt

  echo "[OK] venv: $VENV_DIR (python: $(python -V))"
  echo "[OK] Model: $MODEL_NAME"
  echo "[OK] To keep it active after: source $VENV_DIR/bin/activate"

  if [[ $RUN -eq 1 ]]; then
    [[ -f main.py ]] || { echo "[ERR] main.py not found."; exit 1; }
    echo "[OK] Running: python main.py $* (with MODEL_NAME=$MODEL_NAME)"
    python -m main "$@"
  fi
fi
