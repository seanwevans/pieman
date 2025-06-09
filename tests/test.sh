#!/bin/bash
set -e
# Navigate to repository root
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Build the program
make clean >/dev/null
make >/dev/null

# Run a short training loop
./nnet >/dev/null

# Verify model.bin is produced
if [ ! -f model.bin ]; then
  echo "model.bin was not created" >&2
  exit 1
fi

echo "Test passed: model.bin exists"
