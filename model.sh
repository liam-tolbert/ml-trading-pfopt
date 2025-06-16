#!/bin/bash

# Resolve script location to make it portable
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pass all arguments ($@) to the Python script
python "$DIR/predict.py" "$@"