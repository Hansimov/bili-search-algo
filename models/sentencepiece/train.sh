#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python -m models.sentencepiece.workflow train "$@"

# Examples:
# chmod +x models/sentencepiece/train.sh
# ./models/sentencepiece/train.sh 1 -p sp_908m -av -fd -e
# ./models/sentencepiece/train.sh all -j 5 -p sp_908m -av -fd -e
# ./models/sentencepiece/train.sh all -j 6 -iw -p sp_908m -av -fd -e