#!/bin/bash

# Learned Hash Embedder Training and Evaluation Pipeline
# This script trains a neural hash function and evaluates its performance

# Usage:
# cd ~/repos/bili-search-algo
# ./models/tembed/hash.sh

# Color codes
RED='\033[1;31m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No Color

# Exit immediately if any command fails or user interrupts (Ctrl+C)
set -e
trap 'echo ""; echo -e "${RED}>>> Pipeline interrupted by user${NC}"; exit 130' INT

echo_header() {
    local color="${2:-$BLUE}"
    echo ""
    echo -e "${color}=========================================${NC}"
    echo -e "${color}>>> $1${NC}"
    echo -e "${color}=========================================${NC}"
}

# cd ~/repos/bili-search-algo

# ========== Training ==========
echo_header "Training Learned Hash Model"
# Train params: 2048 hash_bits, 1024 hidden_dim, 100000 max_samples, 10 epochs
python -m models.tembed.hasher -m train -hb 2048 -hd 1024 -ms 100000 -ep 10

# Some other params (commented out):
# python -m models.tembed.hasher -m train -hb 2048 -ms 1000000 -ep 50
# python -m models.tembed.hasher -m train -ep 50 -dp 0.2 -w

# ========== Testing ==========
echo_header "Testing Learned Hash Model"
# Test inference speed and output quality
python -m models.tembed.hasher -m test

# ========== Pre-Calc ==========
echo_header "Pre-Calc Learned Hash Embeddings"
# Pre-calc learned hash embeddings (overwrite)
python -m models.tembed.calc -p -l -w

# ========== Benchmark ==========
echo_header "Build Benchmark Ranks (Learned Hash)"
# Build benchmark ranks using learned hash embeddings
python -m models.tembed.calc -r -l

# ========== Scoring ==========
echo_header "Score Benchmark Ranks (Learned Hash)"
# Evaluate performance with scoring metrics
python -m models.tembed.calc -s

echo_header "Pipeline Complete!" "$GREEN"
