#!/bin/bash

# Learned Hash Embedder Training and Evaluation Pipeline
# This script trains a neural hash function and evaluates its performance

# Usage:
# cd ~/repos/bili-search-algo
# SYNTAX : ./models/tembed/hash.sh [-hd hidden_dim] [-hb hash_bits] [-ms max_samples] [-ep epochs] [-n max_count] [-rm] [-w]
# Example: ./models/tembed/hash.sh -hd 2048 -hb 2048 -ms 100000 -ep 10 -n 1000 -rm -w

# color codes
RED='\033[1;31m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No Color

# exit immediately if any command fails or user interrupts (Ctrl+C)
echo_interrupt() {
    set -e
    trap 'echo ""; echo -e "${RED}>>> Pipeline interrupted by user${NC}"; exit 130' INT
}

echo_header() {
    local color="${2:-$BLUE}"
    echo ""
    echo -e "${color}=========================================${NC}"
    echo -e "${color}>>> $1${NC}"
    echo -e "${color}=========================================${NC}"
}

echo_params() {
    echo -e "${MAGENTA}> Training Parameters:${NC}"
    echo -e "  * ${CYAN}hidden_dim${NC}   : ${GREEN}$HIDDEN_DIM${NC}"
    echo -e "  * ${CYAN}hash_bits${NC}    : ${GREEN}$HASH_BITS${NC}"
    echo -e "  * ${CYAN}max_samples${NC}  : ${GREEN}$MAX_SAMPLES${NC}"
    echo -e "  * ${CYAN}epochs${NC}       : ${GREEN}$EPOCHS${NC}"
    echo -e "  * ${CYAN}max_count${NC}    : ${GREEN}$MAX_COUNT${NC}"
    echo -e "  * ${CYAN}remove_weights${NC}: ${GREEN}$REMOVE_WEIGHTS${NC}"
    echo -e "  * ${CYAN}overwrite${NC}    : ${GREEN}$OVERWRITE${NC}"
}

echo_complete() {
    echo_header "Pipeline Complete!" "$GREEN"
}

run_cmd() {
    local CMD="$1"
    echo -e "${MAGENTA}> Running:${NC} ${YELLOW}$CMD${NC}"
    eval $CMD
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -hd|--hidden-dim)
                HIDDEN_DIM="$2"
                shift 2
                ;;
            -hb|--hash-bits)
                HASH_BITS="$2"
                shift 2
                ;;
            -ms|--max-samples)
                MAX_SAMPLES="$2"
                shift 2
                ;;
            -ep|--epochs)
                EPOCHS="$2"
                shift 2
                ;;
            -n|--max-count)
                MAX_COUNT="$2"
                shift 2
                ;;
            -rm|--remove-weights)
                REMOVE_WEIGHTS="true"
                shift
                ;;
            -w|--overwrite)
                OVERWRITE="true"
                shift
                ;;
            *)
                echo -e "${RED}Unknown parameter: $1${NC}"
                echo "Usage: $0 [-hd hidden_dim] [-hb hash_bits] [-ms max_samples] [-ep epochs] [-n max_count] [-w] [-rm]"
                exit 1
                ;;
        esac
    done
}

# ====== Run Pipeline ======

# [Interrupt]
echo_interrupt

# [Args]
HASH_BITS=""
HIDDEN_DIM=""
MAX_SAMPLES=""
EPOCHS=""
MAX_COUNT=""
REMOVE_WEIGHTS="false"
OVERWRITE="false"
parse_args "$@"

# [Working]
cd ~/repos/bili-search-algo

# [Train]
echo_header "Training Learned Hash Model"
echo_params
CMD="python -m models.tembed.hasher -m train"
[[ -n "$HIDDEN_DIM" ]] && CMD="$CMD -hd $HIDDEN_DIM"
[[ -n "$HASH_BITS" ]] && CMD="$CMD -hb $HASH_BITS"
[[ -n "$MAX_SAMPLES" ]] && CMD="$CMD -ms $MAX_SAMPLES"
[[ -n "$EPOCHS" ]] && CMD="$CMD -ep $EPOCHS"
[[ "$REMOVE_WEIGHTS" == "true" ]] && CMD="$CMD -rm"
run_cmd "$CMD"

# [Test]
echo_header "Testing Learned Hash Model"
run_cmd "python -m models.tembed.hasher -m test"

# [Pre-Calc]
echo_header "Pre-Calc Learned Hash Embeddings"
CALC_CMD="python -m models.tembed.calc -p -l"
[[ -n "$MAX_COUNT" ]] && CALC_CMD="$CALC_CMD -n $MAX_COUNT"
[[ "$OVERWRITE" == "true" ]] && CALC_CMD="$CALC_CMD -w"
run_cmd "$CALC_CMD"

# [Benchmark]
echo_header "Build Benchmark Ranks (Learned Hash)"
BENCH_CMD="python -m models.tembed.calc -r -l"
[[ -n "$MAX_COUNT" ]] && BENCH_CMD="$BENCH_CMD -n $MAX_COUNT"
run_cmd "$BENCH_CMD"

# [Score]
echo_header "Score Benchmark Ranks (Learned Hash)"
SCORE_CMD="python -m models.tembed.calc -s"
[[ -n "$MAX_COUNT" ]] && SCORE_CMD="$SCORE_CMD -n $MAX_COUNT"
run_cmd "$SCORE_CMD"

# [Complete]
echo_complete
