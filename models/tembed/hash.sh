#!/bin/bash

# Learned Hash Embedder Training and Evaluation Pipeline
# This script trains a neural hash function and evaluates its performance

# Usage:
# cd ~/repos/bili-search-algo
# SYNTAX : ./models/tembed/hash.sh [-p preset] [-a arch] [-bz batch_size] [-hd hidden_dim] [-hb hash_bits] [-ms max_samples] [-ep epochs] [-n max_count] [-rm] [-w]

# Examples:
# ./models/tembed/hash.sh -hd 2048 -hb 2048 -ms 100000 -ep 10 -n 1000 -rm -w
# ./models/tembed/hash.sh -p resmlp_small -a resmlp -hb 2048 -ms 100000 -bz 256 -ep 10 -n 500 -rm -w

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
    echo -e "  * ${CYAN}preset${NC}      : ${GREEN}${PRESET}${NC}"
    echo -e "  * ${CYAN}arch${NC}        : ${GREEN}${ARCH}${NC}"
    echo -e "  * ${CYAN}batch_size${NC}  : ${GREEN}${BATCH_SIZE}${NC}"
    echo -e "  * ${CYAN}hidden_dim${NC}  : ${GREEN}$HIDDEN_DIM${NC}"
    echo -e "  * ${CYAN}hash_bits${NC}   : ${GREEN}$HASH_BITS${NC}"
    echo -e "  * ${CYAN}max_samples${NC} : ${GREEN}$MAX_SAMPLES${NC}"
    echo -e "  * ${CYAN}epochs${NC}      : ${GREEN}$EPOCHS${NC}"
    echo -e "  * ${CYAN}max_count${NC}   : ${GREEN}$MAX_COUNT${NC}"
    echo -e "  * ${CYAN}rm_weights${NC}  : ${GREEN}$REMOVE_WEIGHTS${NC}"
    echo -e "  * ${CYAN}overwrite${NC}   : ${GREEN}$OVERWRITE${NC}"
}

echo_complete() {
    echo_header "Pipeline Complete!" "$GREEN"
}

run_cmd() {
    local -a cmd=("$@")
    echo -e "${MAGENTA}> Running:${NC} ${YELLOW}${cmd[*]}${NC}"
    "${cmd[@]}"
}

arg() {
    # append arg with non-empty value
    # Usage: arg <cmdArray> "--flag" "$value"
    local array_name="$1"
    local flag="$2"
    local value="$3"
    [[ -n "$value" ]] || return 0
    eval "$array_name+=(\"$flag\" \"$value\")"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--preset)
                PRESET="$2"
                shift 2
                ;;
            -a|--arch)
                ARCH="$2"
                shift 2
                ;;
            -bz|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
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
                echo "Usage: $0 [-p preset] [-a arch] [-bz batch_size] [-hd hidden_dim] [-hb hash_bits] [-ms max_samples] [-ep epochs] [-n max_count] [-w] [-rm]"
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
PRESET=""
ARCH=""
BATCH_SIZE=""
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
cmd=(python -m models.tembed.hasher -m train)
arg cmd --preset "$PRESET"
arg cmd --arch "$ARCH"
arg cmd -bz "$BATCH_SIZE"
arg cmd -hd "$HIDDEN_DIM"
arg cmd -hb "$HASH_BITS"
arg cmd -ms "$MAX_SAMPLES"
arg cmd -ep "$EPOCHS"
[[ "$REMOVE_WEIGHTS" == "true" ]] && cmd+=(-rm)
run_cmd "${cmd[@]}"

# [Test]
echo_header "Testing Learned Hash Model"
run_cmd python -m models.tembed.hasher -m test

# [Pre-Calc]
echo_header "Pre-Calc Learned Hash Embeddings"
cmd=(python -m models.tembed.calc -p -l)
arg cmd -n "$MAX_COUNT"
[[ "$OVERWRITE" == "true" ]] && cmd+=(-w)
run_cmd "${cmd[@]}"

# [Benchmark]
echo_header "Build Benchmark Ranks (Learned Hash)"
cmd=(python -m models.tembed.calc -r -l)
arg cmd -n "$MAX_COUNT"
run_cmd "${cmd[@]}"

# [Score]
echo_header "Score Benchmark Ranks (Learned Hash)"
cmd=(python -m models.tembed.calc -s)
arg cmd -n "$MAX_COUNT"
run_cmd "${cmd[@]}"

# [Complete]
echo_complete

# [Log]
python -m tclogger.tmux -n 1 -i "hash.sh"