#!/bin/bash

# args
region=${1:-"music_dance"}

# vars
model_prefix="fasttext_${region}"
dataset_name="video_texts_${region}"
vocab_file="video_texts_${region}_nt"

# constants
dataset_root="parquets"
epochs=1
batch_size=20000
max_final_vocab=300000

cmd="python -m models.fasttext.train -m \"${model_prefix}\" -dr \"${dataset_root}\" -dn \"${dataset_name}\" -vf \"${vocab_file}\" -ep ${epochs} -bs ${batch_size} -mv ${max_final_vocab}"
echo "$cmd"
eval "$cmd"

# chmod +x models/fasttext/train.sh
# ./models/fasttext/train.sh
# ./models/fasttext/train.sh other_game