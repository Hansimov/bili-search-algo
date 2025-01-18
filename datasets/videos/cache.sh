#!/bin/bash

regions1=("cine_movie" "daily_life" "douga_anime" "fashion_ent" "know_info")
regions2=("mobile_game" "music_dance" "other_game" "other_life" "tech_sports")

if [ "$1" = "1" ]; then
    regions=("${regions1[@]}")
    echo "Regions: ${regions[@]}"
elif [ "$1" = "2" ]; then
    regions=("${regions2[@]}")
    echo "Regions: ${regions[@]}"
else
    echo "× Invalid arg : $1"
    echo "  * valid args: 1 | 2"
    exit 1
fi

for region in "${regions[@]}"; do
    cmd="python -m datasets.videos.cache -fd -fg ${region}"
    echo "$cmd"
    eval "$cmd"
done

# chmod +x datasets/videos/cache.sh
# ./datasets/videos/cache.sh 1
# ./datasets/videos/cache.sh 2