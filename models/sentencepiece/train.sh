#!/bin/bash

prefix="sp_703m"
zhwiki_prefix="sp_wiki_8m_400k"
zhwiki_vocab_size=400000

# LINK: models/sentencepiece/filter.py
declare -A groups=(
[1]="cine_movie douga_anime tech_sports"
[2]="music_dance fashion_ent know_info"
[3]="daily_life other_life"
[4]="mobile_game other_game"
[l]="latest"
[w]="zhwiki"
[x]="test"
)

if [[ -z "$1" ]]; then
echo "> Usage: $0 "
echo "* Valid groups: ${!groups[@]}"
exit 1
fi

group="$1"
regions_str="${groups[$group]}"

if [[ -z "$regions_str" ]]; then
echo "× Unknown group: $group"
echo "* Valid groups: ${!groups[@]}"
exit 1
fi

echo "Training regions for group [$group]: $regions_str"

if [[ "$group" == "w" ]]; then
    cmd=(python -m models.sentencepiece.train -m "${zhwiki_prefix}" -db zhwiki -cn pages -bs 1000 -vs ${zhwiki_vocab_size} -fd -e)
    echo "${cmd[@]}"
    "${cmd[@]}"
    exit 0
fi

read -r -a regions <<< "$regions_str"
for region in "${regions[@]}"; do
cmd=(python -m models.sentencepiece.train -m "${prefix}_${region}" -fg "$region" -av -fd -e)
echo "${cmd[@]}"
"${cmd[@]}"
done

# chmod +x models/sentencepiece/train.sh
# ./models/sentencepiece/train.sh 1
# ./models/sentencepiece/train.sh 2
# ./models/sentencepiece/train.sh 3
# ./models/sentencepiece/train.sh 4
# ./models/sentencepiece/train.sh l