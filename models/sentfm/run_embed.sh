# # run this to avoid issue of stuck at downloading `config_sentence_transformers.json`
# MODEL_SNAPSHOT_DIR=$(find ~/.cache/huggingface/hub -type d -path '*/models--Alibaba-NLP--gte-multilingual-base/snapshots/*' -print -quit)
# cp -v ./config_sentence_transformers.json "$MODEL_SNAPSHOT_DIR/config_sentence_transformers.json"

docker run --gpus all -p 28888:80 \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$HOME/repos/bili-search-algo/data/docker_data":/data \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
    -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:1.8 \
    --huggingface-hub-cache /root/.cache/huggingface/hub \
    --model-id Alibaba-NLP/gte-multilingual-base --dtype float16

# docker run --gpus all -p 28888:80 \
#     -v "$HOME/repos/bili-search-algo/data/docker_data":/data \
#     -e HF_ENDPOINT=https://hf-mirror.com \
#     --pull always \
#     ghcr.io/huggingface/text-embeddings-inference:1.8 \
#     --model-id Alibaba-NLP/gte-multilingual-base --dtype float16

# docker exec -it <container_id> env | grep -i HUGGINGFACE
# docker ps -q --filter "ancestor=ghcr.io/huggingface/text-embeddings-inference:1.8" | xargs -r docker stop