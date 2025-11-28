cd ~/repos/bili-search-algo
# pre-calc embeddings
echo ">>> Pre-Calc Embeddings"
# python -m models.tembed.calc -p -c -wk -sd 1
python -m models.tembed.calc -p -c -wk -sd 1 -bn 2048

# benchmark ranks
echo ">>> Build Benchmark Ranks"
# python -m models.tembed.calc -r -c -j
python -m models.tembed.calc -r -c

# score
echo ">>> Score Benchmark Ranks"
python -m models.tembed.calc -s

# 1024(0.5970), 1536(0.6243), 2048(0.6421), 2560(0.6529), 3072(0.6611)