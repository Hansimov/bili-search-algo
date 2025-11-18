cd ~/repos/bili-search-algo
# pre-calc embeddings
echo ">>> Pre-Calc Embeddings"
# python -m models.tembed.calc -p -c -wk -sd 1
python -m models.tembed.calc -p -c -wk -sd 1 -bn 1536

# benchmark ranks
echo ">>> Build Benchmark Ranks"
# python -m models.tembed.calc -r -c -j
python -m models.tembed.calc -r -c

# score
echo ">>> Score Benchmark Ranks"
python -m models.tembed.calc -s