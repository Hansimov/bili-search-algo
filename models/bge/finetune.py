"""
FlagOpen/FlagEmbedding: Retrieval and Retrieval-augmented LLMs
* https://github.com/FlagOpen/FlagEmbedding/tree/master

```sh
# mannually install dependencies: torch, torchvision, flash-attn
pip install torchvision
cd ~/downloads
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install .[finetune]
```

"""


class BgeModelFinetuner:
    pass
