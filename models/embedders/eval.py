from pathlib import Path
from tclogger import logger, dict_to_str

from mteb import MTEB
from C_MTEB import *

from models.fastembed.embed import FastEmbedder

"""
See for more details:
- https://github.com/FlagOpen/FlagEmbedding/blob/master/research/C_MTEB/README.md

Note that this script requires installing specific version of `datasets`:
```sh
pip install datasets==2.16.0
```

Download necessary datasets mannually if the auto-download fails:
- https://huggingface.co/datasets/C-MTEB/T2Retrieval
- https://huggingface.co/docs/datasets/en/cache

Solution 1: (huggingface-cli)

```sh
pip install huggingface_hub
HF_ENDPOINT=https://alpha.hf-mirror.com huggingface-cli download "C-MTEB/DuRetrieval" --repo-type dataset
```

Solution 2: (hfd with aria2)

```sh
sudo apt-get install aria2 jq
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
~/downloads/hfd.sh "C-MTEB/DuRetrieval" --dataset
```

Use backup endpoint (https://alpha.hf-mirror.com) if the original mirror is slow.
"""


class CMtebEvaluator:
    def __init__(self):
        self.output_root = Path(__file__).parent / "cmteb_results"

    def eval(self, embedder, model_name: str):
        output_folder = self.output_root / model_name.replace("/", "_")
        evaluation = MTEB(tasks=["DuRetrieval"])
        results = evaluation.run(embedder, output_folder=output_folder)
        logger.mesg(dict_to_str(results))


def eval_fastembeder():
    model_name = "BAAI/bge-small-zh-v1.5"
    embedder = FastEmbedder(model_name)
    embedder.load_model()
    evaluator = CMtebEvaluator()
    evaluator.eval(embedder, model_name)


if __name__ == "__main__":
    eval_fastembeder()

    # python -m models.embedders.mteb.eval
