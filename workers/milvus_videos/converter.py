import concurrent.futures
import numpy as np

from tclogger import logger, dict_get, Runtimer
from typing import Literal

from configs.envs import FASTTEXT_MERGED_MODEL_PREFIX
from configs.envs import SP_MERGED_MODEL_PREFIX, TOKEN_FREQ_PREFIX
from models.fasttext.run import FasttextModelFrequenizer, FasttextDocVecModelRunner
from models.fasttext.run import FasttextModelRunnerClient
from models.fasttext.preprocess import FasttextModelPreprocessor
from workers.milvus_videos.schema import DOCVEC_DIM, KEEP_COLUMNS

ZEROS_DOCVEC = np.zeros(DOCVEC_DIM).astype(np.float16)


class MongoDocToMilvusDocConverter:
    STAT_KEYS = ["view", "danmaku", "reply", "favorite", "coin", "share", "like"]

    def __init__(
        self, runner_mode: Literal["local", "remote"] = "local", max_workers: int = 32
    ):
        self.runner_mode = runner_mode
        self.max_workers = max_workers
        self.init_docvec_runner()

    def init_docvec_runner(self):
        if self.runner_mode == "local":
            with Runtimer():
                frequenizer = FasttextModelFrequenizer(
                    token_freq_prefix=TOKEN_FREQ_PREFIX
                )
                preprocessor = FasttextModelPreprocessor(
                    tokenizer_prefix=SP_MERGED_MODEL_PREFIX
                )
                self.docvec_runner = FasttextDocVecModelRunner(
                    model_prefix=FASTTEXT_MERGED_MODEL_PREFIX,
                    frequenizer=frequenizer,
                    preprocessor=preprocessor,
                    restrict_vocab=15000,
                    vector_weighted=True,
                    verbose=True,
                )
                self.docvec_runner.load_model()
        else:
            self.docvec_runner = FasttextModelRunnerClient(model_class="doc").runner

    def stats_to_list(self, stats: dict) -> list[int]:
        return [int(stats.get(key, 0)) for key in self.STAT_KEYS]

    def text_to_vec(
        self, text: str, text_type: Literal["query", "sample"] = "sample"
    ) -> np.ndarray:
        try:
            if text_type == "query":
                vec = self.docvec_runner.calc_stretch_query_vector(text)
            else:
                vec = self.docvec_runner.calc_stretch_sample_vector(text)
        except Exception as e:
            logger.warn(f"× <{text}>: {e}")
        if self.runner_mode == "local":
            return vec.astype(np.float16)
        else:
            return np.array(vec).astype(np.float16)

    def get_kept_dict(self, doc: dict) -> dict:
        return {col.replace(".", "_"): dict_get(doc, col) for col in KEEP_COLUMNS}

    def convert(self, doc: dict) -> dict:
        milvus_doc = {
            "bvid": doc["bvid"],
            **self.get_kept_dict(doc),
            "stats_arr": self.stats_to_list(doc["stat"]),
            "title_vec": self.text_to_vec(doc["title"]),
            "title_status": 1,
            "title_tags_owner_vec": ZEROS_DOCVEC,
            "title_tags_owner_desc_vec": ZEROS_DOCVEC,
            "tags_status": 0,
        }
        return milvus_doc

    def convert_batch(self, docs: list[dict]) -> list[dict]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [executor.submit(self.convert, doc) for doc in docs]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results


def test_converter():
    import json
    from configs.envs import LOG_ROOT

    converter = MongoDocToMilvusDocConverter("remote")
    text_vecs = {}
    queries = ["动画素材", "电影"]
    for query in queries:
        text_vecs[query] = converter.text_to_vec(query, text_type="query").tolist()

    json_path = LOG_ROOT / "text_vecs.json"
    if json_path.exists():
        with open(json_path, "r") as rf:
            json_data = json.load(rf)
    else:
        json_data = {}
    json_data.update(text_vecs)
    with open(json_path, "w") as f:
        json.dump(json_data, f, ensure_ascii=False)


if __name__ == "__main__":
    test_converter()

    # python -m workers.milvus_videos.converter
