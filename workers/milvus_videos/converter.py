import numpy as np

from tclogger import logger, dict_get, Runtimer, dict_get
from typing import Literal

from configs.envs import FASTTEXT_MERGED_MODEL_PREFIX
from configs.envs import SP_MERGED_MODEL_PREFIX, TOKEN_FREQ_PREFIX
from models.fasttext.run import FasttextDocVecModelRunner, FasttextModelRunnerClient
from models.fasttext.preprocess import FasttextModelPreprocessor
from workers.milvus_videos.constants import FIELD_VECTOR_WEIGHTS, STAT_KEYS
from workers.milvus_videos.schema import DOCVEC_DIM, KEEP_COLUMNS

ZEROS_DOCVEC = np.zeros(DOCVEC_DIM).astype(np.float16)


class MongoDocToMilvusDocConverter:
    def __init__(
        self, runner_mode: Literal["local", "remote"] = "local", max_workers: int = 32
    ):
        self.runner_mode = runner_mode
        self.max_workers = max_workers
        self.init_docvec_runner()

    def init_docvec_runner(self):
        if self.runner_mode == "local":
            with Runtimer():
                logger.note(f"> Init local docvec runner:")
                preprocessor = FasttextModelPreprocessor(
                    tokenizer_prefix=SP_MERGED_MODEL_PREFIX,
                    token_freq_prefix=TOKEN_FREQ_PREFIX,
                )
                self.docvec_runner = FasttextDocVecModelRunner(
                    model_prefix=FASTTEXT_MERGED_MODEL_PREFIX,
                    preprocessor=preprocessor,
                    restrict_vocab=15000,
                    vector_weighted=True,
                    verbose=True,
                )
                self.docvec_runner.load_model()
        else:
            logger.note(f"> Use remote docvec runner")
            self.docvec_runner = FasttextModelRunnerClient(model_class="doc").runner

    def stats_to_list(self, stats: dict) -> list[int]:
        return [int(stats.get(key, 0)) for key in STAT_KEYS]

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
        vectors_by_field = {
            field: self.text_to_vec(dict_get(doc, field), text_type="sample")
            for field in FIELD_VECTOR_WEIGHTS.keys()
        }
        title_tags_owner_vec = sum(
            FIELD_VECTOR_WEIGHTS[field] * vectors_by_field[field]
            for field in ["title", "tags", "owner.name"]
        )
        title_tags_owner_desc_vec = sum(
            FIELD_VECTOR_WEIGHTS[field] * vectors_by_field[field]
            for field in ["title", "tags", "owner.name", "desc"]
        )
        milvus_doc = {
            "bvid": doc["bvid"],
            **self.get_kept_dict(doc),
            "stats_arr": self.stats_to_list(doc["stat"]),
            "title_tags_owner_vec": title_tags_owner_vec,
            "title_tags_owner_desc_vec": title_tags_owner_desc_vec,
            "vectorized_status": 1,
        }
        return milvus_doc

    def convert_batch(self, docs: list[dict]) -> list[dict]:
        results = []
        # with concurrent.futures.ThreadPoolExecutor(
        #     max_workers=self.max_workers
        # ) as executor:
        #     futures = [executor.submit(self.convert, doc) for doc in docs]
        #     for future in concurrent.futures.as_completed(futures):
        #         results.append(future.result())
        for doc in docs:
            results.append(self.convert(doc))
        return results


def test_converter():
    import json
    from tclogger import dict_to_str
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
        f.write(dict_to_str(json_data, add_quotes=True, is_colored=False))


if __name__ == "__main__":
    test_converter()

    # python -m workers.milvus_videos.converter
