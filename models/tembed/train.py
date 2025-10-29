import argparse
import json
import random
import polars as pl

from copy import deepcopy
from pathlib import Path
from sedb import ElasticOperator
from tclogger import logger, logstr, brk, chars_slice
from tclogger import dict_get, dict_pop, dict_flatten
from tclogger import TCLogbar
from typing import Literal

from models.word.eng import get_dump_path
from configs.envs import SECRETS


class QuerySampler:
    """Sample queries for query-passage pairs."""

    def __init__(self):
        self.save_path = Path(__file__).parent / "keyword_samples.csv"

    def load_csv(self, docs_count: int = None, lang: Literal["zh", "en"] = None):
        logger.note(f"> Loading csv from:")
        freq_path = get_dump_path(docs_count=docs_count, lang=lang)
        self.df = pl.read_csv(freq_path)
        logger.okay(f"  * {freq_path}")
        logger.line(self.df, indent=4)

    def random_samples(self, num: int = 100, seed: int = None):
        logger.note(f"> Picking random samples:", end=" ")
        seed = seed or 0
        random.seed(seed)

        df_sorted = self.df.sort("doc_freq", descending=True)
        interval_size = df_sorted.height / num
        idxs = [
            (random.randint(int(i * interval_size), int((i + 1) * interval_size) - 1))
            for i in range(num)
        ]
        samples = df_sorted[idxs]
        seed_str = logstr.mesg(f"(seed={seed})")
        logger.okay(f"{len(samples)} {seed_str}")
        top_samples = samples.top_k(15, by="doc_freq")
        logger.line(top_samples, indent=4)
        self.samples = samples

    def save_csv(self):
        logger.note(f"> Saving samples to csv:")
        self.samples.write_csv(self.save_path)
        logger.okay(f"  * {self.save_path}")

    def run(self):
        self.load_csv(docs_count=770000000, lang="zh")
        self.random_samples(num=10000, seed=2)
        self.save_csv()


class ElasticResultsParser:
    """Parse results from elasticsearch."""

    def parse_source(self, source: dict) -> dict:
        res = deepcopy(source)
        for key in ["_index", "_id"]:
            dict_pop(res, key)
        for keys in ["owner", "stat"]:
            dict_flatten(res, keys=keys, in_replace=True, expand_sub=True)
        return res

    def parse_hit(self, hit: dict) -> dict:
        source = dict_get(hit, "_source", {})
        parsed_source = self.parse_source(source)
        extra_info = {
            "elastic_score": dict_get(hit, "_score", 0),
        }
        hit_info = {
            **parsed_source,
            **extra_info,
        }
        return hit_info

    def parse(self, es_resp: dict) -> dict:
        """Example output:
        {
            "hits": [
                {
                    "bvid": "...",
                    "title": "...",
                    "tags": "...",
                    "desc": "...",
                    "owner.name": "...",
                    "stat.favorite": ...,
                    "elastic_score": ...,
                    "elastic_score_rank": ...,
                },
                ...
            ],
            "total_hits": ...,
            "return_hits": ...,
        }
        """
        hits_dict = dict_get(dict(es_resp), "hits", {})
        hits = hits_dict.get("hits", [])
        hits_info = []
        for idx, hit in enumerate(hits):
            hit_info = self.parse_hit(hit)
            hit_info["elastic_score_rank"] = idx + 1
            hits_info.append(hit_info)
        res = {
            "hits": hits_info,
            "return_hits": len(hits_info),
            "total_hits": hits_dict.get("total", {}).get("value", -1),
        }
        return res


class QuerySearcher:
    """Search results of queries from elasticsearch."""

    def __init__(self):
        self.configs = SECRETS["elastic_dev"]
        self.index_name = "bili_videos_dev5"
        self.es = ElasticOperator(configs=self.configs, connect_cls=self.__class__)
        self.parser = ElasticResultsParser()

    def query_to_es_dict(self, query: str) -> dict:
        query_dict = {
            "query": {
                "es_tok_query_string": {
                    "query": query,
                    "type": "cross_fields",
                    "fields": [
                        "title.words^3",
                        "tags.words^2.5",
                        "owner.name.words^2",
                        "desc.words^0.1",
                    ],
                    "max_freq": 1000000,
                    "min_kept_tokens_count": 2,
                    "min_kept_tokens_ratio": -1,
                }
            },
            "_source": ["bvid", "title", "tags", "owner.name", "desc", "stat.favorite"],
            "size": 10,
            "track_total_hits": True,
        }
        return query_dict

    def search(self, query: str) -> dict:
        query_dict = self.query_to_es_dict(query)
        es_resp = self.es.client.search(index=self.index_name, body=query_dict)
        res = self.parser.parse(es_resp)
        return res


class PassagesGenerator:
    """Generate data for text-embedding examination."""

    def __init__(self):
        self.samples_path = Path(__file__).parent / "keyword_samples.csv"
        self.passages_path = Path(__file__).parent / "query_passages.json"
        self.searcher = QuerySearcher()

    def load_samples(self) -> pl.DataFrame:
        logger.note(f"> Loading samples from:")
        self.df = pl.read_csv(self.samples_path)
        logger.okay(f"  * {self.samples_path}")
        # logger.line(self.df, indent=4)

    def load_passages(self) -> dict:
        if not self.passages_path.exists():
            self.passages_path.parent.mkdir(parents=True, exist_ok=True)
            passages = {}
        else:
            with open(self.passages_path, encoding="utf-8") as f:
                passages = json.load(f)
        self.passages = passages

    def save_result(self, query: str, result: dict):
        self.passages[query] = result
        with open(self.passages_path, encoding="utf-8", mode="w") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)

    def run(self):
        self.load_samples()
        self.load_passages()
        bar = TCLogbar(total=len(self.df), desc="* keyword")
        for idx, (word, doc_freq) in enumerate(
            zip(self.df["word"], self.df["doc_freq"])
        ):
            bar.update(increment=1, desc=f"{chars_slice(word,end=10)}")
            if word in self.passages:
                continue
            try:
                res = self.searcher.search(word)
            except Exception as e:
                logger.warn(f"× Error query: {word}")
                # break
                continue
            res.update({"query": word, "doc_freq": doc_freq, "query_idx": idx})
            self.save_result(word, res)


class TembedTrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-s", "--sample", action="store_true")
        self.add_argument("-g", "--generate", action="store_true")
        self.args = self.parse_args()


def main():
    arg_parser = TembedTrainerArgParser()
    args = arg_parser.args

    if args.sample:
        sampler = QuerySampler()
        sampler.run()

    if args.generate:
        generator = PassagesGenerator()
        generator.run()


if __name__ == "__main__":
    main()

    # python -m models.tembed.train -s
    # python -m models.tembed.train -g
