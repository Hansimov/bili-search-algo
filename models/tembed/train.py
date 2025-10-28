import argparse
import random
import polars as pl

from pathlib import Path
from sedb import ElasticOperator
from tclogger import logger, logstr, brk, dict_to_str
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


class TextEmbedExamineDataGenerator:
    """Generate examine data for text embedding model tuning."""

    def __init__(self):
        self.configs = SECRETS["elastic_dev"]
        self.index_name = "bili_videos_dev5"
        self.es = ElasticOperator(configs=self.configs, connect_cls=self.__class__)

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
            "_source": ["title", "tags", "owner.name", "desc", "stat.favorite"],
            "size": 10,
        }
        return query_dict

    def search(self, query: str) -> dict:
        query_dict = self.query_to_es_dict(query)
        res = self.es.client.search(index=self.index_name, body=query_dict)
        return res


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
        sampler.load_csv(docs_count=770000000, lang="zh")
        sampler.random_samples(num=10000, seed=2)
        sampler.save_csv()

    if args.generate:
        generator = TextEmbedExamineDataGenerator()
        queries = ["机器学习"]
        for query in queries:
            logger.note(f"> Query: {logstr.mesg(brk(query))}")
            res = generator.search(query)
            logger.line(dict_to_str(res), indent=4)


if __name__ == "__main__":
    main()

    # python -m models.tembed.train -s
    # python -m models.tembed.train -g
