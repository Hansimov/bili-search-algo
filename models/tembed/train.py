import random
import polars as pl

from pathlib import Path
from sedb import ElasticOperator
from tclogger import logger, logstr
from typing import Literal

from models.word.eng import get_dump_path


class QuerySampler:
    """Sample queries for query-passage pairs."""

    def load_csv(self, docs_count: int = None, lang: Literal["zh", "en"] = None):
        logger.note(f"> Loading csv from:")
        csv_path = get_dump_path(docs_count=docs_count, lang=lang)
        self.df = pl.read_csv(csv_path)
        logger.okay(f"  * {csv_path}")
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
        save_path = Path(__file__).parent / "keyword_samples.csv"
        self.samples.write_csv(save_path)
        logger.okay(f"  * {save_path}")


class TextEmbedExamineDataGenerator:
    """Generate examine data for text embedding model tuning."""

    pass


def main():
    sampler = QuerySampler()
    sampler.load_csv(docs_count=770000000, lang="zh")
    sampler.random_samples(num=10000, seed=2)
    sampler.save_csv()


if __name__ == "__main__":
    main()

    # python -m models.tembed.train
