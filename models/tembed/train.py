import argparse

from pathlib import Path
from sedb import MongoDocsGenerator, MongoDocsGeneratorArgParser
from sedb import RedisOperator, RocksOperator
from tclogger import StrsType, logger, dict_get, MergedArgParser
from tfmx import EmbedClient
from typing import Literal, Union, Any

from configs.envs import REDIS_ENVS, MONGO_ENVS


TEXT_FIELDS = ["title", "tags", "desc", "owner.name"]
TEXT_TAGS = {
    "title": "标题",
    "tags": "标签",
    "desc": "简介",
    "owner.name": "作者",
}

BASE_NAME = "qwen3_06b"
REDIS_PREFIX = f"bv.emb:"
EMB_FIELD = f"is_emb_{BASE_NAME}"

# sudo mkdir -p /media/data/tembed && sudo chown -R "$USER:$USER" /media/data/tembed
STORAGE_DIR = Path("/media/data/tembed")
ROCKS_DB = f"{BASE_NAME}.rkdb"


class MongoDocConverter:
    def __init__(
        self, text_fields: list[str] = TEXT_FIELDS, text_tags: dict = TEXT_TAGS
    ):
        self.text_fields = text_fields
        self.text_tags = text_tags

    def convert(self, doc: dict) -> str:
        text = ""
        for field in self.text_fields:
            field_str = dict_get(doc, field)
            if field == "desc":
                if not field_str or field_str == "-":
                    continue
            field_str = str(field_str).strip()
            if field_str:
                field_tag = self.text_tags.get(field, field)
                text += f"<{field_tag}>{field_str}</{field_tag}>"
        return text


class EmbedBatcher:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.batch: dict[str, str] = {}
        self.init_processors()
        self.init_embed_client()

    def init_processors(self):
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        self.rocks_db = STORAGE_DIR / ROCKS_DB
        self.rocks = RocksOperator(
            configs={"db_path": self.rocks_db}, connect_cls=self.__class__
        )

    def init_embed_client(self):
        # python -m tfmx.embed_server -t "tei" -m "Qwen/Qwen3-Embedding-0.6B" -p 28887 -b
        self.embedder = EmbedClient(
            endpoint="http://localhost:28887/embed",
            api_format="tei",
            res_format="list2d",
        )

    def should_submit(self) -> bool:
        return len(self.batch) >= self.batch_size

    def append(self, bvid: str, text: str):
        self.batch[bvid] = text
        if self.should_submit():
            self.submit()

    def bvids_to_redis_name_fields(self, bvids: list[str]) -> list[tuple[str, str]]:
        return [(f"{REDIS_PREFIX}{bvid}", EMB_FIELD) for bvid in bvids]

    def redis_name_fields_to_bvids(
        self, name_fields: list[tuple[str, str]]
    ) -> list[str]:
        return [name.removeprefix(REDIS_PREFIX) for name, _ in name_fields]

    def name_fields_to_rocks_keys(
        self, name_fields: list[tuple[str, str]]
    ) -> list[str]:
        return [f"{name}.{field}" for name, field in name_fields]

    def submit(self):
        if not self.batch:
            return
        # get non exist redis name-fields
        bvids = list(self.batch.keys())
        name_fields = self.bvids_to_redis_name_fields(bvids)
        non_exist_name_fields = self.redis.get_non_exist_hashes(name_fields)
        if not non_exist_name_fields:
            return
        # calc embeddings
        non_exist_bvids = self.redis_name_fields_to_bvids(non_exist_name_fields)
        texts = [self.batch[bvid] for bvid in non_exist_bvids]
        embeddings = self.embedder.embed(texts)
        # write to rocks and redis
        rocks_data = {
            bvid: embedding for bvid, embedding in zip(non_exist_bvids, embeddings)
        }
        self.rocks.mset(rocks_data)
        self.redis.set_hashes_exist(non_exist_name_fields)
        # clear batch
        self.batch = {}


class EmbeddingDataCalculator:
    """Pre-calculate embeddings for downstream tasks"""

    def __init__(self):
        self.converter = MongoDocConverter()
        self.batcher = EmbedBatcher(batch_size=100)
        self.init_docs_generator()

    def init_docs_generator(self):
        generator = MongoDocsGenerator()
        generator.init_cli_args(
            ikvs={
                **MONGO_ENVS,
                "mongo_collection": "videos",
                "include_fields": "bvid,title,tags,desc,owner.name",
                # "extra_filters": "u:stat.coin>10",  # ~ 8kw docs
                "extra_filters": "u:stat.coin>250",  # ~ 1kw docs
            }
        )
        generator.init_all_with_cli_args()
        self.generator = generator

    def run(self):
        for doc_idx, doc in enumerate(self.generator.doc_generator()):
            text = self.converter.convert(doc)
            if not text:
                continue
            bvid = dict_get(doc, "bvid")
            self.batcher.append(bvid, text)
        self.batcher.submit()


class TrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-cc", "--calc", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    arg_parser = MergedArgParser(TrainerArgParser, MongoDocsGeneratorArgParser)
    args = arg_parser.parse_args()
    if args.calc:
        calculator = EmbeddingDataCalculator()
        calculator.run()


if __name__ == "__main__":
    main()

    # python -m models.tembed.train -cc -ec -mn 1000
