import argparse
import faiss
import numpy as np

from pathlib import Path
from sedb import MongoDocsGenerator, MongoDocsGeneratorArgParser
from sedb import RedisOperator, RocksOperator
from tclogger import logstr, logger, dict_get, brk, MergedArgParser
from tfmx import EmbedClient
from typing import Literal, Union, Any, Generator

from configs.envs import REDIS_ENVS, MONGO_ENVS


TEXT_FIELDS = ["title", "tags", "desc", "owner.name"]
TEXT_TAGS = {
    "title": "标题",
    "tags": "标签",
    "desc": "简介",
    "owner.name": "作者",
}

BASE_NAME = "qwen3_06b"
EMB_DIM = 1024

EMB_FIELD = f"{BASE_NAME}"
FSS_FIELD = "in_fss"

# sudo mkdir -p /media/data/tembed && sudo chown -R "$USER:$USER" /media/data/tembed
STORAGE_DIR = Path("/media/data/tembed")
ROCKS_DB = f"{BASE_NAME}.rkdb"
FAISS_DB = f"{BASE_NAME}.faiss"

ROCKS_DB_PATH = STORAGE_DIR / ROCKS_DB
FAISS_DB_PATH = STORAGE_DIR / FAISS_DB

REDIS_PREFIX = f"bv.emb:"
REDIS_PT = f"{REDIS_PREFIX}*"


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
        self.rocks = RocksOperator(
            configs={"db_path": ROCKS_DB_PATH}, connect_cls=self.__class__
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

    def clear(self):
        self.batch.clear()

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

    def append(self, bvid: str, text: str):
        self.batch[bvid] = text
        if self.should_submit():
            self.submit()
            self.clear()


class EmbeddingDataCalculator:
    """Pre-calculate embeddings for downstream tasks"""

    def __init__(self):
        self.converter = MongoDocConverter()
        self.batcher = EmbedBatcher()
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


class FaissOperator:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = None

    def init_db(self, dim: int):
        """Initialize Faiss HNSW index optimized for 10M scale data"""
        logger.note(f"> Initializing Faiss HNSW index ...")
        # HNSW parameters for 10M vectors:
        # M: number of connections per layer (32 is good for 10M scale)
        # efConstruction: search depth during construction (higher = better quality but slower)
        M = 32
        efConstruction = 40
        self.db = faiss.IndexHNSWFlat(dim, M)
        self.db.hnsw.efConstruction = efConstruction
        logger.okay(f"  * (dim={dim}, M={M}, efConstruction={efConstruction})")

    def set_search_params(self):
        """Set search parameters for HNSW index"""
        # efSearch: search depth during query (higher = better recall but slower)
        # For 10M scale, 64 is a good balance
        efSearch = 64
        self.db.hnsw.efSearch = efSearch

    def add_embeddings(self, embeddings: np.ndarray):
        """Add embeddings to Faiss index"""
        if self.db is None:
            dim = embeddings.shape[1]
            self.init_db(dim)
        self.db.add(embeddings)

    def save(self):
        if self.db is None:
            logger.warn("× No index to save")
            return
        self.set_search_params()
        logger.note(f"> Saving Faiss index ...")
        faiss.write_index(self.db, str(self.db_path))
        logger.okay(f"  * {self.db_path}")
        logger.mesg(f"  * {self.total_count()} rows")

    def total_count(self) -> int:
        if self.db is None:
            return 0
        return self.db.ntotal


class FaissBuilder:
    """Build Faiss index with Redis and RocksDB"""

    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size
        self.init_processors()

    def init_processors(self):
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        rocks_configs = {"db_path": ROCKS_DB_PATH}
        self.rocks = RocksOperator(configs=rocks_configs, connect_cls=self.__class__)
        self.faiss = FaissOperator(FAISS_DB_PATH)

    def redis_key_to_bvid(self, key: str) -> str:
        """'bv.emb:BV1234567890' -> 'BV1234567890'"""
        return key.removeprefix(REDIS_PREFIX)

    def scan_redis_keys(self) -> Generator[list[str], None, None]:
        """Scan Redis keys, and yield batch_bvids"""
        logger.note("> Scanning Redis for keys in batches...")
        cursor = 0
        batch_bvids = []
        while True:
            cursor, keys = self.redis.client.scan(
                cursor=cursor, match=REDIS_PT, count=1000
            )
            for key in keys:
                bvid = self.redis_key_to_bvid(key)
                batch_bvids.append(bvid)
                if len(batch_bvids) >= self.batch_size:
                    yield batch_bvids
                    batch_bvids = []
            if cursor == 0:
                break
        if batch_bvids:
            yield batch_bvids
            batch_bvids = []
        logger.okay(f"* Finished Redis scan")

    def load_embeddings_from_rocks(
        self, bvids: list[str]
    ) -> tuple[list[str], np.ndarray]:
        """Load embeddings from RocksDB for given bvids.
        Return: (valid_bvids, embs_arr)
        """
        valid_bvids: list[str] = []
        embs: list[np.ndarray] = []
        for bvid in bvids:
            try:
                emb = self.rocks.get(bvid)
                if emb is not None:
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype="float32")
                    valid_bvids.append(bvid)
                    embs.append(emb)
            except Exception as e:
                logger.warn(f"× Failed to get embedding for [{bvid}]: {e}")
        if not embs:
            return [], np.array([])
        embs_arr = np.vstack(embs).astype("float32")
        return valid_bvids, embs_arr

    def bvids_to_faiss_hashes(self, bvids: list[str]) -> list[tuple[str, str]]:
        return [(f"{REDIS_PREFIX}{bvid}", FSS_FIELD) for bvid in bvids]

    def write_to_faiss(self, bvids: list[str], embeddings: np.ndarray):
        """Write embeddings to Faiss index and mark in Redis"""
        if len(bvids) == 0 or len(embeddings) == 0:
            return
        self.faiss.add_embeddings(embeddings)
        faiss_hashes = self.bvids_to_faiss_hashes(bvids)
        self.redis.set_hashes_exist(faiss_hashes)

    def run(self):
        batch_idx = 0
        # scan Redis, load from RocksDB, write to Faiss
        for batch_bvids in self.scan_redis_keys():
            batch_idx += 1
            logger.note(f"> Processing batch {batch_idx} ...")
            valid_bvids, embeddings = self.load_embeddings_from_rocks(batch_bvids)
            self.write_to_faiss(valid_bvids, embeddings)
        self.faiss.save()


class TrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-cc", "--calc", action="store_true")
        self.add_argument("-bf", "--build-faiss", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    arg_parser = MergedArgParser(TrainerArgParser, MongoDocsGeneratorArgParser)
    args = arg_parser.parse_args()

    if args.calc:
        calculator = EmbeddingDataCalculator()
        calculator.run()

    if args.build_faiss:
        builder = FaissBuilder()
        builder.run()


if __name__ == "__main__":
    main()

    # Case: calc and store embeddings
    # python -m models.tembed.train -cc -ec -mn 10000
    # python -m models.tembed.train -cc

    # Case: build faiss HNSW index from Redis and RocksDB
    # python -m models.tembed.train -bf
