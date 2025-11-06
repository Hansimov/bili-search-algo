import argparse
import json

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from sedb import RedisOperator, RocksOperator
from tclogger import StrsType, log_error, dict_get, TCLogbar, chars_slice
from tfmx import EmbedClient
from typing import Literal, Optional, Union, Any

from configs.envs import REDIS_ENVS
from models.tembed.sample import PassageJsonManager

EmbedModelType = Literal["gte", "bge"]
EmbedKeyType = Literal["qr", "bv"]


@dataclass(frozen=True)
class KeyParts:
    raw_key: str
    key_type: EmbedKeyType
    rocks_key: str
    redis_key: str
    redis_field: Optional[str]


QR_PREFIX = "tembed.qr:"
BV_PREFIX = "tembed.bv:"


class EmbeddingKeyConverter:
    """Convert keys mappings for embeddings in RocksDB and Redis."""

    def is_qr_key(self, key: str) -> bool:
        return key.startswith(QR_PREFIX)

    def is_bv_key(self, key: str) -> bool:
        return key.startswith(BV_PREFIX)

    def to_qr_keyparts(self, key: str) -> KeyParts:
        try:
            _, query = key.split(":", 1)
        except Exception as e:
            err_mesg = f"× Invalid qr:<query>: {key}"
            log_error(err_mesg)

        return KeyParts(
            raw_key=key,
            key_type="qr",
            rocks_key=key,
            redis_key=key,
            redis_field=None,
        )

    def to_bv_keyparts(self, key: str) -> KeyParts:
        try:
            _, bvid, field = key.split(":", 2)
        except Exception as e:
            err_mesg = f"× Invalid bv:<bvid>:<field>: {key}"
            log_error(err_mesg)

        return KeyParts(
            raw_key=key,
            key_type="bv",
            rocks_key=key,
            redis_key=f"{BV_PREFIX}{bvid}",
            redis_field=field,
        )

    def resolve(self, key: str) -> KeyParts:
        if self.is_qr_key(key):
            return self.to_qr_keyparts(key)
        if self.is_bv_key(key):
            return self.to_bv_keyparts(key)
        err_mesg = f"× Invalid key: {key}"
        log_error(err_mesg)


key_converter = EmbeddingKeyConverter()


def resolve_key(key: str) -> KeyParts:
    return key_converter.resolve(key)


def get_redis_key_field(key: str) -> tuple[str, Union[str, None]]:
    parts = key_converter.resolve(key)
    return parts.redis_key, parts.redis_field


def get_rocks_key(key: str) -> str:
    parts = key_converter.resolve(key)
    return parts.rocks_key


def floatize_embedding(embedding: list[Any]) -> list[float]:
    return [float(v) for v in embedding]


def stringify_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding, ensure_ascii=False, separators=(",", ":"))


class EmbeddingPreCalculator:
    """Pre-calculate embeddings for query-passage pairs."""

    def __init__(self, max_count: int = None):
        self.max_count = max_count
        self.init_processors()
        self.init_embed_clients()

    def init_processors(self):
        self.embed_db = Path(__file__).parent / "embeddings.rkdb"
        self.rocks = RocksOperator(
            configs={"db_path": self.embed_db}, connect_cls=self.__class__
        )
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        self.passage_manager = PassageJsonManager()

    def init_embed_clients(self):
        embed_params = {"api_format": "tei", "res_format": "list2d"}
        self.embed_types = ["gte", "bge"]
        # python -m tfmx.embed_server -t "tei" -p 28888 -m "Alibaba-NLP/gte-multilingual-base" -id "Alibaba-NLP--gte-multilingual-base" -b
        self.gte_embed = EmbedClient(
            endpoint="http://localhost:28888/embed", **embed_params
        )
        # python -m tfmx.embed_server -t "tei" -p 28889 -m "BAAI/bge-large-zh-v1.5" -id "BAAI--bge-large-zh-v1.5" -b
        self.bge_embed = EmbedClient(
            endpoint="http://localhost:28889/embed", **embed_params
        )
        self.embed_clients: dict[EmbedModelType, EmbedClient] = {
            "gte": self.gte_embed,
            "bge": self.bge_embed,
        }

    def is_key_exist(self, key: str) -> bool:
        if not key:
            return None
        rc = self.redis.client
        redis_key, redis_field = get_redis_key_field(key)
        if redis_field is None:
            return bool(rc.exists(redis_key))
        else:
            return bool(rc.hexists(redis_key, redis_field))

    def is_keys_exist(self, keys: list[str]) -> list[bool]:
        """Batch check if keys exist in Redis."""
        if not keys:
            return []

        rc = self.redis.client
        pipeline = rc.pipeline()

        # Group keys by their redis_key and field
        key_checks = []  # List of (key, redis_key, redis_field)
        for key in keys:
            redis_key, redis_field = get_redis_key_field(key)
            key_checks.append((key, redis_key, redis_field))
            if redis_field is None:
                pipeline.exists(redis_key)
            else:
                pipeline.hexists(redis_key, redis_field)

        # Execute pipeline and get results
        results = pipeline.execute()

        # Convert results to list of bools
        return [bool(result) for result in results]

    def set_key_exist(self, key: str):
        if not key:
            return
        rc = self.redis.client
        redis_key, redis_field = get_redis_key_field(key)
        if redis_field is None:
            rc.set(redis_key, 1)
        else:
            rc.hset(redis_key, redis_field, 1)

    def set_keys_exist(self, keys: list[str]):
        if not keys:
            return
        rc = self.redis.client
        pipeline = rc.pipeline()
        redis_keys: set[str] = set()
        redis_fields: dict[str, set[str]] = defaultdict(set)
        for key in keys:
            redis_key, redis_field = get_redis_key_field(key)
            if redis_field is None:
                redis_keys.add(redis_key)
            else:
                redis_fields[redis_key].add(redis_field)
        for redis_key in redis_keys:
            pipeline.set(redis_key, 1)
        for redis_key, redis_fields in redis_fields.items():
            for redis_field in redis_fields:
                pipeline.hset(redis_key, redis_field, 1)
        pipeline.execute()

    def del_key_exist(self, key: str):
        if not key:
            return
        rc = self.redis.client
        redis_key, redis_field = get_redis_key_field(key)
        if redis_field is None:
            rc.delete(redis_key)
            return
        rc.hdel(redis_key, redis_field)
        if rc.hlen(redis_key) == 0:
            rc.delete(redis_key)

    def load_embedding(self, key: str) -> list[float]:
        if not key or not self.is_key_exist(key):
            return None
        rocks_key = get_rocks_key(key)
        value = self.rocks.get(rocks_key)
        if value is None:
            return None
        embedding = json.loads(value)
        if not isinstance(embedding, list):
            error_mesg = "× Stored embedding is not a list"
            log_error(error_mesg)
        return floatize_embedding(embedding)

    def save_embedding(self, key: str, embedding: list[float]):
        if not key or not embedding:
            return
        rocks_key = get_rocks_key(key)
        self.rocks.set(rocks_key, stringify_embedding(embedding))
        self.set_key_exist(key)

    def save_embeddings(self, key_embeddings: dict[str, list[float]]):
        if not key_embeddings:
            return
        self.rocks.mset(key_embeddings)

    def embed_text_by_type(
        self, texts: StrsType, embed_type: EmbedModelType
    ) -> list[list[float]]:
        if not texts:
            return []
        embed_client = self.embed_clients.get(embed_type)
        if embed_client:
            embeddings = embed_client.embed(texts)
        else:
            log_error(f"× Unknown embed_type: {embed_type}")
        return embeddings

    def embed_key_text_pairs(
        self, key_text_pairs: list[tuple[str, str]], emb_type: EmbedModelType
    ) -> dict[str, list[float]]:
        if not key_text_pairs:
            return {}
        bv_keys, field_texts = zip(*key_text_pairs)
        embeddings = self.embed_text_by_type(field_texts, emb_type)
        return {bv_key: embedding for bv_key, embedding in zip(bv_keys, embeddings)}

    def calc_qr_embedding(self, query: str):
        if not query:
            return
        qr_keys = []
        qr_embeddings: dict[str, list[float]] = {}
        for emb_type in self.embed_types:
            qr_key = f"{QR_PREFIX}{query}.{emb_type}"
            if self.is_key_exist(qr_key):
                # logger.mesg(f"→ qr_key exists: {qr_key}")
                continue
            qr_keys.append(qr_key)
            qr_embeddings[qr_key] = self.embed_text_by_type(query, emb_type)[0]
        self.save_embeddings(qr_embeddings)
        self.set_keys_exist(qr_keys)

    def get_field_texts(self, hit: dict) -> dict[str, str]:
        title = hit.get("title", "")
        tags = hit.get("tags", "")
        owner_name = dict_get(hit, "owner.name", "")
        desc = hit.get("desc", "")
        merged = f"{title} {tags} {owner_name} {desc}".strip()
        return {
            "title": title,
            "tags": tags,
            "merged": merged,
        }

    def calc_bv_embedding(self, hit: dict):
        if not hit:
            return
        self.calc_hits_embeddings([hit])

    def calc_hits_embeddings(self, hits: list[dict]):
        """Batch calc embeddings of multi hits"""
        if not hits:
            return

        # collect all bv_keys
        all_bv_keys = []
        bv_keys_info = {}  # Map bv_key to (field_text, emb_type)
        for hit in hits:
            bvid = hit.get("bvid")
            if not bvid:
                continue
            field_texts = self.get_field_texts(hit)
            for field_name, field_text in field_texts.items():
                if not field_text:
                    continue
                for emb_type in self.embed_types:
                    bv_key = f"{BV_PREFIX}{bvid}:{field_name}.{emb_type}"
                    all_bv_keys.append(bv_key)
                    bv_keys_info[bv_key] = (field_text, emb_type)

        # check bv_keys exist
        keys_exist = self.is_keys_exist(all_bv_keys)
        texts_to_embed: dict[EmbedModelType, list[tuple[str, str]]] = {
            "gte": [],
            "bge": [],
        }
        for bv_key, is_exists in zip(all_bv_keys, keys_exist):
            if is_exists:
                # logger.mesg(f"→ bv_key exists: {bv_key}")
                continue
            field_text, emb_type = bv_keys_info[bv_key]
            texts_to_embed[emb_type].append((bv_key, field_text))

        # batch embed all texts in parallel
        all_embeddings: dict[str, list[float]] = {}
        all_keys: list[str] = []
        with ThreadPoolExecutor(max_workers=len(self.embed_types)) as executor:
            futures = [
                executor.submit(
                    self.embed_key_text_pairs, texts_to_embed[emb_type], emb_type
                )
                for emb_type in self.embed_types
            ]
            for future in as_completed(futures):
                embed_res = future.result()
                all_embeddings.update(embed_res)
                all_keys.extend(embed_res.keys())

        # batch save all embeddings
        if all_embeddings:
            self.save_embeddings(all_embeddings)
            self.set_keys_exist(all_keys)

    def get_upper_count(self) -> int:
        max_count = self.max_count
        total_count = self.passage_manager.get_total_count()
        if max_count:
            upper_count = min(max_count, total_count)
        else:
            upper_count = total_count
        return upper_count

    def run(self):
        bar = TCLogbar(total=self.get_upper_count(), desc="* query")
        for idx, (query, passage) in enumerate(
            self.passage_manager.iter_query_passages()
        ):
            if self.max_count and idx >= self.max_count:
                break
            # logger.note(query)
            bar.update(increment=1, desc=f"{chars_slice(query,end=10)}")
            self.calc_qr_embedding(query)
            hits = passage["hits"]
            self.calc_hits_embeddings(hits)


class EmbeddingModelBenchmarker:
    """Benchmark embedding model with query-passage pairs."""

    def __init__(self):
        pass


class CalculatorArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-n", "--max-count", type=int, default=None)
        self.args, _ = self.parse_known_args()


def main():
    args = CalculatorArgParser().args
    calculator = EmbeddingPreCalculator(max_count=args.max_count)
    calculator.run()


if __name__ == "__main__":
    main()

    # python -m models.tembed.calc -n 100
