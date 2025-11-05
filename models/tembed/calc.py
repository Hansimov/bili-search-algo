import json

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from sedb import RedisOperator, RocksOperator
from tclogger import logger, log_error
from tfmx import EmbedClient
from typing import Literal, Optional, Union, Any

from configs.envs import REDIS_ENVS
from models.tembed.train import PassageJsonManager

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

    def __init__(self):
        self.init_processors()
        self.init_embed_clients()

    def init_processors(self):
        self.embed_db = Path(__file__).parent / "embeddings.rkdb"
        self.rocks = RocksOperator(configs={"db_path": self.embed_db})
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        self.passage_manager = PassageJsonManager()

    def init_embed_clients(self):
        embed_params = {"api_format": "tei", "res_format": "list2d"}
        self.gte_embed = EmbedClient(
            endpoint="http://localhost:28888/embed", **embed_params
        )
        self.bge_embed = EmbedClient(
            endpoint="http://localhost:28889/embed", **embed_params
        )

    def is_key_exist(self, key: str) -> bool:
        if not key:
            return None
        rc = self.redis.client
        redis_key, redis_field = get_redis_key_field(key)
        if redis_field is None:
            return bool(rc.exists(redis_key))
        else:
            return bool(rc.hexists(redis_key, redis_field))

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

    def calc_hits_embeddings(self, hits: list[dict]):
        pass

    def run(self):
        for query, passage in self.passage_manager.iter_query_passages():
            pass


class EmbeddingModelBenchmarker:
    """Benchmark embedding model with query-passage pairs."""

    def __init__(self):
        pass


def main():
    calculator = EmbeddingPreCalculator()


if __name__ == "__main__":
    main()

    # python -m models.tembed.calc
