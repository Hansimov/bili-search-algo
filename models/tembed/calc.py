import json

from dataclasses import dataclass
from pathlib import Path
from sedb import RedisOperator, RocksOperator
from tclogger import logger
from tfmx import EmbedClient
from typing import Literal, Optional, Union, Any

from configs.envs import REDIS_ENVS
from models.tembed.train import PassageJsonManager

EmbedModelType = Literal["gte", "bge"]
EmbedKeyType = Literal["query", "bv"]


@dataclass(frozen=True)
class KeyParts:
    raw_key: str
    key_type: EmbedKeyType
    rocks_key: str
    redis_key: str
    redis_field: Optional[str]


QUERY_PREFIX = "query:"
BV_PREFIX = "bv:"


def log_error(mesg: str):
    logger.warn(mesg)
    raise ValueError(mesg)


class EmbeddingKeyConverter:
    """Convert keys mappings for embeddings in RocksDB and Redis."""

    def is_query_key(self, key: str) -> bool:
        return key.startswith(QUERY_PREFIX)

    def is_bv_key(self, key: str) -> bool:
        return key.startswith(BV_PREFIX)

    def to_query_keyparts(self, key: str) -> KeyParts:
        try:
            _, query = key.split(":", 1)
        except Exception as e:
            err_mesg = f"× Invalid query:<query>: {key}"
            log_error(err_mesg)

        return KeyParts(
            raw_key=key,
            key_type="query",
            rocks_key=key,
            redis_key=key,
            redis_field=None,
        )

    def to_bv_keyparts(self, key: str) -> KeyParts:
        try:
            bv, bvid, field = key.split(":", 2)
        except Exception as e:
            err_mesg = f"× Invalid bv:<bvid>:<field>: {key}"
            log_error(err_mesg)

        return KeyParts(
            raw_key=key,
            key_type="bv",
            rocks_key=key,
            redis_key=f"{bv}:{bvid}",
            redis_field=field,
        )

    def resolve(self, key: str) -> KeyParts:
        if self.is_query_key(key):
            return self.to_query_keyparts(key)
        if self.is_bv_key(key):
            return self.to_bv_keyparts(key)
        err_mesg = f"× Invalid key: {key}"
        log_error(err_mesg)


key_converter = EmbeddingKeyConverter()


def resolve_key(key: str) -> KeyParts:
    return key_converter.resolve(key)


def resolve_embedding(embedding: list[Any]) -> list[float]:
    return [float(v) for v in embedding]


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
        parts = resolve_key(key)
        rc = self.redis.client
        if parts.redis_field is None:
            return bool(rc.exists(parts.redis_key))
        return bool(rc.hexists(parts.redis_key, parts.redis_field))

    def set_key_exist(self, key: str) -> bool:
        parts = resolve_key(key)
        rc = self.redis.client
        if parts.redis_field is None:
            rc.set(parts.redis_key, "1")
            return True
        rc.hset(parts.redis_key, parts.redis_field, 1)
        return True

    def del_key_exist(self, key: str) -> bool:
        parts = resolve_key(key)
        rc = self.redis.client
        if parts.redis_field is None:
            rc.delete(parts.redis_key)
            return True
        rc.hdel(parts.redis_key, parts.redis_field)
        if rc.hlen(parts.redis_key) == 0:
            rc.delete(parts.redis_key)
        return True

    def load_embedding(self, key: str) -> list[float]:
        if not self.is_key_exist(key):
            return None
        parts = resolve_key(key)
        value = self.rocks.get(parts.rocks_key)
        if value is None:
            return None
        embedding = json.loads(value)
        if not isinstance(embedding, list):
            error_mesg = "× Stored embedding is not a list"
            log_error(error_mesg)
        return [float(v) for v in embedding]

    def save_embedding(self, key: str, embedding: list[float]):
        parts = resolve_key(key)
        sanitized = [float(v) for v in embedding]
        embedding_str = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
        self.rocks.db[parts.rocks_key] = embedding_str
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
