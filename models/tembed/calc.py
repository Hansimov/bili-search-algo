import argparse
import json
import numpy as np

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from sedb import RedisOperator, RocksOperator
from tclogger import StrsType, logger, log_error, dict_get, TCLogbar, chars_slice, brk
from tfmx import EmbedClient
from typing import Literal, Optional, Union, Any

from configs.envs import REDIS_ENVS
from models.tembed.sample import PassageJsonManager

EmbedModelType = Literal["gte", "bge", "qwen3_06b"]
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


def dot_sim(vec1: list[float], vec2: list[float]) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def calc_median_ranks(ranks_by_emb: dict[str, list[int]]) -> list[int]:
    """Calculate median ranks across different embedding types."""
    ranks_matrix = np.array(list(ranks_by_emb.values()))
    return np.median(ranks_matrix, axis=0).astype(int).tolist()


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
        self.embed_types = ["gte", "bge", "qwen3_06b"]
        # python -m tfmx.embed_server -t "tei" -m "Alibaba-NLP/gte-multilingual-base" -p 28888 -b
        self.gte_embed = EmbedClient(
            endpoint="http://localhost:28888/embed", **embed_params
        )
        # python -m tfmx.embed_server -t "tei" -m "BAAI/bge-large-zh-v1.5" -p 28889 -b
        self.bge_embed = EmbedClient(
            endpoint="http://localhost:28889/embed", **embed_params
        )
        # python -m tfmx.embed_server -t "tei" -m "Qwen/Qwen3-Embedding-0.6B" -p 28887 -b
        self.qwen3_06b_embed = EmbedClient(
            endpoint="http://localhost:28887/embed", **embed_params
        )
        self.embed_clients: dict[EmbedModelType, EmbedClient] = {
            "gte": self.gte_embed,
            "bge": self.bge_embed,
            "qwen3_06b": self.qwen3_06b_embed,
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
            emb_type: [] for emb_type in self.embed_types
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


class EmbeddingBenchmarkBuilder:
    """Build benchmark with embeddings from multi models."""

    def __init__(self, max_count: int = None):
        self.max_count = max_count
        self.init_processors()

    def init_processors(self):
        self.embed_db = Path(__file__).parent / "embeddings.rkdb"
        self.rocks = RocksOperator(
            configs={"db_path": self.embed_db}, connect_cls=self.__class__
        )
        self.passage_manager = PassageJsonManager()
        self.embed_types = ["gte", "bge", "qwen3_06b"]
        self.benchmark_ranks_path = (
            Path(__file__).parent / "benchmarks" / "benchmark_ranks.json"
        )

    def load_embedding(self, key: str) -> list[float]:
        if not key or not self.is_key_exist(key):
            return None
        rocks_key = get_rocks_key(key)
        value = self.rocks.get(rocks_key)
        if value is None:
            return None
        if isinstance(value, list):
            embedding = value
        else:
            embedding = json.loads(value)
        if not isinstance(embedding, list):
            error_mesg = "× Stored embedding is not a list"
            log_error(error_mesg)
        return floatize_embedding(embedding)

    def load_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        rocks_keys = [get_rocks_key(key) for key in keys]
        values = self.rocks.mget(rocks_keys)
        embeddings: dict[str, list[float]] = {}
        for key, value in zip(keys, values):
            if value is None:
                continue
            try:
                if isinstance(value, list):
                    embedding = value
                else:
                    embedding = json.loads(value)
                embeddings[key] = floatize_embedding(embedding)
            except Exception as e:
                error_mesg = f"× Fail to load embedding of key {brk(key)}: {e}"
                log_error(error_mesg)
        return embeddings

    def get_upper_count(self) -> int:
        max_count = self.max_count
        total_count = self.passage_manager.get_total_count()
        if max_count:
            upper_count = min(max_count, total_count)
        else:
            upper_count = total_count
        return upper_count

    def calc_ranks_for_anchor(
        self,
        hits: list[dict],
        anchor_idx: int,
        field_name: str,
        emb_type: str,
        embeddings: dict[str, list[float]],
    ) -> list[int]:
        """Calculate ranks of all hits relative to anchor_idx hit for given field+emb_type."""
        # get anchor embedding
        anchor_bvid = hits[anchor_idx].get("bvid")
        anchor_key = f"{BV_PREFIX}{anchor_bvid}:{field_name}.{emb_type}"
        anchor_emb = embeddings.get(anchor_key)
        if anchor_emb is None:
            return None

        # calc similarity scores
        scores: list[tuple[int, float]] = []
        for idx, hit in enumerate(hits):
            bvid = hit.get("bvid")
            key = f"{BV_PREFIX}{bvid}:{field_name}.{emb_type}"
            emb = embeddings.get(key)
            if emb is None:
                scores.append((idx, -1))
            else:
                sim = dot_sim(anchor_emb, emb)
                scores.append((idx, sim))

        # sort scores desc, and get ranks
        scores.sort(key=lambda x: x[1], reverse=True)
        idx_to_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(scores)}
        return [idx_to_rank[i] for i in range(len(hits))]

    def calc_hits_ranks(
        self,
        hits: list[dict],
        anchor_indices: list[int] = [0, 1, 4, 9],
        field_names: list[str] = ["title", "tags", "merged"],
    ) -> dict[int, dict]:
        """Calc ranks for hits by anchor indices and field names.

        Example output:
        {
            0: {
                "anchor_idx": 0,
                "title.gte": [1, 3, 2, 5, 4, 7, 6, 9, 8, 10],
                "title.bge": [1, 2, 4, 3, 5, 6, 8, 7, 9, 10],
                "title.qwen3_06b": [1, 4, 3, 2, 6, 5, 7, 9, 8, 10],
                "title.med": [1, 3, 3, 3, 5, 6, 7, 9, 8, 10],
                "tags.gte": [...],
                "tags.bge": [...],
                "tags.qwen3_06b": [...],
                "tags.med": [...],
                "merged.gte": [...],
                "merged.bge": [...],
                "merged.qwen3_06b": [...],
                "merged.med": [...]
            },
            1: {...},
            4: {...},
            9: {...}
        }
        """
        # batch load all embeddings at once
        all_keys: list[str] = []
        for hit in hits:
            bvid = hit.get("bvid")
            if not bvid:
                continue
            for field_name in field_names:
                for emb_type in self.embed_types:
                    key = f"{BV_PREFIX}{bvid}:{field_name}.{emb_type}"
                    all_keys.append(key)
        embeddings = self.load_embeddings(all_keys)

        # calc adjacent-hits-ranks for each anchor_idx
        result: dict[int, dict] = {}
        for anchor_idx in anchor_indices:
            if anchor_idx >= len(hits):
                continue
            anchor_result = {"anchor_idx": anchor_idx}
            for field_name in field_names:
                ranks_by_emb = {}
                for emb_type in self.embed_types:
                    field_emb_key = f"{field_name}.{emb_type}"
                    ranks = self.calc_ranks_for_anchor(
                        hits, anchor_idx, field_name, emb_type, embeddings
                    )
                    if ranks:
                        anchor_result[field_emb_key] = ranks
                        ranks_by_emb[emb_type] = ranks
                field_med_key = f"{field_name}.med"
                anchor_result[field_med_key] = calc_median_ranks(ranks_by_emb)
            result[anchor_idx] = anchor_result
        return result

    def save_benchmark_ranks(self, results: dict):
        self.benchmark_ranks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.benchmark_ranks_path, encoding="utf-8", mode="w") as wf:
            json.dump(results, wf, ensure_ascii=False, indent=2)
        logger.note(f"> Save benchmark ranks to:")
        logger.file(f"  * {self.benchmark_ranks_path}")

    def run(self):
        results: dict[str, dict] = {}
        bar = TCLogbar(total=self.get_upper_count(), desc="* query")
        for idx, (query, passage) in enumerate(
            self.passage_manager.iter_query_passages()
        ):
            if self.max_count and idx >= self.max_count:
                break
            bar.update(increment=1, desc=f"{chars_slice(query,end=10)}")
            hits = passage["hits"]
            results[query] = self.calc_hits_ranks(hits)
        print()
        self.save_benchmark_ranks(results)


class CalculatorArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-n", "--max-count", type=int, default=None)
        self.add_argument("-c", "--calc", action="store_true")
        self.add_argument("-b", "--build", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    args = CalculatorArgParser().args

    if args.calc:
        calculator = EmbeddingPreCalculator(max_count=args.max_count)
        calculator.run()

    if args.build:
        benchmark_builder = EmbeddingBenchmarkBuilder(max_count=args.max_count)
        benchmark_builder.run()


if __name__ == "__main__":
    main()

    # python -m models.tembed.calc -c
    # python -m models.tembed.calc -b -n 10
