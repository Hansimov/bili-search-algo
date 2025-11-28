import argparse
import json
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from sedb import RedisOperator, RocksOperator
from tclogger import StrsType, logger, log_error, dict_get, TCLogbar, chars_slice, brk
from tclogger import shell_cmd
from tfmx import EmbedClient, floats_to_bits, bits_to_hash, dot_sim, hash_sim
from typing import Literal, Optional, Union, Any

from configs.envs import REDIS_ENVS
from models.tembed.sample import PassageJsonManager
from models.tembed.lsh import LSHConverter

EmbedModelType = Literal["gte", "bge", "qwen3_06b"]
EmbedKeyType = Literal["qr", "bv"]
EmbedValType = Union[list[float], list[int], str]


@dataclass(frozen=True)
class KeyParts:
    raw_key: str
    key_type: EmbedKeyType
    rocks_key: str
    redis_key: str
    redis_field: Optional[str]


QR_PREFIX = "tembed.qr:"
BV_PREFIX = "tembed.bv:"

PARENT_DIR = Path(__file__).parent

EMB_HOST = "localhost"
# EMB_HOST = "11.24.11.122"


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


def calc_median_ranks(ranks_by_emb: dict[str, list[int]]) -> list[int]:
    """Calculate median ranks across different embedding types."""
    ranks_matrix = np.array(list(ranks_by_emb.values()))
    return np.median(ranks_matrix, axis=0).astype(int).tolist()


class EmbedInterface(ABC):
    @abstractmethod
    def embed(self, texts: StrsType) -> list[EmbedValType]:
        """Return embeddings (2d-list) of texts."""
        pass


class RandomEmbedder(EmbedInterface):
    """Generate random embeddings with given dimension."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, texts: StrsType) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for _ in texts:
            emb = np.random.rand(self.dim).tolist()
            embeddings.append(emb)
        return embeddings


class HashEmbedder(EmbedInterface):
    """Convert float-type gte embeddings to hash-type embeddings."""

    def __init__(self):
        self.init_client()

    def init_client(self):
        self.client = EmbedClient(
            endpoint=f"http://{EMB_HOST}:28888/embed",
            api_format="tei",
            res_format="list2d",
            verbose=False,
        )

    def embed(self, texts: StrsType) -> list[str]:
        floats_embs = self.client.embed(texts)
        hash_embs = [bits_to_hash(floats_to_bits(emb, k=2)) for emb in floats_embs]
        return hash_embs


class LSHEmbedder(EmbedInterface):
    def __init__(self, bitn: int = None, seed: int = None):
        self.init_client()
        self.init_lsh(bitn=bitn, seed=seed)

    def init_client(self):
        # python -m tfmx.embed_server -t "tei" -m "Qwen/Qwen3-Embedding-0.6B" -p 28887 -b
        self.client = EmbedClient(
            endpoint=f"http://{EMB_HOST}:28887/embed",
            api_format="tei",
            res_format="list2d",
            verbose=False,
        )

    def init_lsh(self, bitn: int = None, seed: int = None):
        lsh_params = {}
        if bitn is not None:
            lsh_params["bitn"] = bitn
        if seed is not None:
            lsh_params["seed"] = seed
        self.lsh = LSHConverter(**lsh_params)

    def embed(self, texts: StrsType) -> list[str]:
        """Get embeddings from qwen3-0.6b and convert to LSH hash strings."""
        embs = self.client.embed(texts)
        embs_np = np.array(embs, dtype=np.float32)
        bits = self.lsh.embs_to_bits(embs_np)
        hash_strs = [self.lsh.bits_to_hex(bits_row) for bits_row in bits]
        return hash_strs


EmbedClientType = Union[EmbedClient, RandomEmbedder, HashEmbedder, LSHEmbedder]


class EmbeddingPreCalculator:
    """Pre-calculate embeddings for query-passage pairs."""

    def __init__(
        self, max_count: int = None, overwrite: bool = False, skip_set: bool = False
    ):
        self.max_count = max_count
        self.overwrite = overwrite
        self.skip_set = skip_set
        self.init_processors()

    def init_processors(self):
        self.embed_db = PARENT_DIR / "embeddings.rkdb"
        self.rocks = RocksOperator(
            configs={"db_path": self.embed_db}, connect_cls=self.__class__
        )
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        self.passage_manager = PassageJsonManager()

    def init_embed_clients(self):
        """Must call this before using `embed_text_by_type()`."""
        embed_params = {"api_format": "tei", "res_format": "list2d"}
        self.embed_types = ["gte", "bge", "qwen3_06b"]
        # python -m tfmx.embed_server -t "tei" -m "Alibaba-NLP/gte-multilingual-base" -p 28888 -b
        self.gte_embed = EmbedClient(
            endpoint=f"http://{EMB_HOST}:28888/embed", **embed_params
        )
        # python -m tfmx.embed_server -t "tei" -m "BAAI/bge-large-zh-v1.5" -p 28889 -b
        self.bge_embed = EmbedClient(
            endpoint=f"http://{EMB_HOST}:28889/embed", **embed_params
        )
        # python -m tfmx.embed_server -t "tei" -m "Qwen/Qwen3-Embedding-0.6B" -p 28887 -b
        self.qwen3_06b_embed = EmbedClient(
            endpoint=f"http://{EMB_HOST}:28887/embed", **embed_params
        )
        self.embed_clients: dict[EmbedModelType, EmbedClientType] = {
            "gte": self.gte_embed,
            "bge": self.bge_embed,
            "qwen3_06b": self.qwen3_06b_embed,
        }

    def set_embed_clients(self, embed_clients: dict[EmbedModelType, EmbedClientType]):
        """Dynamically set embed types and clients. Must call this before using `embed_text_by_type()`."""
        self.embed_types = list(embed_clients.keys())
        self.embed_clients: dict[EmbedModelType, EmbedClientType] = embed_clients

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
        key_checks = []  # list of (key, redis_key, redis_field)
        for key in keys:
            redis_key, redis_field = get_redis_key_field(key)
            key_checks.append((key, redis_key, redis_field))
            if redis_field is None:
                pipeline.exists(redis_key)
            else:
                pipeline.hexists(redis_key, redis_field)
        results = pipeline.execute()
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

    def save_embedding(self, key: str, embedding: EmbedValType):
        if not key or not embedding:
            return
        rocks_key = get_rocks_key(key)
        self.rocks.set(rocks_key, stringify_embedding(embedding))
        self.set_key_exist(key)

    def save_embeddings(self, key_embeddings: dict[str, EmbedValType]):
        if not key_embeddings:
            return
        self.rocks.mset(key_embeddings)

    def embed_text_by_type(
        self, texts: StrsType, embed_type: EmbedModelType
    ) -> list[EmbedValType]:
        if not texts:
            return []
        embed_client: EmbedClientType = self.embed_clients.get(embed_type)
        if embed_client:
            embeddings = embed_client.embed(texts)
        else:
            log_error(f"× Unknown embed_type: {embed_type}")
        return embeddings

    def embed_key_text_pairs(
        self, key_text_pairs: list[tuple[str, str]], emb_type: EmbedModelType
    ) -> dict[str, EmbedValType]:
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
        if self.overwrite:
            keys_exist = [False] * len(all_bv_keys)
        else:
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
            if not self.skip_set:
                # often used at same time with "overwrite" in a re-calc
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
        logger.note(f"> Calculating embeddings:", end=" ")
        if self.overwrite:
            logger.mesg(f"(overwrite={self.overwrite})")
        else:
            logger.mesg("")
        bar = TCLogbar(total=self.get_upper_count(), desc="* query")
        for idx, (query, passage) in enumerate(
            self.passage_manager.iter_query_passages()
        ):
            if self.max_count and idx >= self.max_count:
                break
            # logger.note(query)
            bar.update(increment=1, desc=f"{chars_slice(query,end=10)}")
            # self.calc_qr_embedding(query)
            hits = passage["hits"]
            self.calc_hits_embeddings(hits)
        print()


class EmbeddingBenchmarkBuilder:
    """Build benchmark with embeddings from multi models."""

    def __init__(self, max_count: int = None, prettify_json: bool = False):
        self.max_count = max_count
        self.prettify_json = prettify_json
        self.init_processors()

    def init_processors(self):
        self.embed_db = PARENT_DIR / "embeddings.rkdb"
        self.rocks = RocksOperator(
            configs={"db_path": self.embed_db}, connect_cls=self.__class__
        )
        self.passage_manager = PassageJsonManager()
        self.bm_ranks_path = PARENT_DIR / "benchmarks" / "benchmark_ranks.json"

    def init_embed_types(self):
        """Must call this before using `calc_ranks_for_anchor()` or `calc_hits_ranks()`."""
        self.embed_types = ["gte", "bge", "qwen3_06b"]
        self.is_calc_rank_med = True
        self.ele_type = "dot"

    def set_embed_types(
        self,
        embed_types: list[str],
        is_calc_rank_med: bool = False,
        ele_type: Literal["dot", "hash"] = "dot",
    ):
        """Dynamically set embed types for rank calc. Must call this before using `calc_ranks_for_anchor()` or `calc_hits_ranks()`."""
        if not isinstance(embed_types, (list, tuple)):
            embed_types = [embed_types]
        self.embed_types = embed_types
        self.is_calc_rank_med = is_calc_rank_med
        self.ele_type = ele_type

    def is_hash_ele(self):
        return self.ele_type == "hash"

    def load_embedding(self, key: str) -> EmbedValType:
        if not key:
            return None
        rocks_key = get_rocks_key(key)
        value = self.rocks.get(rocks_key)
        if value is None:
            return None
        try:
            if self.is_hash_ele():
                embedding = value
            else:
                if isinstance(value, list):
                    embedding = value
                else:
                    embedding = json.loads(value)
                embedding = floatize_embedding(embedding)
        except Exception as e:
            error_mesg = f"× Fail to load embedding of key {brk(key)}: {e}"
            log_error(error_mesg)
        return embedding

    def load_embeddings(self, keys: list[str]) -> dict[str, EmbedValType]:
        if not keys:
            return {}
        rocks_keys = [get_rocks_key(key) for key in keys]
        values = self.rocks.mget(rocks_keys)
        embeddings: dict[str, EmbedValType] = {}
        for key, value in zip(keys, values):
            if value is None:
                continue
            try:
                if self.is_hash_ele():
                    embeddings[key] = value
                else:
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
        embeddings: dict[str, EmbedValType],
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
                if self.is_hash_ele():
                    sim = hash_sim(anchor_emb, emb)
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
        field_names: list[str] = ["merged"],
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
                if self.is_calc_rank_med:
                    field_med_key = f"{field_name}.med"
                    anchor_result[field_med_key] = calc_median_ranks(ranks_by_emb)
            result[anchor_idx] = anchor_result
        return result

    def update_benchmark_data(self, data: dict, results: dict):
        for query, ranks_data in results.items():
            if query not in data:
                data[query] = defaultdict(dict)
            for anchor_idx, anchor_ranks in ranks_data.items():
                data[query][str(anchor_idx)].update(anchor_ranks)
        return data

    def save_benchmark_ranks(self, results: dict):
        self.bm_ranks_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self.bm_ranks_path.exists():
            try:
                with open(self.bm_ranks_path, encoding="utf-8", mode="r") as rf:
                    data = json.load(rf)
            except Exception as e:
                err_mesg = f"× Failed to load existing benchmark ranks: {e}"
                log_error(err_mesg)
        data = self.update_benchmark_data(data, results)
        with open(self.bm_ranks_path, encoding="utf-8", mode="w") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        logger.note(f"> Save benchmark ranks to:")
        logger.file(f"  * {self.bm_ranks_path}")

    def prettify_ranks_json(self):
        cmd_prettier = (
            f"cd {PARENT_DIR.resolve()} && "
            f"npx prettier --write benchmarks/benchmark_ranks.json"
        )
        shell_cmd(cmd_prettier)

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
        if self.prettify_json:
            self.prettify_ranks_json()


class EmbeddingBenchmarkScorer:
    """Score embedding models with benchmarks based on pre-calced ranks."""

    def __init__(self, max_count: int = None):
        self.max_count = max_count
        self.init_processors()

    def init_processors(self):
        self.bm_ranks_path = PARENT_DIR / "benchmarks" / "benchmark_ranks.json"
        self.bm_scores_path = PARENT_DIR / "benchmarks" / "benchmark_scores.json"

    def load_benchmark_ranks(self) -> dict:
        if not self.bm_ranks_path.exists():
            err_mesg = f"× bm_ranks_path not found: {self.bm_ranks_path}"
            log_error(err_mesg)
        try:
            with open(self.bm_ranks_path, encoding="utf-8", mode="r") as rf:
                return json.load(rf)
        except Exception as e:
            err_mesg = f"× Failed to load benchmark ranks: {e}"
            log_error(err_mesg)

    def calc_max_diff_sum(self, med_ranks: list[int]) -> float:
        """Calc max sum of rank diffs: SUM(n - med_rank[i]) - bias
        Use bias -1.85*n to simulate (with amortization) real-world distribution of ranks.
        """
        n = len(med_ranks)
        return sum(n - rank for rank in med_ranks) - (1.85 * n)

    def calc_rank_score(
        self, emb_ranks: list[int], med_ranks: list[int], max_diff_sum: float
    ) -> float:
        """Calc score for one embedding type against median ranks.

        Score = 1 - (SUM(abs(diff)) / max_diff_sum())
        """
        diff_sum = sum(abs(e - m) for e, m in zip(emb_ranks, med_ranks))
        score = 1.0 - diff_sum / max_diff_sum
        return score

    def calc_anchor_scores(
        self, anchor_ranks: dict[str, list[int]], field_name: str = "merged"
    ) -> dict[str, float]:
        """Calc scores for all emb_types at one anchor_idx.

        Returns: `{emb_type: score}`
        """
        scores = {}
        med_key = f"{field_name}.med"
        med_ranks = anchor_ranks.get(med_key)
        max_diff_sum = self.calc_max_diff_sum(med_ranks)
        field_dot = f"{field_name}."
        for key, emb_ranks in anchor_ranks.items():
            if not key.startswith(field_dot):
                continue
            if key.endswith(".med"):
                continue
            emb_type = key.split(".", 1)[1]
            score = self.calc_rank_score(emb_ranks, med_ranks, max_diff_sum)
            scores[emb_type] = score
        return scores

    def calc_query_scores(
        self, query_data: dict, field_name: str = "merged"
    ) -> dict[str, float]:
        """Calc avg scores for all emb_types across all anchor_indices.

        Returns: `{emb_type: avg_score}`
        """
        emb_scores_by_anchor: dict[str, list[float]] = defaultdict(list)

        for anchor_idx, anchor_ranks in query_data.items():
            anchor_scores = self.calc_anchor_scores(anchor_ranks, field_name)
            for emb_type, score in anchor_scores.items():
                emb_scores_by_anchor[emb_type].append(score)
        avg_scores = {}
        for emb_type, scores in emb_scores_by_anchor.items():
            avg_scores[emb_type] = sum(scores) / len(scores)
        return avg_scores

    def calc_all_scores(
        self, bm_data: dict, field_name: str = "merged"
    ) -> dict[str, dict]:
        """Calc scores for all queries, and calc overall avg score by emb_type.

        Returns:
        {
            "byquery": {query: {emb_type: score}},
            "overall": {emb_type: avg_score}
        }
        """
        # calc scores for each query
        query_scores = {}
        emb_scores_all: dict[str, list[float]] = defaultdict(list)
        bar = TCLogbar(total=len(bm_data), desc="* scoring")
        for query, query_data in bm_data.items():
            bar.update(increment=1, desc=f"{chars_slice(query, end=10)}")
            scores = self.calc_query_scores(query_data, field_name)
            query_scores[query] = scores
            for emb_type, score in scores.items():
                emb_scores_all[emb_type].append(score)
        print()

        # calc overall avg for each embedding type
        overall_scores = {}
        for emb_type, scores in emb_scores_all.items():
            overall_scores[emb_type] = sum(scores) / len(scores)
        return {"byquery": query_scores, "overall": overall_scores}

    def save_benchmark_scores(self, scores_data: dict):
        self.bm_scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm_scores_path, encoding="utf-8", mode="w") as wf:
            json.dump(scores_data, wf, ensure_ascii=False, indent=2)
        logger.note(f"> Save benchmark scores to:")
        logger.file(f"  * {self.bm_scores_path}")
        logger.note(f"> Overall Scores:")
        overall_data = scores_data.get("overall", {})
        for emb_type, score in sorted(
            overall_data.items(), key=lambda x: x[1], reverse=True
        ):
            logger.okay(f"  * {emb_type:10s}: {score:.4f}")

    def run(self):
        bm_data = self.load_benchmark_ranks()
        scores_data = self.calc_all_scores(bm_data, field_name="merged")
        self.save_benchmark_scores(scores_data)

    """
    > Overall Scores:
    * qwen3_06b : 0.7543
    * gte       : 0.7469
    * bge       : 0.7042
    * lsh       : 0.6412
    * hash      : 0.4023
    * test      : 0.0094
    """


class CalculatorArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # flags: max_count, overwrite
        self.add_argument("-n", "--max-count", type=int, default=None)
        self.add_argument("-w", "--overwrite", action="store_true")
        self.add_argument("-k", "--skip-set", action="store_true")
        # lsh params: bitn, seed
        self.add_argument("-bn", "--bitn", type=int, default=None)
        self.add_argument("-sd", "--seed", type=int, default=None)
        # tasks: precalc, builder, scorer
        self.add_argument("-p", "--precalc", action="store_true")
        self.add_argument("-r", "--builder", action="store_true")
        self.add_argument("-s", "--scorer", action="store_true")
        # options: prettify_json
        self.add_argument("-j", "--prettify-json", action="store_true")
        # models: baseline, compare
        self.add_argument("-b", "--baseline", action="store_true")
        self.add_argument("-c", "--compare", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    args = CalculatorArgParser().args

    if args.precalc:
        calculator = EmbeddingPreCalculator(
            max_count=args.max_count, overwrite=args.overwrite, skip_set=args.skip_set
        )
        if args.baseline:
            calculator.init_embed_clients()
            calculator.run()
        if args.compare:
            # random_embedder = RandomEmbedder(dim=128)
            # calculator.set_embed_clients({"test": random_embedder})
            # hash_embedder = HashEmbedder()
            # calculator.set_embed_clients({"hash": hash_embedder})
            lsh_embedder = LSHEmbedder(bitn=args.bitn, seed=args.seed)
            calculator.set_embed_clients({"lsh": lsh_embedder})
            calculator.run()

    if args.builder:
        benchmark_builder = EmbeddingBenchmarkBuilder(
            max_count=args.max_count, prettify_json=args.prettify_json
        )
        if args.baseline:
            benchmark_builder.init_embed_types()
            benchmark_builder.run()
        if args.compare:
            # embed_types = ["hash"]
            embed_types = ["lsh"]
            benchmark_builder.set_embed_types(
                embed_types, is_calc_rank_med=False, ele_type="hash"
            )
            benchmark_builder.run()

    if args.scorer:
        benchmark_scorer = EmbeddingBenchmarkScorer(max_count=args.max_count)
        benchmark_scorer.run()


if __name__ == "__main__":
    main()
    # Case 1: (baseline) pre-calc embeddings for embed_types ["gte", "bge", "qwen3_06b"]
    # python -m models.tembed.calc -p -b

    # Case 2: (baseline) build benchmark ranks based on pre-calced embeddings
    # python -m models.tembed.calc -r -b -n 10
    # python -m models.tembed.calc -r -b

    # Case 3: (compare) pre-calc embeddings
    # python -m models.tembed.calc -p -c
    # python -m models.tembed.calc -p -c -wk
    # python -m models.tembed.calc -p -c -wk -sd 1

    # Case 4: (compare) build benchmark ranks
    # python -m models.tembed.calc -r -c -n 10
    # python -m models.tembed.calc -r -c
    # python -m models.tembed.calc -r -c -f

    # format json
    # cd ~/repos/bili-search-algo/models/tembed
    # npx prettier --write benchmarks/benchmark_ranks.json

    # Case 5: run for both baseline and compare
    # python -m models.tembed.calc -p -b -c
    # python -m models.tembed.calc -r -b -c

    # Case 6: score embeddings based on benchmark ranks
    # python -m models.tembed.calc -s
