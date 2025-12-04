import argparse
import numpy as np
import pickle

from pathlib import Path
from typing import Iterator

from sedb import MongoDocsGenerator, MongoDocsGeneratorArgParser
from sedb import RedisOperator, RocksOperator
from sedb import FaissOperator, FaissClient, FAISS_PORT
from tclogger import TCLogger, TCLogbar, logstr, dict_get, brk, dict_to_str
from tclogger import raise_breakpoint, MergedArgParser
from tfmx import EmbedClient

from configs.envs import REDIS_ENVS, MONGO_ENVS

logger = TCLogger()


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
SAMPLES_DIR = STORAGE_DIR / "train_samples"
MAX_SAMPLES = int(1e10)  # max samples count, used to determine idx_width in filenames

REDIS_PREFIX = f"bv.emb:"
REDIS_PT = f"{REDIS_PREFIX}*"


def bvids_to_redis_name_fields(bvids: list[str]) -> list[tuple[str, str]]:
    """Convert bvids to Redis hash name-field pairs"""
    return [(f"{REDIS_PREFIX}{bvid}", EMB_FIELD) for bvid in bvids]


def redis_name_fields_to_bvids(name_fields: list[tuple[str, str]]) -> list[str]:
    """Extract bvids from Redis hash name-field pairs"""
    return [name.removeprefix(REDIS_PREFIX) for name, _ in name_fields]


def redis_keys_to_bvids(keys: list[str]) -> list[str]:
    """Convert Redis keys to bvids"""
    return [key.removeprefix(REDIS_PREFIX) for key in keys]


def name_fields_to_rocks_keys(name_fields: list[tuple[str, str]]) -> list[str]:
    """Convert Redis name-field pairs to RocksDB keys"""
    return [f"{name}.{field}" for name, field in name_fields]


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

    def submit(self):
        if not self.batch:
            return
        # get non exist redis name-fields
        bvids = list(self.batch.keys())
        name_fields = bvids_to_redis_name_fields(bvids)
        non_exist_name_fields = self.redis.get_non_exist_hashes(name_fields)
        if not non_exist_name_fields:
            return
        # calc embeddings
        non_exist_bvids = redis_name_fields_to_bvids(non_exist_name_fields)
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


class FaissBuilder:
    """Build Faiss index with Redis and RocksDB"""

    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size
        self.init_processors()

    def init_processors(self):
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        rocks_configs = {"db_path": ROCKS_DB_PATH}
        self.rocks = RocksOperator(configs=rocks_configs, connect_cls=self.__class__)
        self.faiss = FaissOperator(FAISS_DB_PATH, dim=EMB_DIM)
        self.faiss.init_db()

    def redis_key_to_bvid(self, key: str) -> str:
        """'bv.emb:BV1234567890' -> 'BV1234567890'"""
        return key.removeprefix(REDIS_PREFIX)

    def redis_keys_to_bvids(self, keys: list[str]) -> list[str]:
        return [self.redis_key_to_bvid(key) for key in keys]

    def bvids_to_faiss_hashes(self, bvids: list[str]) -> list[tuple[str, str]]:
        return [(f"{REDIS_PREFIX}{bvid}", FSS_FIELD) for bvid in bvids]

    def write_to_faiss(self, bvids: list[str], embeddings: np.ndarray):
        """Write embeddings to Faiss index and flag in Redis"""
        self.faiss.add_embs(bvids, embeddings)
        # faiss_hashes = self.bvids_to_faiss_hashes(bvids)
        # self.redis.set_hashes_exist(faiss_hashes)

    def run(self):
        # scan rocks, and write to Faiss
        for rocks_items in self.rocks.iter_items(batch_size=10000):
            bvids, embs = zip(*rocks_items)
            self.write_to_faiss(bvids, embs)
        logger.okay(f"* Finished Rocks scan")
        self.faiss.save()


class FaissTester:
    def __init__(self):
        self.faiss = FaissClient(port=FAISS_PORT)

    def run(self):
        logger.note(f"> total_count():")
        result = self.faiss.total_count()
        logger.okay(result)

        eid = "BV1114y1h7rj"
        logger.note(f"> get_emb_by_eid():")
        result = self.faiss.get_emb_by_eid(eid=eid)
        logger.okay(result)

        logger.note(f"> top():")
        result = self.faiss.top(eid=eid, topk=5)
        logger.okay(result)

        eids = ["BV1114y1h7rj", "BV112xVeuEqw"]
        logger.note(f"> get_embs_by_eids():")
        results = self.faiss.get_embs_by_eids(eids=eids)
        logger.okay(results)

        logger.note(f"> tops():")
        results = self.faiss.tops(eids=eids, return_emb=False)
        logger.okay(results)


class TrainSamplesManager:
    """Manager for training samples storage and retrieval

    Handles both writing (during construction) and reading (during training):
    - Writing: buffer samples, save batches, merge into shards
    - Reading: load shards on demand, provide unified access interface

    Storage layout (per shard):
    - eids: [anchor0, pos0_1, pos0_2, anchor1, pos1_1, pos1_2, ...]
    - embs: [emb0_anchor, emb0_pos1, emb0_pos2, emb1_anchor, emb1_pos1, emb1_pos2, ...]

    Access pattern:
    - sample i starts at: idx = i * group_size
    - anchor: eids[idx], embs[idx]
    - positive j: eids[idx + j], embs[idx + j]  (j=1,2)
    - negative: use previous sample (i-1)'s anchor

    Note: Valid sample indices are [1, num_samples), since sample 0 has no negative.
    Use __len__ to get the number of valid samples (num_samples - 1).
    """

    # Maximum shards to keep in memory (LRU cache)
    MAX_CACHED_SHARDS = 3

    def __init__(
        self,
        data_dir: str | Path,
        num_positives: int = 2,
        shard_size: int = 100_000,
        save_batch_size: int = 1000,
        mode: str = "read",
    ):
        """
        Args:
            data_dir: directory for storing/loading training data
            num_positives: number of positive samples per anchor (for write mode)
            shard_size: number of samples per shard file (for write mode)
            save_batch_size: buffer size before saving batch (for write mode)
            mode: "read" for loading existing data, "write" for creating new data
        """
        self.data_dir = Path(data_dir)
        self.batches_dir = self.data_dir / "batches"
        self.shards_dir = self.data_dir / "shards"
        self.mode = mode

        if mode == "read":
            self._init_read_mode()
        else:
            self._init_write_mode(num_positives, shard_size, save_batch_size)

    def is_read_mode(self) -> bool:
        return self.mode == "read"

    def is_write_mode(self) -> bool:
        return self.mode == "write"

    def _init_read_mode(self):
        """initialize for reading existing data"""
        meta_file = self.data_dir / "train_meta.npz"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        # load metadata
        meta = np.load(meta_file)
        self.group_size = int(meta["group_size"])
        self.num_positives = int(meta["num_positives"])
        self.num_samples = int(meta["num_samples"])
        self.num_shards = int(meta["num_shards"])
        self.shard_size = int(meta["shard_size"])

        # LRU shard cache for reading: {shard_idx: {"eids": ..., "embs": ..., "access_order": int}}
        self._shard_cache = {}
        self._access_counter = 0

        # calculate idx_width for filenames
        self.idx_width = self._calc_idx_width()

        logger.note(f"> Loaded TrainSamplesManager:")
        info_dict = {
            "num_samples": self.num_samples,
            "num_shards": self.num_shards,
            "group_size": [
                self.group_size,
                f"1 anchor + {self.num_positives} positives",
            ],
            "shard_size": self.shard_size,
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _calc_idx_width(self) -> int:
        """calculate index width for filenames based on MAX_SAMPLES and shard_size

        e.g., if shard_size=1e5, max_shards=1e10/1e5=1e5, width=len('100000')=6
        """
        import math

        max_shards = MAX_SAMPLES // self.shard_size
        return len(str(max_shards))

    def _init_write_mode(
        self, num_positives: int, shard_size: int, save_batch_size: int
    ):
        """initialize for writing new data"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

        self.num_positives = num_positives
        self.shard_size = shard_size
        self.save_batch_size = save_batch_size
        self.group_size = 1 + num_positives  # anchor + positives

        # write buffers
        self.eids = []
        self.embs = []
        self.batch_idx = 0

        # calculate idx_width for filenames
        self.idx_width = self._calc_idx_width()

        # queried eids cache (to avoid re-querying same bvids)
        self.queried_cache_file = self.data_dir / "queried_eids.pkl"
        self.queried_eids = self._load_queried_cache()

    # ========== Write mode methods ==========

    def _load_queried_cache(self) -> set:
        """load cache of already queried eids"""
        if self.queried_cache_file.exists():
            with open(self.queried_cache_file, "rb") as f:
                return pickle.load(f)
        return set()

    def save_queried_cache(self):
        """save queried eids cache"""
        with open(self.queried_cache_file, "wb") as f:
            pickle.dump(self.queried_eids, f, protocol=pickle.HIGHEST_PROTOCOL)

    def mark_queried(self, eid: str):
        """mark an eid as queried"""
        self.queried_eids.add(eid)

    def is_queried(self, eid: str) -> bool:
        """check if an eid has been queried"""
        return eid in self.queried_eids

    def get_queried_count(self) -> int:
        """get number of queried eids"""
        return len(self.queried_eids)

    def append_sample(self, group_eids: list[str], group_embs: list[np.ndarray]):
        """append a training sample (anchor + positives) to buffer

        Args:
            group_eids: list of eids [anchor, pos1, pos2, ...]
            group_embs: list of embeddings [anchor_emb, pos1_emb, pos2_emb, ...]
        """
        self.eids.extend(group_eids)
        self.embs.extend(group_embs)

    def get_buffer_sample_count(self) -> int:
        """get current number of samples in buffer"""
        return len(self.eids) // self.group_size

    def should_save_batch(self) -> bool:
        """check if current buffer should be saved"""
        return self.get_buffer_sample_count() >= self.save_batch_size

    def save_batch(self) -> int:
        """save current buffer to disk and clear buffer

        Returns:
            batch index of saved batch
        """
        if not self.eids:
            return self.batch_idx

        batch_file = self.batches_dir / f"batch_{self.batch_idx:0{self.idx_width}d}.npz"
        np.savez_compressed(
            batch_file,
            eids=np.array(self.eids, dtype=object),
            embs=np.array(self.embs, dtype=np.float32),
            group_size=self.group_size,
            num_positives=self.num_positives,
        )

        saved_batch_idx = self.batch_idx
        self.batch_idx += 1

        # clear buffers
        self.eids.clear()
        self.embs.clear()

        return saved_batch_idx

    def merge_batches(self):
        """merge batch files into sharded training data files

        Creates sharded NPZ files:
        - shards/shard_000000.npz, shard_000001.npz, ...
        - train_meta.npz: metadata
        """
        # load all batches
        batch_files = sorted(self.batches_dir.glob("batch_*.npz"))
        if not batch_files:
            logger.warn("* No batch files to merge")
            return

        logger.note("> Merge batch files into shards:")
        all_eids = []
        all_embs = []
        for batch_file in batch_files:
            data = np.load(batch_file, allow_pickle=True)
            all_eids.append(data["eids"])
            all_embs.append(data["embs"])

        # concatenate all data
        final_eids = np.concatenate(all_eids)
        final_embs = np.concatenate(all_embs)

        # calculate totals
        total_items = len(final_eids)
        num_samples = total_items // self.group_size
        items_per_shard = self.shard_size * self.group_size
        num_shards = (total_items + items_per_shard - 1) // items_per_shard

        # create shards directory
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # save shards
        for shard_idx in range(num_shards):
            start_item = shard_idx * items_per_shard
            end_item = min((shard_idx + 1) * items_per_shard, total_items)

            shard_file = self.shards_dir / f"shard_{shard_idx:0{self.idx_width}d}.npz"
            np.savez_compressed(
                shard_file,
                eids=final_eids[start_item:end_item],
                embs=final_embs[start_item:end_item],
            )
            shard_samples = (end_item - start_item) // self.group_size
            logger.mesg(
                f"  * Saved to shard {shard_idx}: {logstr.okay(brk(shard_samples))}"
            )

        # save metadata
        np.savez(
            self.data_dir / "train_meta.npz",
            group_size=self.group_size,
            num_positives=self.num_positives,
            num_samples=num_samples,
            num_shards=num_shards,
            shard_size=self.shard_size,
        )

        # update instance state for potential read operations
        self.num_samples = num_samples
        self.num_shards = num_shards

        # clean up batch files after successful merge
        for batch_file in batch_files:
            batch_file.unlink()
        logger.mesg(f"  * Cleared batch files: {len(batch_files)} ")

        logger.okay(f"> Created {num_shards} shards with {num_samples} total samples")
        info_dict = {
            "shard_size": self.shard_size,
            "group_size": [
                self.group_size,
                f"1 anchor + {self.num_positives} positives",
            ],
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def finalize(self):
        """finalize writing: save remaining buffer and merge batches"""
        if self.eids:
            self.save_batch()
        self.save_queried_cache()
        self.merge_batches()

    # ========== Read mode methods ==========

    def preload_all(self):
        """preload all shards into memory

        Warning: This may consume a lot of memory for large datasets.
        Consider using iter_samples() for memory-efficient sequential access.
        """
        logger.warn(
            f"* Loading all {self.num_shards} shards into memory. "
            f"Use iter_samples() for memory-efficient access."
        )
        # temporarily disable LRU eviction
        original_max = self.MAX_CACHED_SHARDS
        self.MAX_CACHED_SHARDS = self.num_shards + 1
        for shard_idx in range(self.num_shards):
            self._load_shard(shard_idx)
        self.MAX_CACHED_SHARDS = original_max

    def _load_shard(self, shard_idx: int):
        """load a shard into cache with LRU eviction"""
        if shard_idx in self._shard_cache:
            # update access order for LRU
            self._access_counter += 1
            self._shard_cache[shard_idx]["access_order"] = self._access_counter
            return

        # LRU eviction if cache is full
        while len(self._shard_cache) >= self.MAX_CACHED_SHARDS:
            # find least recently used shard
            lru_idx = min(
                self._shard_cache.keys(),
                key=lambda k: self._shard_cache[k]["access_order"],
            )
            del self._shard_cache[lru_idx]

        shard_file = self.shards_dir / f"shard_{shard_idx:0{self.idx_width}d}.npz"
        if not shard_file.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_file}")

        data = np.load(shard_file, allow_pickle=True)
        self._access_counter += 1
        self._shard_cache[shard_idx] = {
            "eids": data["eids"],
            "embs": data["embs"],
            "access_order": self._access_counter,
        }

    def _get_shard_for_sample(self, sample_idx: int) -> tuple[int, int]:
        """get shard index and item offset for a sample

        Returns:
            (shard_idx, item_offset_in_shard)
        """
        shard_idx = sample_idx // self.shard_size
        offset = (sample_idx % self.shard_size) * self.group_size
        return shard_idx, offset

    def __len__(self) -> int:
        """return number of valid samples (excluding sample 0 which has no negative)"""
        return self.num_samples - 1 if self.num_samples > 0 else 0

    def __getitem__(self, sample_idx: int) -> dict:
        """get training sample at index

        Args:
            sample_idx: valid range is [1, num_samples)

        Returns:
            dict with anchor, positives, and negative (from previous sample)
        """
        if not (0 < sample_idx < self.num_samples):
            raise IndexError(
                f"sample_idx {sample_idx} out of range [1, {self.num_samples})"
            )

        # get shard info
        shard_idx, shard_offset = self._get_shard_for_sample(sample_idx)
        self._load_shard(shard_idx)

        shard = self._shard_cache[shard_idx]

        # anchor (first in group)
        anchor_eid = shard["eids"][shard_offset]
        anchor_emb = shard["embs"][shard_offset]

        # positives (rest in group)
        pos_eids = [shard["eids"][shard_offset + j] for j in range(1, self.group_size)]
        pos_embs = np.array(
            [shard["embs"][shard_offset + j] for j in range(1, self.group_size)]
        )

        # negative (previous sample's anchor) - may be in different shard
        prev_sample_idx = sample_idx - 1
        prev_shard_idx, prev_shard_offset = self._get_shard_for_sample(prev_sample_idx)

        if prev_shard_idx != shard_idx:
            self._load_shard(prev_shard_idx)
            prev_shard = self._shard_cache[prev_shard_idx]
        else:
            prev_shard = shard

        neg_eid = prev_shard["eids"][prev_shard_offset]
        neg_emb = prev_shard["embs"][prev_shard_offset]

        return {
            "anchor_eid": anchor_eid,
            "anchor_emb": anchor_emb,
            "pos_eids": pos_eids,
            "pos_embs": pos_embs,
            "neg_eid": neg_eid,
            "neg_emb": neg_emb,
        }

    def iter_samples(self, start: int = 1, end: int = None) -> Iterator[dict]:
        """iterate through samples sequentially (memory efficient)

        Args:
            start: starting sample index (default 1, since 0 has no negative)
            end: ending sample index (exclusive, default None for all)

        Raises:
            ValueError: if called in write mode
        """
        if not self.is_read_mode():
            raise ValueError("iter_samples() is only available in read mode")

        if end is None:
            end = self.num_samples

        # validate range
        start = max(1, start)  # ensure start >= 1
        end = min(end, self.num_samples)

        for sample_idx in range(start, end):
            yield self[sample_idx]

    def iter_shard_samples(self, shard_idx: int) -> Iterator[dict]:
        """iterate through all samples in a specific shard

        Useful for parallel loading in distributed training.

        Raises:
            ValueError: if called in write mode or shard_idx out of range
        """
        if not self.is_read_mode():
            raise ValueError("iter_shard_samples() is only available in read mode")

        if shard_idx < 0 or shard_idx >= self.num_shards:
            raise ValueError(
                f"Shard index {shard_idx} out of range [0, {self.num_shards})"
            )

        start_sample = shard_idx * self.shard_size
        end_sample = min((shard_idx + 1) * self.shard_size, self.num_samples)

        # skip first sample if it's the very first
        if start_sample == 0:
            start_sample = 1

        for sample_idx in range(start_sample, end_sample):
            yield self[sample_idx]

    def get_shard_range(self, shard_idx: int) -> tuple[int, int]:
        """get sample index range for a shard

        Returns:
            (start_sample_idx, end_sample_idx) - note: start may be 1 for shard 0
        """
        start = shard_idx * self.shard_size
        end = min((shard_idx + 1) * self.shard_size, self.num_samples)
        # adjust start for first shard (sample 0 is invalid)
        if start == 0:
            start = 1
        return start, end

    def get_stats(self) -> dict:
        """get statistics about the training data"""
        return {
            "num_samples": self.num_samples,
            "num_shards": self.num_shards,
            "shard_size": self.shard_size,
            "group_size": self.group_size,
            "num_positives": self.num_positives,
            "cached_shards": len(self._shard_cache),
            "max_cached_shards": self.MAX_CACHED_SHARDS,
        }


class TrainSamplesConstructor:
    """Construct training tuples for contrastive learning

    Efficiently generates training tuples by:
    1. iterating through Redis keys to get bvids
    2. using FaissClient.tops() to retrieve top-k similar embeddings in batches
    3. using TrainSamplesManager for data storage and caching

    Storage is handled by TrainSamplesManager with:
    - continuous layout: [anchor, pos1, pos2, anchor, pos1, pos2, ...]
    - sharded storage for memory efficiency
    - processed eids cache for resumable construction
    """

    def __init__(
        self,
        manager: TrainSamplesManager,
        samples_count: int = 1_000_000,
        query_batch_size: int = 50,
    ):
        """
        Args:
            manager: TrainSamplesManager instance for data storage
            samples_count: number of training samples to generate
            query_batch_size: batch size for querying Faiss tops() (default: 50)
        """
        if not manager.is_write_mode():
            raise ValueError("TrainSamplesConstructor requires manager in write mode")

        self.manager = manager
        self.samples_count = samples_count
        self.num_positives = manager.num_positives
        self.topk = self.num_positives + 1  # top1 (anchor) + num_positives
        self.query_batch_size = query_batch_size

        # statistics
        self.stats = {
            "scanned_count": 0,
            "queried_count": 0,
            "built_count": 0,
            "failed_count": 0,
            "skip_count": 0,
        }

        # initialize processors
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        self.faiss = FaissClient(port=FAISS_PORT)

    def _build_sample(self, top_results: list) -> bool:
        """build and append training sample from top-k results

        Args:
            top_results: list of dicts {'eid': ..., 'emb': ..., 'sim': ...}
                from FaissClient.tops() with return_emb=True

        Returns:
            True if sample was successfully built and appended
        """
        # need at least 1 (anchor) + num_positives results
        if not top_results or len(top_results) < self.num_positives + 1:
            return False

        # top1 is anchor, top2/top3/... are positives
        group_eids = []
        group_embs = []

        for i in range(self.num_positives + 1):  # anchor + positives
            item = top_results[i]
            eid = item.get("eid")
            emb = item.get("emb")
            if eid is None or emb is None:
                return False
            group_eids.append(eid)
            group_embs.append(emb)

        # append to manager
        self.manager.append_sample(group_eids, group_embs)
        return True

    def _query_and_build_samples(self, bvid_batch: list) -> tuple[int, int]:
        """query Faiss for top-k and build training samples

        Args:
            bvid_batch: list of bvids to query

        Returns:
            tuple of (successfully built count, failed count)
        """
        built_count = 0
        failed_count = 0

        try:
            # get top-k from Faiss in batch with embeddings
            batch_results = self.faiss.tops(
                eids=bvid_batch, topk=self.topk, return_emb=True
            )

            # build sample for each query result
            for query_bvid, top_results in zip(bvid_batch, batch_results):
                self.stats["queried_count"] += 1

                # build sample (top1=anchor, top2/top3=positives)
                if self._build_sample(top_results):
                    # mark as queried only on success
                    self.manager.mark_queried(query_bvid)
                    built_count += 1
                    self.stats["built_count"] += 1
                else:
                    failed_count += 1
                    self.stats["failed_count"] += 1

        except Exception as e:
            logger.warn(f"Error querying Faiss batch: {e}")
            failed_count = len(bvid_batch)
            self.stats["failed_count"] += failed_count

        return built_count, failed_count

    def _should_stop(self) -> bool:
        """check if scanned eids count reached samples_count"""
        return self.stats["scanned_count"] >= self.samples_count

    def run(self):
        """main execution: construct training tuples"""
        self._log_start_info()

        # initialize state
        total_built = 0

        # create progress bar (tracks scanned count, not built count)
        self.bar = TCLogbar(
            total=self.samples_count,
            desc=logstr.note("* Construct samples"),
            show_iter_per_second=True,
        )

        # scan and process Redis keys in batches
        for batch_keys in self.redis.scan_keys(
            pattern=REDIS_PT, batch_size=self.query_batch_size
        ):
            all_bvids = redis_keys_to_bvids(batch_keys)

            # update scanned count (all eids, including queried)
            self.stats["scanned_count"] += len(all_bvids)
            self.bar.update(increment=len(all_bvids))

            # filter out already queried eids
            bvid_batch = [
                bvid for bvid in all_bvids if not self.manager.is_queried(bvid)
            ]

            # count already queried as skipped
            already_queried = len(all_bvids) - len(bvid_batch)
            self.stats["skip_count"] += already_queried

            if not bvid_batch:
                if self._should_stop():
                    break
                continue

            # query Faiss and build samples
            built_in_batch, _ = self._query_and_build_samples(bvid_batch)
            total_built += built_in_batch

            # save batch and cache periodically
            if self.manager.should_save_batch():
                self.manager.save_batch()
                if self.manager.batch_idx % 10 == 0:
                    self.manager.save_queried_cache()

            if self._should_stop():
                break

        # finalize
        self.bar.update(increment=0, flush=True)
        print()

        self._finalize(total_built)

    def _log_start_info(self):
        """log initialization information"""
        logger.note(f"> Constructing samples:")
        info_dict = {
            "samples_count": self.samples_count,
            "num_positives": self.num_positives,
            "query_batch_size": self.query_batch_size,
            "data_dir": self.manager.data_dir,
            "queried_count": self.manager.get_queried_count(),
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _finalize(self, total_built: int):
        """finalize construction: save remaining data and merge batches

        Args:
            total_built: total number of built samples
        """
        # finalize manager (save remaining, merge batches)
        self.manager.finalize()

        # log final statistics
        logger.okay(f"> Construct complete:")
        info_dict = {
            "total_built": total_built,
            "tuple_size": f"3 (1 anchor + {self.num_positives} positives)",
            "samples_path": str(self.manager.data_dir),
            "stats": {
                "scanned_count": self.stats["scanned_count"],
                "skip_count": self.stats["skip_count"],
                "queried_count": self.stats["queried_count"],
                "built_count": self.stats["built_count"],
                "failed_count": self.stats["failed_count"],
                "built_ratio": (
                    f"{self.stats['built_count'] / max(1, self.stats['queried_count']) * 100:.2f}%"
                ),
            },
        }
        logger.mesg(dict_to_str(info_dict), indent=2)


class TrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-cc", "--calc", action="store_true")
        self.add_argument("-bf", "--build-faiss", action="store_true")
        self.add_argument("-tf", "--test-faiss", action="store_true")
        self.add_argument("-cs", "--construct-samples", action="store_true")
        self.add_argument("-sn", "--samples-count", type=int, default=1_000_000)
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

    if args.test_faiss:
        tester = FaissTester()
        tester.run()

    if args.construct_samples:
        manager = TrainSamplesManager(
            data_dir=SAMPLES_DIR,
            num_positives=2,
            shard_size=100_000,
            save_batch_size=1000,
            mode="write",
        )
        constructor = TrainSamplesConstructor(
            manager=manager,
            samples_count=args.samples_count,
        )
        constructor.run()


if __name__ == "__main__":
    main()

    # Case: calc and store embeddings
    # python -m models.tembed.train -cc -ec -mn 10000
    # python -m models.tembed.train -cc

    # Case: build faiss HNSW index from RocksDB
    # python -m models.tembed.train -bf

    # Case: test faiss index
    # python -m models.tembed.train -tf

    # Case: construct training samples for contrastive learning
    # python -m models.tembed.train -cs
    # python -m models.tembed.train -cs -sn 10000
