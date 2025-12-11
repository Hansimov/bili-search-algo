import argparse
import json
import numpy as np
import pickle

from pathlib import Path
from typing import Iterator

from sedb import MongoDocsGenerator, MongoDocsGeneratorArgParser
from sedb import RedisOperator, RocksOperator
from sedb import FaissOperator, FaissClient, FAISS_PORT
from tclogger import TCLogger, TCLogbar, logstr
from tclogger import dict_get, brk, dict_to_str, int_bits
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
MAX_SAMPLES = int(1e10)

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


class BuffersTracker:
    """Tracker for buffer files status, persisted in buffers.json

    Tracks:
    - file_count: number of buffer files in the directory
    - last_file: filename of the last buffer file
    - last_file_items: number of items in the last buffer file
    """

    def __init__(self, buffers_dir: Path, group_size: int, buffer_size: int):
        """
        Args:
            buffers_dir: directory containing buffer files
            group_size: number of items per sample (1 anchor + num_positives)
            buffer_size: number of samples per buffer file
        """
        self.buffers_dir = buffers_dir
        self.group_size = group_size
        self.buffer_size = buffer_size
        self.json_file = buffers_dir.parent / "buffers.json"

        # tracked state
        self.file_count: int = 0
        self.last_file: str = ""
        self.last_file_samples: int = 0

    def load(self) -> "BuffersTracker":
        """Load tracker state from JSON file"""
        if self.json_file.exists():
            try:
                with open(self.json_file, "r") as f:
                    data = json.load(f)
                self.file_count = data.get("file_count", 0)
                self.last_file = data.get("last_file", "")
                self.last_file_samples = data.get("last_file_samples", 0)
            except Exception as e:
                logger.warn(f"Error loading buffers.json: {e}")
        return self

    def save(self) -> "BuffersTracker":
        """Save tracker state to JSON file"""
        self.buffers_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "file_count": self.file_count,
            "last_file": self.last_file,
            "last_file_samples": self.last_file_samples,
        }
        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=2)
        return self

    def is_last_file_full(self) -> bool:
        """Check if the last buffer file is full"""
        return self.last_file_samples >= self.buffer_size

    def get_next_buffer_idx(self) -> int:
        """Get the next buffer index to use

        If last file is not full, return (file_count - 1) to overwrite it.
        Otherwise, return file_count to create a new file.
        """
        if self.file_count == 0:
            return 0
        if self.is_last_file_full():
            return self.file_count
        return self.file_count - 1

    def get_total_samples(self) -> int:
        """Get total number of samples across all buffer files"""
        if self.file_count == 0:
            return 0
        # (file_count - 1) full buffers + last buffer samples
        full_buffers = self.file_count - 1
        return full_buffers * self.buffer_size + self.last_file_samples

    def update(self, file_count: int, last_file: str, last_file_samples: int):
        """Update tracker state"""
        self.file_count = file_count
        self.last_file = last_file
        self.last_file_samples = last_file_samples

    def log_info(self):
        """Log tracker information"""
        if self.file_count > 0:
            ratio = round(self.last_file_samples / self.buffer_size * 100, 1)
            logger.note(f"> BuffersTracker:")
            info_dict = {
                "file_count": self.file_count,
                "last_file": self.last_file,
                "last_file_samples": f"{self.last_file_samples} ({ratio}%)",
                "next_buffer_idx": self.get_next_buffer_idx(),
            }
            logger.mesg(dict_to_str(info_dict), indent=2)


class ShardsTracker:
    """Tracker for shard files status, persisted in shards.json

    Tracks:
    - file_count: number of shard files in the directory
    - last_file: filename of the last shard file
    - last_file_items: number of items in the last shard file
    - total_samples: total samples across all shards
    """

    def __init__(self, shards_dir: Path, group_size: int, shard_size: int):
        """
        Args:
            shards_dir: directory containing shard files
            group_size: number of items per sample (1 anchor + num_positives)
            shard_size: number of samples per shard file
        """
        self.shards_dir = shards_dir
        self.group_size = group_size
        self.shard_size = shard_size
        self.json_file = shards_dir.parent / "shards.json"

        # tracked state
        self.file_count: int = 0
        self.last_file: str = ""
        self.last_file_samples: int = 0
        self.total_samples: int = 0

    def load(self) -> "ShardsTracker":
        """Load tracker state from JSON file"""
        if self.json_file.exists():
            try:
                with open(self.json_file, "r") as f:
                    data = json.load(f)
                self.file_count = data.get("file_count", 0)
                self.last_file = data.get("last_file", "")
                self.last_file_samples = data.get("last_file_samples", 0)
                self.total_samples = data.get("total_samples", 0)
            except Exception as e:
                logger.warn(f"Error loading shards.json: {e}")
        return self

    def save(self) -> "ShardsTracker":
        """Save tracker state to JSON file"""
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "file_count": self.file_count,
            "last_file": self.last_file,
            "last_file_samples": self.last_file_samples,
            "total_samples": self.total_samples,
        }
        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=2)
        return self

    def is_last_file_full(self) -> bool:
        """Check if the last shard file is full"""
        return self.last_file_samples >= self.shard_size

    def get_fill_target(self) -> int:
        """Get number of samples needed to fill the last shard

        Returns:
            Number of samples needed, or 0 if last shard is full or no shards exist
        """
        if self.file_count == 0 or self.is_last_file_full():
            return 0
        return self.shard_size - self.last_file_samples

    def is_filling_mode(self) -> bool:
        """Check if we need to fill the last partial shard"""
        return self.get_fill_target() > 0

    def update(
        self,
        file_count: int,
        last_file: str,
        last_file_samples: int,
        total_samples: int,
    ):
        """Update tracker state"""
        self.file_count = file_count
        self.last_file = last_file
        self.last_file_samples = last_file_samples
        self.total_samples = total_samples

    def log_info(self):
        """Log tracker information"""
        if self.file_count > 0:
            ratio = round(self.last_file_samples / self.shard_size * 100, 1)
            logger.note(f"> ShardsTracker:")
            info_dict = {
                "file_count": self.file_count,
                "last_file": self.last_file,
                "last_file_samples": f"{self.last_file_samples} ({ratio}%)",
                "total_samples": self.total_samples,
                "fill_target": self.get_fill_target(),
            }
            logger.mesg(dict_to_str(info_dict), indent=2)


class BuffersWriter:
    """Writer for buffer files with tracking support"""

    def __init__(
        self,
        buffers_dir: Path,
        group_size: int,
        buffer_size: int,
        shard_size: int,
    ):
        """
        Args:
            buffers_dir: directory for buffer files
            group_size: number of items per sample (1 anchor + num_positives)
            buffer_size: number of samples per buffer file
            shard_size: number of samples per shard file (used for idx_width calculation)
        """
        self.buffers_dir = buffers_dir
        self.group_size = group_size
        self.buffer_size = buffer_size
        self.shard_size = shard_size
        # calculate idx_width: max buffers = shard_size / buffer_size + 1
        max_buffers = shard_size // buffer_size + 1
        self.idx_width = int_bits(max_buffers)

        # in-memory buffer
        self.eids: list = []
        self.embs: list = []

        # tracker
        self.tracker = BuffersTracker(buffers_dir, group_size, buffer_size).load()
        self.tracker.log_info()

        # current buffer index
        self.buffer_idx = self.tracker.get_next_buffer_idx()

        # load partial last buffer if exists
        self._load_partial_last_buffer()

    def _load_partial_last_buffer(self):
        """Load partial last buffer into memory if it exists and is not full"""
        if self.tracker.file_count == 0:
            return

        if self.tracker.is_last_file_full():
            return

        last_buffer_file = self.buffers_dir / self.tracker.last_file
        if last_buffer_file.exists():
            try:
                data = np.load(last_buffer_file, allow_pickle=True)
                self.eids = list(data["eids"])
                self.embs = list(data["embs"])
                logger.note(
                    f"> Loaded partial buffer: {len(self.eids) // self.group_size} samples"
                )
            except Exception as e:
                logger.warn(f"Error loading partial buffer: {e}")

    def get_buffer_file(self, buffer_idx: int) -> Path:
        """Get buffer file path for a given buffer index"""
        return self.buffers_dir / f"buffer_{buffer_idx:0{self.idx_width}d}.npz"

    def get_memory_sample_count(self) -> int:
        """Get current number of samples in memory buffer"""
        return len(self.eids) // self.group_size

    def get_total_sample_count(self) -> int:
        """Get total number of samples in buffer files + memory"""
        return self.tracker.get_total_samples() + self.get_memory_sample_count()

    def next_flush_size(self) -> int:
        """Calculate the target number of samples to trigger next buffer flush"""
        return self.buffer_size

    def should_flush(self) -> bool:
        """Check if current in-memory buffer should be flushed to file"""
        return self.get_memory_sample_count() >= self.next_flush_size()

    def append(self, group_eids: list[str], group_embs: list[np.ndarray]) -> bool:
        """Append a training sample to buffer

        Args:
            group_eids: list of eids [anchor, pos1, pos2, ...]
            group_embs: list of embeddings [anchor_emb, pos1_emb, pos2_emb, ...]

        Returns:
            True if buffer was flushed after appending
        """
        self.eids.extend(group_eids)
        self.embs.extend(group_embs)

        if self.should_flush():
            self.flush()
            return True
        return False

    def flush(self) -> int:
        """Flush current in-memory buffer to file and clear buffer

        Returns:
            buffer index of saved buffer file
        """
        if not self.eids:
            return self.buffer_idx

        buffer_file = self.get_buffer_file(self.buffer_idx)
        np.savez_compressed(
            buffer_file,
            eids=np.array(self.eids, dtype=object),
            embs=np.array(self.embs, dtype=np.float32),
            group_size=self.group_size,
        )

        # update tracker
        self.tracker.update(
            file_count=self.buffer_idx + 1,
            last_file=buffer_file.name,
            last_file_samples=len(self.eids) // self.group_size,
        )
        self.tracker.save()

        saved_buffer_idx = self.buffer_idx
        self.buffer_idx += 1

        # clear in-memory buffers
        self.eids.clear()
        self.embs.clear()

        return saved_buffer_idx

    def load_all(self) -> tuple[np.ndarray, np.ndarray, list[Path]] | None:
        """Load and concatenate all buffer files

        Returns:
            tuple of (eids, embs, buffer_files) or None if no buffers
        """
        buffer_files = sorted(self.buffers_dir.glob("buffer_*.npz"))
        if not buffer_files:
            logger.warn("* No buffer files to merge")
            return None

        all_eids = []
        all_embs = []
        for buffer_file in buffer_files:
            data = np.load(buffer_file, allow_pickle=True)
            all_eids.append(data["eids"])
            all_embs.append(data["embs"])

        return np.concatenate(all_eids), np.concatenate(all_embs), buffer_files

    def clear_files(self, buffer_files: list[Path]):
        """Clear buffer files and reset tracker"""
        for buffer_file in buffer_files:
            buffer_file.unlink()
        logger.mesg(f"  * Cleared buffer files: {len(buffer_files)}")

        # reset buffer index and tracker
        self.buffer_idx = 0
        self.tracker.update(file_count=0, last_file="", last_file_samples=0)
        self.tracker.save()


class ShardsWriter:
    """Writer for shard files with tracking support"""

    def __init__(
        self,
        shards_dir: Path,
        group_size: int,
        shard_size: int,
    ):
        """
        Args:
            shards_dir: directory for shard files
            group_size: number of items per sample (1 anchor + num_positives)
            shard_size: number of samples per shard file
        """
        self.shards_dir = shards_dir
        self.group_size = group_size
        self.shard_size = shard_size
        # calculate idx_width: max shards = MAX_SAMPLES / shard_size + 1
        max_shards = MAX_SAMPLES // shard_size + 1
        self.idx_width = int_bits(max_shards)

        # tracker
        self.tracker = ShardsTracker(shards_dir, group_size, shard_size).load()
        self.tracker.log_info()

    def shard_idx_to_suffix(self, shard_idx: int) -> str:
        """Convert shard index to zero-padded suffix string"""
        return f"{shard_idx:0{self.idx_width}d}"

    def get_shard_file(self, shard_idx: int) -> Path:
        """Get shard file path for a given shard index"""
        return self.shards_dir / f"shard_{self.shard_idx_to_suffix(shard_idx)}.npz"

    def next_merge_size(self) -> int:
        """Calculate the target number of samples to trigger next shard merge

        In filling mode: need (shard_size - last_shard_samples) to fill the last shard
        In normal mode: need shard_size for a new full shard

        Returns:
            Number of buffered samples that will trigger the next merge
        """
        fill_target = self.tracker.get_fill_target()
        if fill_target > 0:
            return fill_target
        return self.shard_size

    def fill_partial_shard(
        self, new_eids: np.ndarray, new_embs: np.ndarray
    ) -> tuple[int, int]:
        """Fill up the last partial shard with new data

        Args:
            new_eids: new eids data from buffers
            new_embs: new embs data from buffers

        Returns:
            tuple of (written_samples, consumed_items)
        """
        last_shard_idx = self.tracker.file_count - 1
        samples_needed = self.tracker.get_fill_target()
        items_needed = samples_needed * self.group_size

        # load last shard
        last_shard_file = self.get_shard_file(last_shard_idx)
        last_shard_data = np.load(last_shard_file, allow_pickle=True)
        last_eids = last_shard_data["eids"]
        last_embs = last_shard_data["embs"]

        # fill up the last shard
        actual_items = min(items_needed, len(new_eids))
        filled_eids = np.concatenate([last_eids, new_eids[:actual_items]])
        filled_embs = np.concatenate([last_embs, new_embs[:actual_items]])

        # overwrite the last shard
        np.savez_compressed(
            last_shard_file,
            eids=filled_eids,
            embs=filled_embs,
        )

        filled_samples = actual_items // self.group_size
        total_shard_samples = len(filled_eids) // self.group_size
        logger.mesg(
            f"  * Filled last shard {self.shard_idx_to_suffix(last_shard_idx)}: "
            f"{self.tracker.last_file_samples} + {filled_samples} = {logstr.okay(brk(total_shard_samples))}"
        )

        # update tracker
        self.tracker.update(
            file_count=self.tracker.file_count,
            last_file=last_shard_file.name,
            last_file_samples=total_shard_samples,
            total_samples=self.tracker.total_samples + filled_samples,
        )

        return filled_samples, actual_items

    def write_new_shards(
        self,
        eids: np.ndarray,
        embs: np.ndarray,
        force: bool = False,
    ) -> tuple[int, int]:
        """Write new shards from data

        Args:
            eids: eids data to write
            embs: embs data to write
            force: if True, write partial shard; if False, only write full shards

        Returns:
            tuple of (written_samples, consumed_items)
        """
        num_samples = len(eids) // self.group_size
        if num_samples == 0:
            return 0, 0

        items_per_shard = self.shard_size * self.group_size
        num_full_shards = num_samples // self.shard_size
        remainder_samples = num_samples % self.shard_size

        # decide how many shards to write
        if force and remainder_samples > 0:
            num_shards_to_write = num_full_shards + 1
        else:
            num_shards_to_write = num_full_shards

        if num_shards_to_write == 0:
            if remainder_samples > 0:
                logger.mesg(
                    f"  * Not enough data for full shard, keeping {remainder_samples} samples in buffer"
                )
            return 0, 0

        written_samples = 0
        start_shard_idx = self.tracker.file_count

        for i in range(num_shards_to_write):
            shard_idx = start_shard_idx + i
            start_item = i * items_per_shard
            end_item = min((i + 1) * items_per_shard, len(eids))

            shard_file = self.get_shard_file(shard_idx)
            np.savez_compressed(
                shard_file,
                eids=eids[start_item:end_item],
                embs=embs[start_item:end_item],
            )

            shard_samples = (end_item - start_item) // self.group_size
            logger.mesg(
                f"  * Saved to shard {self.shard_idx_to_suffix(shard_idx)}: {logstr.okay(brk(shard_samples))}"
            )
            written_samples += shard_samples

            # update tracker after each shard
            self.tracker.update(
                file_count=shard_idx + 1,
                last_file=shard_file.name,
                last_file_samples=shard_samples,
                total_samples=self.tracker.total_samples + shard_samples,
            )

        consumed_items = written_samples * self.group_size
        return written_samples, consumed_items


class ShardsReader:
    """Reader for shard files with LRU caching

    Handles loading shards on demand with memory-efficient LRU cache.

    Implementation Details:
    - Shards contain groups of (anchor, pos1, pos2, ...) stored sequentially
    - Sample indexing is 0-based for shard storage but 1-based for external API
    - LRU cache prevents memory overflow when dataset has many shards
    - Each shard contains shard_size samples, each sample has group_size items
    """

    # Maximum shards to keep in memory (LRU cache)
    # Each shard ~1GB, then 50 shards consume ~50GB memory
    # This prevents constant shard eviction during shuffled batch loading
    MAX_CACHED_SHARDS = 50

    def __init__(
        self,
        shards_dir: Path,
        group_size: int,
        shard_size: int,
    ):
        """
        Args:
            shards_dir: directory for shard files
            group_size: number of items per sample (1 anchor + num_positives)
            shard_size: number of samples per shard file
        """
        self.shards_dir = shards_dir
        self.group_size = group_size
        self.shard_size = shard_size
        # calculate idx_width: max shards = MAX_SAMPLES / shard_size + 1
        max_shards = MAX_SAMPLES // shard_size + 1
        self.idx_width = int_bits(max_shards)

        # LRU shard cache: {shard_idx: {"eids": ..., "embs": ..., "access_order": int}}
        self._shard_cache = {}
        self._access_counter = 0

    def shard_idx_to_suffix(self, shard_idx: int) -> str:
        """Convert shard index to zero-padded suffix string"""
        return f"{shard_idx:0{self.idx_width}d}"

    def get_shard_file(self, shard_idx: int) -> Path:
        """Get shard file path for a given shard index"""
        return self.shards_dir / f"shard_{self.shard_idx_to_suffix(shard_idx)}.npz"

    def load_shard(self, shard_idx: int):
        """Load a shard into cache with LRU eviction"""
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

        shard_file = self.get_shard_file(shard_idx)
        if not shard_file.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_file}")

        data = np.load(shard_file, allow_pickle=True)
        self._access_counter += 1
        self._shard_cache[shard_idx] = {
            "eids": data["eids"],
            "embs": data["embs"],
            "access_order": self._access_counter,
        }

    def get_shard_for_sample(self, sample_idx: int) -> tuple[int, int]:
        """Get shard index and item offset for a sample

        The mapping formula:
        - shard_idx = sample_idx // shard_size (which shard file)
        - offset = (sample_idx % shard_size) * group_size (item position within shard)

        Note: offset is in ITEMS not SAMPLES, because each sample has group_size items

        Example: shard_size=100000, group_size=3, sample_idx=250000
        - shard_idx = 250000 // 100000 = 2 (third shard file)
        - offset = (250000 % 100000) * 3 = 50000 * 3 = 150000 (item index in shard)

        Returns:
            (shard_idx, item_offset_in_shard)
        """
        shard_idx = sample_idx // self.shard_size
        offset = (sample_idx % self.shard_size) * self.group_size
        return shard_idx, offset

    def get_shard_data(self, shard_idx: int) -> dict:
        """Get shard data, loading if necessary

        Returns:
            dict with "eids" and "embs" arrays
        """
        self.load_shard(shard_idx)
        return self._shard_cache[shard_idx]

    def get_num_shards(self) -> int:
        """Get total number of shard files by scanning the directory"""
        return len(list(self.shards_dir.glob("shard_*.npz")))

    def preload_all(self):
        """Preload all shards into memory

        Warning: This may consume a lot of memory for large datasets.
        """
        num_shards = self.get_num_shards()
        logger.warn(
            f"* Loading all {num_shards} shards into memory. "
            f"Use iter_samples() for memory-efficient access."
        )
        # temporarily disable LRU eviction
        original_max = self.MAX_CACHED_SHARDS
        self.MAX_CACHED_SHARDS = num_shards + 1
        for shard_idx in range(num_shards):
            self.load_shard(shard_idx)
        self.MAX_CACHED_SHARDS = original_max


class TrainSamplesManager:
    """Manager for training samples storage and retrieval

    Handles both writing (during construction) and reading (during training):
    - Writing: buffer samples, flush to files, merge into shards
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

    def __init__(
        self,
        data_dir: str | Path,
        num_positives: int = 2,
        shard_size: int = 100_000,
        buffer_size: int = 1000,
        mode: str = "read",
    ):
        """
        Args:
            data_dir: directory for storing/loading training data
            num_positives: number of positive samples per anchor (for write mode)
            shard_size: number of samples per shard file (for write mode)
            buffer_size: buffer size before flushing to file (for write mode)
            mode: "read" for loading exist data, "write" for creating new data
        """
        self.data_dir = Path(data_dir)
        self.meta_file = self.data_dir / "meta.json"
        self.mode = mode

        if mode == "read":
            self._init_read_mode()
        else:
            self._init_write_mode(num_positives, shard_size, buffer_size)

    def is_read_mode(self) -> bool:
        return self.mode == "read"

    def is_write_mode(self) -> bool:
        return self.mode == "write"

    def _init_read_mode(self):
        """initialize for reading exist data"""
        meta_file = self.meta_file
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        # load metadata
        with open(meta_file, "r") as f:
            meta = json.load(f)
        self.group_size = int(meta["group_size"])
        self.num_positives = int(meta["num_positives"])
        self.num_samples = int(meta["num_samples"])
        self.num_shards = int(meta["num_shards"])
        self.shard_size = int(meta["shard_size"])
        self.emb_dim = int(meta["emb_dim"])

        # initialize shard reader
        shards_dir = self.data_dir / "shards"
        self.shards_reader = ShardsReader(
            shards_dir,
            self.group_size,
            self.shard_size,
        )

        logger.note(f"> Loaded TrainSamplesManager:")
        info_dict = {
            "num_samples": self.num_samples,
            "num_shards": self.num_shards,
            "group_size": self.group_size,
            "shard_size": self.shard_size,
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

    def _init_write_mode(self, num_positives: int, shard_size: int, buffer_size: int):
        """initialize for writing new data"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        buffers_dir = self.data_dir / "buffers"
        shards_dir = self.data_dir / "shards"
        buffers_dir.mkdir(parents=True, exist_ok=True)
        shards_dir.mkdir(parents=True, exist_ok=True)

        self.num_positives = num_positives
        self.group_size = 1 + num_positives  # anchor + positives
        self.shard_size = shard_size

        # initialize writers
        self.buffers_writer = BuffersWriter(
            buffers_dir,
            self.group_size,
            buffer_size,
            shard_size,
        )
        self.shards_writer = ShardsWriter(
            shards_dir,
            self.group_size,
            shard_size,
        )

        # queried eids cache (to avoid re-querying same bvids)
        self.queried_cache_file = self.data_dir / "queried_eids.pkl"
        self.queried_eids = self._load_queried_cache()

    # ========== Write mode methods ==========

    def _load_queried_cache(self) -> set:
        """load cache of already queried eids"""
        if self.queried_cache_file.exists():
            try:
                with open(self.queried_cache_file, "rb") as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.warn(f"× Error loading queried_eids.pkl: {e}, rebuilding ...")
                return self._rebuild_queried_cache()
        return set()

    def _rebuild_queried_cache(self) -> set:
        """Rebuild queried eids cache from exist samples count"""
        # get total samples from shards and buffers
        total_samples = (
            self.shards_writer.tracker.total_samples
            + self.buffers_writer.tracker.get_total_samples()
        )
        if total_samples == 0:
            logger.mesg("  * No exist samples, starting fresh")
            return set()

        logger.note(
            f"> Rebuilding queried_eids from exist samples: {logstr.file(brk(total_samples))}"
        )

        # scan redis keys to get first N bvids
        redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        queried_eids = set()
        for batch_keys in redis.scan_keys(
            pattern=REDIS_PT,
            batch_size=1000,
            max_count=total_samples,
        ):
            bvids = redis_keys_to_bvids(batch_keys)
            queried_eids.update(bvids)
            if len(queried_eids) >= total_samples:
                break

        # truncate to exact count (scan may return slightly more)
        if len(queried_eids) > total_samples:
            queried_eids = set(list(queried_eids)[:total_samples])

        print()
        logger.okay(f"+ Rebuilt  queried eids: {brk(len(queried_eids))}")

        # save rebuilt cache
        with open(self.queried_cache_file, "wb") as f:
            pickle.dump(queried_eids, f, protocol=pickle.HIGHEST_PROTOCOL)

        return queried_eids

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

        Returns:
            True if buffer was flushed after appending
        """
        return self.buffers_writer.append(group_eids, group_embs)

    def should_merge_buffers(self) -> bool:
        """check if buffer files should be merged into shards"""
        buffered_samples = self.buffers_writer.get_total_sample_count()
        return buffered_samples >= self.shards_writer.next_merge_size()

    def merge_buffers(self, force: bool = False):
        """merge buffer files into sharded training data files

        Creates sharded NPZ files:
        - shards/shard_00000.npz, shard_00001.npz, ...
        - meta.json: metadata

        Args:
            force:
            - if True, write all data including partial shard (used in finalize)
            - if False, only write full shards
        """
        # Step 1: Load all buffers
        result = self.buffers_writer.load_all()
        if result is None:
            return
        new_eids, new_embs, buffer_files = result
        new_samples = len(new_eids) // self.group_size

        logger.store_indent()
        logger.indent(2)

        print()
        logger.note("> Merge buffer files into shards:")

        written_samples = 0
        total_consumed_items = 0

        # Step 2: Fill partial last shard if in filling mode
        if self.shards_writer.tracker.is_filling_mode():
            filled_samples, consumed_items = self.shards_writer.fill_partial_shard(
                new_eids, new_embs
            )
            written_samples += filled_samples
            total_consumed_items += consumed_items

        # Step 3: Write remaining data into new shards
        remaining_eids = new_eids[total_consumed_items:]
        remaining_embs = new_embs[total_consumed_items:]
        shard_written_samples, shard_consumed_items = (
            self.shards_writer.write_new_shards(
                remaining_eids, remaining_embs, force=force
            )
        )
        written_samples += shard_written_samples
        total_consumed_items += shard_consumed_items

        # Step 4: Save shards tracker and metadata
        self.shards_writer.tracker.save()
        self._save_metadata()

        # Step 5: Handle remaining data not written to shards
        leftover_eids = new_eids[total_consumed_items:]
        leftover_embs = new_embs[total_consumed_items:]
        leftover_samples = len(leftover_eids) // self.group_size

        # Step 6: Clear buffer files
        self.buffers_writer.clear_files(buffer_files)

        # Step 7: If there's leftover data, save it back to buffer
        if leftover_samples > 0:
            self.buffers_writer.eids = list(leftover_eids)
            self.buffers_writer.embs = list(leftover_embs)
            self.buffers_writer.flush()
            logger.mesg(f"  * Saved {leftover_samples} leftover samples back to buffer")

        # Step 8: Log summary
        tracker = self.shards_writer.tracker
        info_dict = {
            "new_samples": new_samples,
            "written_samples": written_samples,
            "leftover_samples": leftover_samples,
            "total_shards": tracker.file_count,
            "total_samples": tracker.total_samples,
            "last_shard_samples": f"{tracker.last_file_samples} ({round(tracker.last_file_samples / self.shard_size * 100, 1)}%)",
        }
        logger.mesg(dict_to_str(info_dict), indent=2)
        logger.restore_indent()

    def _save_metadata(self):
        """Save metadata file with current state"""
        tracker = self.shards_writer.tracker
        meta = {
            "group_size": self.group_size,
            "num_positives": self.num_positives,
            "num_samples": tracker.total_samples,
            "num_shards": tracker.file_count,
            "shard_size": self.shard_size,
            "emb_dim": self.emb_dim,
        }
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        # update instance state
        self.num_samples = tracker.total_samples
        self.num_shards = tracker.file_count

    def finalize(self):
        """finalize writing: flush remaining buffer and merge buffers"""
        if self.buffers_writer.eids:
            self.buffers_writer.flush()
        self.save_queried_cache()
        self.merge_buffers(force=True)

    # ========== Read mode methods ==========

    def preload_all(self):
        """preload all shards into memory

        Warning: This may consume a lot of memory for large datasets.
        Consider using iter_samples() for memory-efficient sequential access.
        """
        self.shards_reader.preload_all()

    def __len__(self) -> int:
        """return number of valid samples (all samples including 0)"""
        return self.num_samples

    def __getitem__(self, sample_idx: int) -> dict:
        """get training sample at index

        Index Convention:
        - Uses 0-based indexing: valid range [0, num_samples)
        - All samples are valid, including sample 0

        Sample Structure:
        - anchor: first item in group (offset + 0)
        - positives: items 1 to group_size-1 (offset + 1, offset + 2, ...)
        - negative: previous sample's anchor, or sample 1's anchor for sample 0

        Special Case:
        - sample_idx=0: uses sample 1's anchor as negative (safe since num_samples >= 2)
        - This is the ONLY place where sample 0 is handled specially

        Args:
            sample_idx: valid range is [0, num_samples)

        Returns:
            dict with anchor, positives, and negative
        """
        if not (0 <= sample_idx < self.num_samples):
            raise IndexError(
                f"sample_idx {sample_idx} out of range [0, {self.num_samples})"
            )

        # get shard info
        shard_idx, shard_offset = self.shards_reader.get_shard_for_sample(sample_idx)
        shard = self.shards_reader.get_shard_data(shard_idx)

        # anchor (first in group)
        anchor_eid = shard["eids"][shard_offset]
        anchor_emb = shard["embs"][shard_offset]

        # positives (rest in group)
        pos_eids = [shard["eids"][shard_offset + j] for j in range(1, self.group_size)]
        pos_embs = np.array(
            [shard["embs"][shard_offset + j] for j in range(1, self.group_size)]
        )

        # special case is sample 0, which uses sample 1's anchor,
        # others use previous sample's anchor
        neg_sample_idx = 1 if sample_idx == 0 else sample_idx - 1
        neg_shard_idx, neg_shard_offset = self.shards_reader.get_shard_for_sample(
            neg_sample_idx
        )

        # load negative shard if different from current shard
        if neg_shard_idx != shard_idx:
            neg_shard = self.shards_reader.get_shard_data(neg_shard_idx)
        else:
            neg_shard = shard

        neg_eid = neg_shard["eids"][neg_shard_offset]
        neg_emb = neg_shard["embs"][neg_shard_offset]

        return {
            "anchor_eid": anchor_eid,
            "anchor_emb": anchor_emb,
            "pos_eids": pos_eids,
            "pos_embs": pos_embs,
            "neg_eid": neg_eid,
            "neg_emb": neg_emb,
        }

    def iter_samples(self, start: int = 0, end: int = None) -> Iterator[dict]:
        """iterate through samples sequentially (memory efficient)

        This method efficiently iterates through samples by leveraging the LRU cache
        in ShardsReader. Sequential access ensures optimal cache hit rate.

        Index Convention:
        - Uses 0-based indexing (consistent with __getitem__)
        - Default start=0 to include all samples
        - end is exclusive (Python convention)

        Args:
            start: starting sample index (default 0 for all samples)
            end: ending sample index (exclusive, default None for all)

        Raises:
            ValueError: if called in write mode
        """
        if not self.is_read_mode():
            raise ValueError("iter_samples() is only available in read mode")

        if end is None:
            end = self.num_samples

        # validate range
        start = max(0, start)
        end = min(end, self.num_samples)

        for sample_idx in range(start, end):
            yield self[sample_idx]

    def iter_shard_samples(self, shard_idx: int) -> Iterator[dict]:
        """iterate through all samples in a specific shard

        Useful for parallel loading in distributed training where each worker
        processes a specific shard.

        Example: shard_size=100000, num_samples=3000000
        - Shard 0: samples [0, 100000)
        - Shard 1: samples [100000, 200000)
        - Shard 2: samples [200000, 300000)

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

        for sample_idx in range(start_sample, end_sample):
            yield self[sample_idx]

    def get_shard_range(self, shard_idx: int) -> tuple[int, int]:
        """get sample index range for a shard

        Returns:
            (start_sample_idx, end_sample_idx) - both 0-based
        """
        start = shard_idx * self.shard_size
        end = min((shard_idx + 1) * self.shard_size, self.num_samples)
        return start, end

    def get_stats(self) -> dict:
        """get statistics about the training data"""
        return {
            "num_samples": self.num_samples,
            "num_shards": self.num_shards,
            "shard_size": self.shard_size,
            "group_size": self.group_size,
            "num_positives": self.num_positives,
            "cached_shards": len(self.shards_reader._shard_cache),
            "max_cached_shards": self.shards_reader.MAX_CACHED_SHARDS,
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
            samples_count: target total number of training samples (including exist)
            query_batch_size: batch size for querying Faiss tops() (default: 50)
        """
        if not manager.is_write_mode():
            raise ValueError("TrainSamplesConstructor requires manager in write mode")

        self.manager = manager
        self.samples_count = samples_count
        self.num_positives = manager.num_positives
        self.topk = self.num_positives + 1  # top1 (anchor) + num_positives
        self.query_batch_size = query_batch_size

        # get exist samples count from shards tracker
        self.exist_samples = manager.shards_writer.tracker.total_samples

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

        # append to manager (may auto-flush if buffer is full)
        flushed = self.manager.append_sample(group_eids, group_embs)
        if flushed:
            self.manager.save_queried_cache()
            if self.manager.should_merge_buffers():
                self.manager.merge_buffers()
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

    def _get_current_total_samples(self) -> int:
        """Get current total samples (exist + built in this run)"""
        return self.exist_samples + self.stats["built_count"]

    def _get_remaining_samples(self) -> int:
        """Get number of samples still needed to reach target"""
        return max(0, self.samples_count - self._get_current_total_samples())

    def _should_stop(self) -> bool:
        """Check if total samples count reached samples_count target"""
        return self._get_current_total_samples() >= self.samples_count

    def run(self):
        """main execution: construct training tuples"""
        self._log_start_info()

        # check if already at target
        if self._should_stop():
            logger.note(
                f"> Already have {self.exist_samples} samples, "
                f"target is {self.samples_count}. Nothing to do."
            )
            return

        # initialize state
        total_built = 0
        samples_to_build = self._get_remaining_samples()

        # create progress bar (tracks built count towards target)
        self.bar = TCLogbar(
            total=samples_to_build,
            desc=logstr.note("* Construct samples"),
            show_iter_per_second=True,
        )

        # scan and process Redis keys in batches
        for batch_keys in self.redis.scan_keys(
            pattern=REDIS_PT,
            batch_size=self.query_batch_size,
            max_count=self.samples_count,
        ):
            all_bvids = redis_keys_to_bvids(batch_keys)

            # update scanned count (all eids, including queried)
            self.stats["scanned_count"] += len(all_bvids)

            # filter out already queried eids
            bvid_batch = [
                bvid for bvid in all_bvids if not self.manager.is_queried(bvid)
            ]

            # truncate batch if it would exceed target samples
            remaining = self._get_remaining_samples()
            if len(bvid_batch) > remaining:
                bvid_batch = bvid_batch[:remaining]

            # count already queried as skipped
            already_queried = len(all_bvids) - len(bvid_batch)
            self.stats["skip_count"] += already_queried

            if not bvid_batch:
                if self._should_stop():
                    break
                continue

            # query Faiss and build samples
            # (buffer flush, queried cache save, and merge are handled inside)
            built_in_batch, _ = self._query_and_build_samples(bvid_batch)
            total_built += built_in_batch

            # update progress bar based on built samples
            self.bar.update(increment=built_in_batch)

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
            "target_samples": self.samples_count,
            "exist_samples": self.exist_samples,
            "samples_to_build": self._get_remaining_samples(),
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
            "group_size": f"3 (1 anchor + {self.num_positives} positives)",
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


class TrainSamplesManagerTester:
    """Test class for TrainSamplesManager read/iteration functionality

    Tests:
    - Random access via __getitem__
    - Sequential iteration via iter_samples
    - Shard iteration via iter_shard_samples
    - Sample 0 special handling (uses sample 1's anchor as negative)
    - Cross-shard access patterns
    - EID and embedding consistency
    """

    def __init__(self, data_dir: str | Path):
        """
        Args:
            data_dir: directory containing training data
        """
        self.data_dir = Path(data_dir)
        self.manager = TrainSamplesManager(data_dir=data_dir, mode="read")

    def run_all_tests(self):
        """Run all test cases"""
        logger.note("> Running TrainSamplesManager Tests:")
        self._test_basic_info()
        self._test_random_access()
        self._test_sample_zero()
        self._test_sequential_iteration()
        self._test_shard_iteration()
        self._test_cross_shard_access()
        self._test_boundary_cases()
        logger.okay("✓ All tests passed!")

    def _test_basic_info(self):
        """Test basic manager properties"""
        logger.mesg("* Test 1: Basic Info")
        stats = self.manager.get_stats()
        logger.mesg(dict_to_str(stats), indent=2)
        assert self.manager.num_samples > 0, "num_samples should be > 0"
        assert (
            len(self.manager) == self.manager.num_samples
        ), "__len__ should match num_samples"
        assert (
            self.manager.group_size == 1 + self.manager.num_positives
        ), "group_size mismatch"

        logger.okay(
            f"✓ Passed: {self.manager.num_samples} samples, {self.manager.num_shards} shards"
        )
        print()

    def _test_random_access(self):
        """Test __getitem__ random access"""
        logger.mesg("* Test 2: Random Access (__getitem__)")

        # test first sample (index 1 to avoid sample 0 special case)
        if self.manager.num_samples > 1:
            sample_idx = 1
            sample = self.manager[sample_idx]
            info_dict = {
                "sample_idx": sample_idx,
                "anchor_eid": sample["anchor_eid"],
                "anchor_emb_shape": sample["anchor_emb"].shape,
                "pos_eids": sample["pos_eids"],
                "pos_embs_shape": sample["pos_embs"].shape,
                "neg_eid": sample["neg_eid"],
                "neg_emb_shape": sample["neg_emb"].shape,
            }
            logger.mesg(dict_to_str(info_dict), indent=2)

            # validate structure
            assert isinstance(sample["anchor_eid"], str), "anchor_eid should be string"
            assert sample["anchor_emb"].shape == (
                self.manager.emb_dim,
            ), "anchor_emb shape mismatch"
            assert (
                len(sample["pos_eids"]) == self.manager.num_positives
            ), "pos_eids count mismatch"
            assert sample["pos_embs"].shape == (
                self.manager.num_positives,
                self.manager.emb_dim,
            ), "pos_embs shape mismatch"
            assert sample["neg_emb"].shape == (
                self.manager.emb_dim,
            ), "neg_emb shape mismatch"

            logger.okay(f"✓ Passed: structure validated")
        else:
            logger.warn(f"- Skipped: only {self.manager.num_samples} sample(s)")
        print()

    def _test_sample_zero(self):
        """Test sample 0 special handling (uses sample 1's anchor as negative)"""
        logger.mesg("* Test 3: Sample 0 (Uses Sample 1 Anchor as Negative)")

        sample_0 = self.manager[0]
        info_dict = {
            "sample_idx": 0,
            "anchor_eid": sample_0["anchor_eid"],
            "anchor_emb_shape": sample_0["anchor_emb"].shape,
            "pos_eids": sample_0["pos_eids"],
            "pos_embs_shape": sample_0["pos_embs"].shape,
            "neg_eid": sample_0["neg_eid"],
            "neg_emb_shape": sample_0["neg_emb"].shape,
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

        # validate special handling: sample 0 uses sample 1's anchor as negative
        sample_1 = self.manager[1]
        assert (
            sample_0["neg_eid"] == sample_1["anchor_eid"]
        ), "sample 0 neg_eid should match sample 1's anchor_eid"
        assert np.allclose(
            sample_0["neg_emb"], sample_1["anchor_emb"]
        ), "sample 0 neg_emb should match sample 1's anchor_emb"
        assert sample_0["neg_emb"].shape == (
            self.manager.emb_dim,
        ), "sample 0 neg_emb shape mismatch"

        logger.mesg(
            f"  * Verified: neg matches sample 1 anchor ({sample_1['anchor_eid']})"
        )
        logger.okay(f"✓ Passed: sample 0 uses sample 1's anchor as negative")
        print()

    def _test_sequential_iteration(self):
        """Test iter_samples sequential iteration"""
        logger.mesg("* Test 4: Sequential Iteration (iter_samples)")

        # iterate through first 10 samples (or all if fewer)
        num_to_iterate = min(10, self.manager.num_samples)
        logger.mesg(f"  * Iterating first {num_to_iterate} samples:")
        count = 0
        prev_neg_eid = None
        for i, sample in enumerate(
            self.manager.iter_samples(start=0, end=num_to_iterate)
        ):
            if i == 0:
                # sample 0: verify negative from sample 1
                assert (
                    sample["neg_eid"] == self.manager[1]["anchor_eid"]
                ), f"sample 0 negative mismatch"
            else:
                # sample i (i > 0): negative should be previous anchor
                assert (
                    sample["neg_eid"] == prev_neg_eid
                ), f"sample {i} negative mismatch"

            prev_neg_eid = sample["anchor_eid"]
            count += 1
            # print first 3 samples
            if i >= 3:
                continue
            logger.mesg(
                f"  - [{i}] anchor={sample['anchor_eid']}, neg={sample['neg_eid']}"
            )

        assert count == num_to_iterate, "iteration count mismatch"
        logger.okay(
            f"✓ Passed: {count} samples iterated, negative consistency verified"
        )
        print()

    def _test_shard_iteration(self):
        """Test iter_shard_samples for specific shard"""
        logger.mesg("* Test 5: Shard Iteration (iter_shard_samples)")

        # test first shard
        shard_idx = 0
        start_sample, end_sample = self.manager.get_shard_range(shard_idx)
        logger.mesg(f"  * Shard {shard_idx} range: [{start_sample}, {end_sample})")

        count = 0
        first_eid = None
        last_eid = None
        for sample in self.manager.iter_shard_samples(shard_idx):
            if count == 0:
                first_eid = sample["anchor_eid"]
            last_eid = sample["anchor_eid"]
            count += 1

        expected_count = end_sample - start_sample
        assert (
            count == expected_count
        ), f"shard iteration count mismatch: {count} != {expected_count}"

        info_dict = {
            "shard_idx": shard_idx,
            "expected_count": expected_count,
            "actual_count": count,
            "first_eid": first_eid,
            "last_eid": last_eid,
        }
        logger.mesg(dict_to_str(info_dict), indent=2)

        logger.okay(f"✓ Passed: {count} samples in shard {shard_idx}")
        print()

    def _test_cross_shard_access(self):
        """Test accessing samples across shard boundaries"""
        logger.mesg("* Test 6: Cross-Shard Access")

        if self.manager.num_shards < 2:
            logger.warn(f"- Skipped: only {self.manager.num_shards} shard(s)")
            print()
            return

        # access sample at boundary of first two shards
        boundary_idx = self.manager.shard_size
        if boundary_idx < self.manager.num_samples:
            sample_before = self.manager[boundary_idx - 1]
            sample_after = self.manager[boundary_idx]

            logger.mesg(f"  * Boundary at sample {boundary_idx}:")
            info_dict = {
                "sample_before_idx": boundary_idx - 1,
                "sample_before_anchor_eid": sample_before["anchor_eid"],
                "sample_after_idx": boundary_idx,
                "sample_after_anchor_eid": sample_after["anchor_eid"],
                "sample_after_neg_eid": sample_after["neg_eid"],
            }
            logger.mesg(dict_to_str(info_dict), indent=2)

            # verify cross-shard negative reference
            assert (
                sample_after["neg_eid"] == sample_before["anchor_eid"]
            ), "cross-shard negative mismatch"

            logger.okay(f"✓ Passed: cross-shard negative reference correct")
        else:
            logger.warn(
                f"- Skipped: boundary_idx {boundary_idx} >= num_samples {self.manager.num_samples}"
            )
        print()

    def _test_boundary_cases(self):
        """Test boundary cases and error handling"""
        logger.mesg("* Test 7: Boundary Cases")

        # test valid boundaries
        first_sample = self.manager[0]
        last_sample = self.manager[self.manager.num_samples - 1]
        logger.mesg(f"  * First sample (0): {first_sample['anchor_eid']}")
        logger.mesg(
            f"  * Last sample ({self.manager.num_samples - 1}): {last_sample['anchor_eid']}"
        )

        # test out-of-bounds
        try:
            _ = self.manager[-1]
            logger.fail(f"✗ Failed: should raise IndexError for negative index")
        except IndexError:
            logger.okay(f"✓ Passed: negative index raises IndexError")

        try:
            _ = self.manager[self.manager.num_samples]
            logger.fail(f"✗ Failed: should raise IndexError for index >= num_samples")
        except IndexError:
            logger.okay(f"✓ Passed: index >= num_samples raises IndexError")
        print()


class TrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-cc", "--calc", action="store_true")
        self.add_argument("-bf", "--build-faiss", action="store_true")
        self.add_argument("-tf", "--test-faiss", action="store_true")
        self.add_argument("-cs", "--construct-samples", action="store_true")
        self.add_argument("-ts", "--test-samples", action="store_true")
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
            buffer_size=1000,
            mode="write",
        )
        constructor = TrainSamplesConstructor(
            manager=manager,
            samples_count=args.samples_count,
        )
        constructor.run()

    if args.test_samples:
        tester = TrainSamplesManagerTester(data_dir=SAMPLES_DIR)
        tester.run_all_tests()


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

    # Case: test training samples read/iteration functionality
    # python -m models.tembed.train -ts
