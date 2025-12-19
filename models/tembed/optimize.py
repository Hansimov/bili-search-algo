import argparse
import os
import time

from pathlib import Path
from rocksdict import (
    Rdict,
    Options,
    BlockBasedOptions,
    Cache,
    WriteOptions,
    WriteBatch,
    CompactOptions,
    BottommostLevelCompaction,
    DBCompressionType,
)
from tclogger import TCLogger, TCLogbar, logstr, brk, int_bits

from models.tembed.train import ROCKS_DB_PATH

logger = TCLogger()


class RocksOptimizer:
    """Optimize RocksDB for point-lookup performance.

    Key optimizations:
    1. Large block cache (default 8GB) - cache hot data in RAM
    2. Bloom filter (10 bits/key) - reduce unnecessary disk reads
    3. Cache index/filter blocks - avoid repeated metadata reads
    4. Pin L0 filter/index - prevent cache eviction of L0 metadata
    5. Manual compaction - reduce read amplification
    6. Larger target_file_size - reduce SST file count

    References:
    - rocksdict API: https://rocksdict.github.io/RocksDict/rocksdict.html
    - RocksDB Tuning Guide: https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide
    - Block Cache: https://github.com/facebook/rocksdb/wiki/Block-Cache
    - Bloom Filter: https://github.com/facebook/rocksdb/wiki/RocksDB-Bloom-Filter
    """

    def __init__(
        self,
        db_path: Path = None,
        cache_gb: int = 8,
        target_file_size_mb: int = 256,
        bits_per_key: int = 10,
    ):
        self.db_path = db_path or ROCKS_DB_PATH
        self.cache_gb = cache_gb
        self.target_file_size_mb = target_file_size_mb
        self.bits_per_key = bits_per_key
        self.db = None

    def build_optimized_options(self, for_write: bool = False) -> Options:
        """Build optimized RocksDB options for point-lookup workload.

        Args:
            for_write: If True, include write-time options (target_file_size, etc.)
                       These only affect new SST files, not existing ones.
        """
        opts = Options(raw_mode=False)
        opts.create_if_missing(True)

        # Parallelism: use CPU cores, but don't go too high on HDD
        parallelism = min(max(4, os.cpu_count() or 4), 8)
        opts.increase_parallelism(parallelism)
        opts.set_max_background_jobs(parallelism)

        # Compression: LZ4 is fast and reduces IO (good for HDD)
        opts.set_compression_type(DBCompressionType.lz4())

        if for_write:
            # Write-time options (only affect new SST files)
            # Larger target_file_size reduces total SST count
            opts.set_target_file_size_base(self.target_file_size_mb * 1024 * 1024)
            opts.set_write_buffer_size(64 * 1024 * 1024)

        # Block-based table options (affect reads immediately)
        table_opts = BlockBasedOptions()

        # Large block cache (default only 8MB, way too small for 80GB+ DB)
        # This is the most important optimization for random reads
        cache = Cache(self.cache_gb * 1024 * 1024 * 1024)
        table_opts.set_block_cache(cache)

        # Bloom filter: reduce unnecessary disk reads for point lookups
        # block_based=False means full filter (better for point queries)
        table_opts.set_bloom_filter(bits_per_key=self.bits_per_key, block_based=False)

        # Cache index and filter blocks in block cache
        # This helps when cache is large enough
        table_opts.set_cache_index_and_filter_blocks(True)

        # Pin L0 filter and index blocks to avoid eviction
        table_opts.set_pin_l0_filter_and_index_blocks_in_cache(True)

        opts.set_block_based_table_factory(table_opts)
        return opts

    def open_db(self, for_write: bool = False):
        """Open RocksDB with optimized options."""
        logger.note(f"> Opening RocksDB with optimized options:")
        logger.file(f"  * db_path: {self.db_path}")
        logger.mesg(f"  * cache_gb: {logstr.file(brk(self.cache_gb))}")
        logger.mesg(
            f"  * target_file_size_mb: {logstr.file(brk(self.target_file_size_mb))}"
        )
        logger.mesg(f"  * bits_per_key: {logstr.file(brk(self.bits_per_key))}")

        opts = self.build_optimized_options(for_write=for_write)
        self.db = Rdict(path=str(self.db_path.resolve()), options=opts)
        logger.okay(f"  * [Opened]")

    def close_db(self):
        """Close RocksDB connection."""
        if self.db:
            try:
                self.db.close()
                logger.note(f"> RocksDB closed")
            except Exception:
                pass
            self.db = None

    def get_property_int(self, name: str) -> int:
        """Get integer property value from RocksDB."""
        try:
            return self.db.property_int_value(name)
        except Exception:
            return None

    def get_property_str(self, name: str) -> str:
        """Get string property value from RocksDB."""
        try:
            return self.db.property_value(name)
        except Exception:
            return None

    def dump_lsm_summary(self):
        """Dump brief LSM summary."""
        logger.note(f"> LSM Summary:")
        num_keys = self.get_property_int("rocksdb.estimate-num-keys")
        if num_keys:
            logger.mesg(f"  * keys: {logstr.file(brk(f'{num_keys:,}'))}")

        total_files = 0
        for lvl in range(7):
            val = self.get_property_str(f"rocksdb.num-files-at-level{lvl}")
            if val:
                count = int(val)
                total_files += count
                if count > 0:
                    logger.mesg(f"  * L{lvl}: {logstr.file(brk(count))} files")
        logger.mesg(f"  * Total: {logstr.file(brk(total_files))} files")

    def run_compaction(self):
        """Run full manual compaction to reduce read amplification.

        This is the most effective way to immediately improve read performance
        by reducing the number of SST files and organizing data better.

        WARNING: This can take a long time for large databases!
        """
        if not self.db:
            self.open_db(for_write=True)

        logger.note(f"> Running manual compaction...")
        logger.warn(f"  ! This may take a long time for large databases")

        self.dump_lsm_summary()

        # Configure compaction options
        compact_opts = CompactOptions()
        compact_opts.set_exclusive_manual_compaction(True)
        compact_opts.set_bottommost_level_compaction(
            BottommostLevelCompaction.force_optimized()
        )

        start_time = time.perf_counter()
        logger.mesg(f"  * Starting compaction...")

        # Run compaction on entire key range
        self.db.compact_range(None, None, compact_opts)

        elapsed = time.perf_counter() - start_time
        logger.okay(f"  * Compaction completed in {elapsed:.2f} seconds")

        logger.note(f"\n> After compaction:")
        self.dump_lsm_summary()

    def rebuild_db(self, new_db_path: Path = None):
        """Rebuild database with new optimized settings.

        This is necessary to apply write-time options (like target_file_size)
        to all data, not just new writes.

        Args:
            new_db_path: Path for the new database. If None, uses db_path + "_optimized"
        """
        if new_db_path is None:
            new_db_path = self.db_path.parent / f"{self.db_path.name}_optimized"

        logger.note(f"> Rebuilding database with optimized settings:")
        logger.file(f"  * source: {self.db_path}")
        logger.file(f"  * target: {new_db_path}")

        # Open source DB (read-only)
        logger.mesg(f"  * Opening source database...")
        src_opts = Options(raw_mode=False)
        src_opts.create_if_missing(False)
        src_db = Rdict(path=str(self.db_path.resolve()), options=src_opts)

        # Get total count for progress bar
        total_keys = src_db.property_int_value("rocksdb.estimate-num-keys") or 0
        logger.mesg(f"  * Source keys: {logstr.file(brk(f'{total_keys:,}'))}")

        # Open target DB with optimized options
        logger.mesg(f"  * Creating target database...")
        dst_opts = self.build_optimized_options(for_write=True)
        dst_db = Rdict(path=str(new_db_path.resolve()), options=dst_opts)

        # Copy all data
        logger.mesg(f"  * Copying data...")
        start_time = time.perf_counter()
        bar = TCLogbar(total=total_keys, desc="  * progress")
        copied = 0
        batch_size = 10000

        batch = {}
        for key, value in src_db.items():
            batch[key] = value
            copied += 1

            if len(batch) >= batch_size:
                # Use WriteBatch for efficient batch writes
                wb = WriteBatch()
                for k, v in batch.items():
                    wb.put(k, v)
                dst_db.write(wb)
                batch.clear()
                bar.update(batch_size)

        # Write remaining batch
        if batch:
            wb = WriteBatch()
            for k, v in batch.items():
                wb.put(k, v)
            dst_db.write(wb)
            bar.update(len(batch))
        print()

        elapsed = time.perf_counter() - start_time
        logger.okay(f"  * Copied {copied:,} keys in {elapsed:.2f} seconds")

        # Close databases
        src_db.close()
        dst_db.close()

        logger.note(f"\n> Rebuild complete!")
        logger.mesg(f"  * New database: {logstr.file(str(new_db_path))}")
        logger.mesg(f"  * To use it, update ROCKS_DB_PATH or rename directories")

    def show_recommendations(self):
        """Show optimization recommendations based on current state."""
        if not self.db:
            self.open_db(for_write=False)

        logger.note(f"\n{'='*60}")
        logger.note(f"> RocksDB Optimization Recommendations")
        logger.note(f"{'='*60}\n")

        self.dump_lsm_summary()

        # Check block cache
        cache_capacity = self.get_property_int("rocksdb.block-cache-capacity")
        if cache_capacity:
            cache_mb = cache_capacity / (1024 * 1024)
            logger.note(f"\n> Block Cache:")
            logger.mesg(f"  * current: {logstr.file(f'{cache_mb:.2f} MB')}")
            if cache_mb < 1000:
                logger.warn(f"  ! Block cache is too small for large DB")
                logger.warn(f"    → Recommend: 4-8 GB for 80GB+ database")
            else:
                logger.okay(f"  ✓ Block cache size looks good")

        # Check L0 files
        l0_files = self.get_property_str("rocksdb.num-files-at-level0")
        if l0_files:
            l0_count = int(l0_files)
            logger.note(f"\n> L0 Files:")
            logger.mesg(f"  * current: {logstr.file(brk(l0_count))}")
            if l0_count > 4:
                logger.warn(f"  ! High L0 file count increases read amplification")
                logger.warn(f"    → Run: python -m models.tembed.optimize -c")
            else:
                logger.okay(f"  ✓ L0 file count is healthy")

        # Recommendations summary
        logger.note(f"\n> Quick Actions:")
        logger.mesg(f"  1. Run compaction: python -m models.tembed.optimize -c")
        logger.mesg(
            f"  2. Rebuild with optimized settings: python -m models.tembed.optimize -r"
        )
        logger.mesg(f"  3. For best results, move database to SSD")


class OptimizeArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # db options
        self.add_argument(
            "-d", "--db-path", type=str, default=None, help="Path to RocksDB"
        )
        # optimization parameters
        self.add_argument(
            "--cache-gb", type=int, default=8, help="Block cache size in GB"
        )
        self.add_argument(
            "--target-file-size-mb",
            type=int,
            default=256,
            help="Target SST file size in MB",
        )
        self.add_argument(
            "--bits-per-key", type=int, default=10, help="Bloom filter bits per key"
        )
        # actions
        self.add_argument(
            "-c", "--compact", action="store_true", help="Run manual compaction"
        )
        self.add_argument(
            "-r", "--rebuild", action="store_true", help="Rebuild database"
        )
        self.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="Output path for rebuilt database",
        )
        self.add_argument(
            "-i", "--info", action="store_true", help="Show recommendations"
        )
        self.args, _ = self.parse_known_args()


def main():
    args = OptimizeArgParser().args

    db_path = Path(args.db_path) if args.db_path else None
    optimizer = RocksOptimizer(
        db_path=db_path,
        cache_gb=args.cache_gb,
        target_file_size_mb=args.target_file_size_mb,
        bits_per_key=args.bits_per_key,
    )

    try:
        if args.compact:
            optimizer.run_compaction()
        elif args.rebuild:
            output_path = Path(args.output) if args.output else None
            optimizer.rebuild_db(new_db_path=output_path)
        elif args.info:
            optimizer.show_recommendations()
        else:
            # Default: show recommendations
            optimizer.show_recommendations()
    finally:
        optimizer.close_db()


if __name__ == "__main__":
    main()

    # Show optimization recommendations (default)
    # python -m models.tembed.optimize
    # python -m models.tembed.optimize -i

    # Run manual compaction (reduce read amplification)
    # python -m models.tembed.optimize -c

    # Rebuild database with optimized settings
    # python -m models.tembed.optimize -r
    # python -m models.tembed.optimize -r -o /path/to/new_db

    # Custom parameters
    # python -m models.tembed.optimize -c --cache-gb 16
    # python -m models.tembed.optimize -r --target-file-size-mb 512

    # Custom db path
    # python -m models.tembed.optimize -d /path/to/rocksdb -c
