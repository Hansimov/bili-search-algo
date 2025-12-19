import argparse
import random
import re
import time

from pathlib import Path
from rocksdict import Rdict, Options, ReadOptions
from sedb import RedisOperator, RocksOperator
from tclogger import TCLogger, TCLogbar, logstr, brk, int_bits

from configs.envs import REDIS_ENVS
from models.tembed.train import ROCKS_DB_PATH, REDIS_PT, REDIS_PREFIX

logger = TCLogger()
ROCKS_DB_PATH = ROCKS_DB_PATH.parent / f"{ROCKS_DB_PATH.name}_optimized"


class RocksStatsAnalyzer:
    """Analyze RocksDB performance metrics and LSM structure.

    References:
    - rocksdict API: https://rocksdict.github.io/RocksDict/rocksdict.html
    - RocksDB Tuning Guide: https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide
    - RocksDB properties: https://docs.rs/rust-rocksdb/latest/rust_rocksdb/properties/index.html
    """

    # Maximum LSM levels to check
    MAX_LEVELS = 7

    def __init__(self, db_path: Path = None, verbose: bool = True):
        self.db_path = db_path or ROCKS_DB_PATH
        self.verbose = verbose
        self.init_processors()

    def init_processors(self):
        """Use RocksOperator to get optimized options (8GB block cache, bloom filter, etc.)"""
        logger.note(f"> Opening RocksDB:")
        logger.file(f"  * {self.db_path}")
        self.rocks = RocksOperator(
            configs={"db_path": self.db_path},
            connect_cls=self.__class__,
            verbose=False,
        )
        self.db = self.rocks.db
        logger.okay(f"  * [Opened]")

    def get_property_int(self, name: str) -> int:
        """Get integer property value from RocksDB."""
        try:
            return self.db.property_int_value(name)
        except Exception as e:
            if self.verbose:
                logger.warn(f"  × {name}: {e}")
            return None

    def get_property_str(self, name: str) -> str:
        """Get string property value from RocksDB."""
        try:
            return self.db.property_value(name)
        except Exception as e:
            if self.verbose:
                logger.warn(f"  × {name}: {e}")
            return None

    def dump_estimate_keys(self) -> int:
        """Dump estimated number of keys in DB."""
        logger.note(f"> Estimated Keys:")
        num_keys = self.get_property_int("rocksdb.estimate-num-keys")
        if num_keys is not None:
            logger.mesg(f"  * estimate-num-keys: {logstr.file(brk(f'{num_keys:,}'))}")
        return num_keys

    def dump_lsm_shape(self) -> dict:
        """Dump LSM tree shape: file counts at each level.

        High L0 file count indicates compaction debt and potential read slowdown.
        """
        logger.note(f"> LSM Shape (num-files-at-level):")
        level_files = {}
        total_files = 0

        for lvl in range(self.MAX_LEVELS):
            name = f"rocksdb.num-files-at-level{lvl}"
            value = self.get_property_str(name)
            if value is not None:
                file_count = int(value)
                level_files[lvl] = file_count
                total_files += file_count
                # highlight L0 if too many files (potential bottleneck)
                if lvl == 0 and file_count > 4:
                    logger.warn(
                        f"  * L{lvl}: {logstr.warn(brk(file_count))} ← high L0!"
                    )
                else:
                    logger.mesg(f"  * L{lvl}: {logstr.file(brk(file_count))}")

        logger.mesg(f"  * Total: {logstr.file(brk(total_files))}")
        return level_files

    def dump_level_stats(self) -> dict:
        """Dump per-level statistics including size and read/write amplification."""
        logger.note(f"> Level Stats:")
        stats = {}

        for lvl in range(self.MAX_LEVELS):
            # Get level size
            size_name = f"rocksdb.total-sst-files-size-at-level{lvl}"
            size_val = self.get_property_int(size_name)
            if size_val is not None and size_val > 0:
                size_mb = size_val / (1024 * 1024)
                stats[lvl] = {"size_bytes": size_val, "size_mb": size_mb}
                logger.mesg(f"  * L{lvl} size: {logstr.file(f'{size_mb:.2f} MB')}")

        return stats

    def dump_compaction_stats(self) -> dict:
        """Dump compaction-related statistics."""
        logger.note(f"> Compaction Stats:")
        stats = {}

        props = [
            ("rocksdb.compaction-pending", "compaction-pending"),
            ("rocksdb.num-running-compactions", "running-compactions"),
            ("rocksdb.num-running-flushes", "running-flushes"),
            ("rocksdb.estimate-pending-compaction-bytes", "pending-compaction-bytes"),
        ]

        for prop_name, display_name in props:
            value = self.get_property_int(prop_name)
            if value is not None:
                stats[display_name] = value
                if "bytes" in display_name:
                    size_mb = value / (1024 * 1024)
                    logger.mesg(
                        f"  * {display_name}: {logstr.file(f'{size_mb:.2f} MB')}"
                    )
                else:
                    logger.mesg(f"  * {display_name}: {logstr.file(brk(value))}")

        return stats

    def dump_cache_stats(self) -> dict:
        """Dump block cache statistics."""
        logger.note(f"> Block Cache Stats:")
        stats = {}

        props = [
            ("rocksdb.block-cache-capacity", "capacity"),
            ("rocksdb.block-cache-usage", "usage"),
            ("rocksdb.block-cache-pinned-usage", "pinned-usage"),
        ]

        for prop_name, display_name in props:
            value = self.get_property_int(prop_name)
            if value is not None:
                stats[display_name] = value
                size_mb = value / (1024 * 1024)
                logger.mesg(f"  * {display_name}: {logstr.file(f'{size_mb:.2f} MB')}")

        return stats

    def dump_io_stats(self) -> dict:
        """Dump I/O statistics (read/write bytes)."""
        logger.note(f"> I/O Stats:")
        stats = {}

        props = [
            ("rocksdb.estimate-live-data-size", "live-data-size"),
            ("rocksdb.total-sst-files-size", "total-sst-size"),
            ("rocksdb.size-all-mem-tables", "memtable-size"),
        ]

        for prop_name, display_name in props:
            value = self.get_property_int(prop_name)
            if value is not None:
                stats[display_name] = value
                if value > 1024 * 1024 * 1024:
                    size_str = f"{value / (1024**3):.2f} GB"
                else:
                    size_str = f"{value / (1024**2):.2f} MB"
                logger.mesg(f"  * {display_name}: {logstr.file(size_str)}")

        return stats

    def dump_stats_raw(self) -> str:
        """Dump raw rocksdb.stats output (multi-line text)."""
        logger.note(f"> Raw Stats (rocksdb.stats):")
        stats = self.get_property_str("rocksdb.stats")
        if stats:
            # print with indentation
            for line in stats.strip().split("\n"):
                logger.mesg(f"  {line}")
        return stats

    def dump_sstables(self) -> str:
        """Dump SST files summary."""
        logger.note(f"> SST Files Summary (rocksdb.sstables):")
        sstables = self.get_property_str("rocksdb.sstables")
        if sstables:
            for line in sstables.strip().split("\n"):
                logger.mesg(f"  {line}")
        return sstables

    def parse_stats_summary(self, stats_text: str) -> dict:
        """Parse stats text to extract key metrics.

        Looks for patterns like:
        - Cumulative writes: XXX writes, XXX keys, XXX MB
        - Cumulative compaction: XXX GB write, XXX GB read
        - Stalls: XXX
        """
        if not stats_text:
            return {}

        summary = {}

        # Parse cumulative writes
        writes_match = re.search(
            r"Cumulative writes:\s+([\d.]+)\s+\w+\s+writes,\s+([\d.]+)\s+\w+\s+keys",
            stats_text,
        )
        if writes_match:
            summary["cumulative_writes"] = float(writes_match.group(1))
            summary["cumulative_keys"] = float(writes_match.group(2))

        # Parse cumulative compaction
        compact_match = re.search(
            r"Cumulative compaction:\s+([\d.]+)\s+GB\s+write,\s+([\d.]+)\s+GB\s+read",
            stats_text,
        )
        if compact_match:
            summary["compaction_write_gb"] = float(compact_match.group(1))
            summary["compaction_read_gb"] = float(compact_match.group(2))

        # Parse stalls
        stall_match = re.search(r"Stalls\(count\):\s+(\d+)", stats_text)
        if stall_match:
            summary["stall_count"] = int(stall_match.group(1))

        return summary

    def analyze_read_amplification(self, level_files: dict) -> float:
        """Estimate read amplification based on LSM structure.

        Read amplification ≈ L0 files + number of levels with data
        High L0 count significantly increases read amplification.
        """
        if not level_files:
            return None

        l0_files = level_files.get(0, 0)
        non_empty_levels = sum(1 for lvl, count in level_files.items() if count > 0)

        # Simplified read amplification estimate
        # Real value depends on many factors (bloom filters, cache, etc.)
        read_amp = l0_files + non_empty_levels
        return read_amp

    def run_quick(self):
        """Run quick analysis: essential metrics only."""
        logger.note(f"\n{'='*60}")
        logger.note(f"> RocksDB Quick Analysis")
        logger.note(f"{'='*60}\n")

        self.dump_estimate_keys()
        level_files = self.dump_lsm_shape()
        self.dump_io_stats()
        self.dump_compaction_stats()
        self.dump_cache_stats()

        # Analyze and provide recommendations
        logger.note(f"\n> Analysis:")
        read_amp = self.analyze_read_amplification(level_files)
        if read_amp:
            logger.mesg(f"  * estimated read-amp: {logstr.file(brk(read_amp))}")

        l0_files = level_files.get(0, 0)
        if l0_files > 10:
            logger.warn(f"  ! L0 file count is high ({l0_files})")
            logger.warn(f"    → Consider running manual compaction")
            logger.warn(f"    → This may significantly slow random reads")
        elif l0_files > 4:
            logger.mesg(f"  * L0 file count is moderate ({l0_files})")
            logger.mesg(f"    → Compaction might help performance")

    def run_full(self):
        """Run full analysis: all metrics including raw stats."""
        self.run_quick()
        print()
        self.dump_level_stats()
        print()
        self.dump_stats_raw()

    def run(self, full: bool = False):
        """Run analysis."""
        if full:
            self.run_full()
        else:
            self.run_quick()

    def close(self):
        """Close RocksDB connection."""
        try:
            self.rocks.close()
            logger.note(f"> RocksDB closed")
        except Exception:
            pass


class RocksBenchmark:
    """Benchmark RocksDB sequential and random read performance.

    Block Cache behavior:
    - First read: data is fetched from disk and cached in block cache
    - Subsequent reads of same data: served from cache (much faster)
    - Cache size (8GB) << data size (88GB), so only ~9% can be cached
    - Random reads of different keys will have low cache hit rate
    - Repeated reads of same keys will show cache benefit
    """

    def __init__(
        self,
        max_count: int = 100_000,
        batch_size: int = 1000,
        warmup_rounds: int = 0,
        repeat_rounds: int = 1,
    ):
        self.max_count = max_count
        self.batch_size = batch_size
        self.warmup_rounds = warmup_rounds
        self.repeat_rounds = repeat_rounds
        self.init_processors()

    def init_processors(self):
        self.redis = RedisOperator(configs=REDIS_ENVS, connect_cls=self.__class__)
        self.rocks = RocksOperator(
            configs={"db_path": ROCKS_DB_PATH}, connect_cls=self.__class__
        )
        # Create ReadOptions with fill_cache enabled (default is True, but be explicit)
        self.read_options = ReadOptions()
        self.read_options.fill_cache(True)  # Ensure data is cached after read
        self.rocks.db.set_read_options(self.read_options)

    def get_cache_stats(self) -> dict:
        """Get current block cache statistics."""
        db = self.rocks.db
        stats = {
            "capacity_mb": int(db.property_value("rocksdb.block-cache-capacity") or 0)
            / (1024 * 1024),
            "usage_mb": int(db.property_value("rocksdb.block-cache-usage") or 0)
            / (1024 * 1024),
            "pinned_mb": int(db.property_value("rocksdb.block-cache-pinned-usage") or 0)
            / (1024 * 1024),
        }
        return stats

    def print_cache_stats(self, prefix: str = ""):
        """Print current cache statistics."""
        stats = self.get_cache_stats()
        usage_pct = (
            stats["usage_mb"] / stats["capacity_mb"] * 100
            if stats["capacity_mb"] > 0
            else 0
        )
        usage_str = f'{stats["usage_mb"]:.1f}'
        capacity_str = f'{stats["capacity_mb"]:.0f}'
        pct_str = f"{usage_pct:.1f}%"
        logger.mesg(
            f"  {prefix}cache: {logstr.file(usage_str)}/"
            f"{logstr.file(capacity_str)} MB "
            f"({logstr.file(pct_str)})"
        )

    def benchmark_sequential_read(self) -> dict:
        """Benchmark RocksDB sequential read using iter_items."""
        logger.note(f"> Benchmark: Sequential Read")
        logger.mesg(f"  * max_count : {logstr.file(brk(self.max_count))}")
        logger.mesg(f"  * batch_size: {logstr.file(brk(self.batch_size))}")

        read_count = 0
        total_bytes = 0
        start_time = time.perf_counter()

        bar = TCLogbar(total=self.max_count, desc="  * reading")
        for items_batch in self.rocks.iter_items(
            max_count=self.max_count, batch_size=self.batch_size
        ):
            for key, value in items_batch:
                read_count += 1
                if value is not None:
                    # Fast byte counting - avoid slow str() conversion
                    if isinstance(value, bytes):
                        total_bytes += len(value)
                    elif isinstance(value, (list, tuple)):
                        # Assume list of floats (embedding vector)
                        total_bytes += len(value) * 8
                    elif isinstance(value, str):
                        total_bytes += len(value)
            bar.update(len(items_batch))
            if read_count >= self.max_count:
                break
        print()

        elapsed = time.perf_counter() - start_time
        qps = read_count / elapsed if elapsed > 0 else 0
        throughput_mb = total_bytes / (1024 * 1024) / elapsed if elapsed > 0 else 0

        result = {
            "read_count": read_count,
            "total_bytes": total_bytes,
            "elapsed_sec": elapsed,
            "qps": qps,
            "throughput_mb_s": throughput_mb,
        }

        logger.okay(f"> Sequential Read Result:")
        logger.mesg(f"  * read_count    : {logstr.file(brk(read_count))}")
        logger.mesg(f"  * total_bytes   : {logstr.file(brk(int_bits(total_bytes)))}")
        logger.mesg(f"  * elapsed_sec   : {logstr.file(f'{elapsed:.3f} s')}")
        logger.mesg(f"  * qps           : {logstr.file(f'{qps:.2f} ops/s')}")
        logger.mesg(f"  * throughput    : {logstr.file(f'{throughput_mb:.2f} MB/s')}")

        return result

    def collect_random_keys(self) -> list[str]:
        """Collect random keys from Redis for random read benchmark."""
        logger.note(f"> Collecting random keys from Redis")
        all_keys = []
        for batch_keys in self.redis.scan_keys(
            pattern=REDIS_PT, max_count=self.max_count, batch_size=self.batch_size
        ):
            all_keys.extend(batch_keys)
            if len(all_keys) >= self.max_count:
                break
        all_keys = all_keys[: self.max_count]
        random.shuffle(all_keys)
        print()
        logger.mesg(f"  * collected: {logstr.file(brk(len(all_keys)))} keys")
        return all_keys

    def redis_keys_to_rocks_keys(self, redis_keys: list[str]) -> list[str]:
        """Convert Redis keys to RocksDB keys.

        Redis key format: 'bv.emb:BV1234567890'
        Rocks key format: 'BV1234567890'
        """
        return [key.removeprefix(REDIS_PREFIX) for key in redis_keys]

    def _read_keys_batch(self, rocks_keys: list[str], desc: str = "reading") -> dict:
        """Read a batch of keys and return statistics."""
        read_count = 0
        total_bytes = 0
        start_time = time.perf_counter()

        bar = TCLogbar(total=len(rocks_keys), desc=f"  * {desc}")
        for i in range(0, len(rocks_keys), self.batch_size):
            batch_keys = rocks_keys[i : i + self.batch_size]
            values = self.rocks.mget(batch_keys)
            for value in values:
                read_count += 1
                if value is not None:
                    # Fast byte counting - avoid slow str() conversion
                    if isinstance(value, bytes):
                        total_bytes += len(value)
                    elif isinstance(value, (list, tuple)):
                        # Assume list of floats (embedding vector)
                        # Each float is 8 bytes (Python float = C double)
                        total_bytes += len(value) * 8
                    elif isinstance(value, str):
                        total_bytes += len(value)
            bar.update(len(batch_keys))
        print()

        elapsed = time.perf_counter() - start_time
        qps = read_count / elapsed if elapsed > 0 else 0
        throughput_mb = total_bytes / (1024 * 1024) / elapsed if elapsed > 0 else 0

        return {
            "read_count": read_count,
            "total_bytes": total_bytes,
            "elapsed_sec": elapsed,
            "qps": qps,
            "throughput_mb_s": throughput_mb,
        }

    def benchmark_random_read(self) -> dict:
        """Benchmark RocksDB random read using mget with shuffled keys.

        With warmup_rounds > 0:
            - First, run warmup rounds to fill the cache (results discarded)
            - Then, run the actual benchmark (cache should be warm)

        With repeat_rounds > 1:
            - Read the same keys multiple times
            - Shows cache hit performance after first pass
        """
        logger.note(f"> Benchmark: Random Read")
        logger.mesg(f"  * max_count : {logstr.file(brk(self.max_count))}")
        logger.mesg(f"  * batch_size: {logstr.file(brk(self.batch_size))}")
        if self.warmup_rounds > 0:
            logger.mesg(
                f"  * warmup    : {logstr.file(brk(self.warmup_rounds))} rounds"
            )
        if self.repeat_rounds > 1:
            logger.mesg(
                f"  * repeat    : {logstr.file(brk(self.repeat_rounds))} rounds"
            )

        # collect and shuffle keys
        redis_keys = self.collect_random_keys()
        if not redis_keys:
            logger.warn(f"  × No keys found in Redis")
            return {}

        rocks_keys = self.redis_keys_to_rocks_keys(redis_keys)

        # Show initial cache state
        self.print_cache_stats(prefix="* initial ")

        # Warmup phase: read keys to fill cache (results discarded)
        if self.warmup_rounds > 0:
            logger.note(f"> Warmup Phase ({self.warmup_rounds} rounds)")
            for round_idx in range(self.warmup_rounds):
                _ = self._read_keys_batch(
                    rocks_keys, desc=f"warmup {round_idx+1}/{self.warmup_rounds}"
                )
            self.print_cache_stats(prefix="* after warmup ")

        # Actual benchmark
        logger.note(f"> Benchmark Phase")
        results = []
        for round_idx in range(self.repeat_rounds):
            if self.repeat_rounds > 1:
                desc = f"round {round_idx+1}/{self.repeat_rounds}"
            else:
                desc = "reading"
            result = self._read_keys_batch(rocks_keys, desc=desc)
            results.append(result)
            if self.repeat_rounds > 1:
                logger.mesg(f"    round {round_idx+1}: {result['qps']:.0f} ops/s")

        # Use last round result (or average if multiple rounds)
        if len(results) == 1:
            final_result = results[0]
        else:
            # Average results from all rounds
            final_result = {
                "read_count": sum(r["read_count"] for r in results),
                "total_bytes": sum(r["total_bytes"] for r in results),
                "elapsed_sec": sum(r["elapsed_sec"] for r in results),
                "qps": sum(r["qps"] for r in results) / len(results),
                "throughput_mb_s": sum(r["throughput_mb_s"] for r in results)
                / len(results),
            }
            # Also show improvement from first to last round
            first_qps = results[0]["qps"]
            last_qps = results[-1]["qps"]
            improvement = (last_qps / first_qps - 1) * 100 if first_qps > 0 else 0
            speedup_str = f"{improvement:+.1f}%"
            logger.mesg(
                f"  * cache speedup: {logstr.file(speedup_str)} (round 1 → {self.repeat_rounds})"
            )

        # Show final cache state
        self.print_cache_stats(prefix="* final ")

        elapsed_str = f"{final_result['elapsed_sec']:.3f} s"
        qps_str = f"{final_result['qps']:.2f} ops/s"
        throughput_str = f"{final_result['throughput_mb_s']:.2f} MB/s"

        logger.okay(f"> Random Read Result:")
        logger.mesg(
            f"  * read_count    : {logstr.file(brk(final_result['read_count']))}"
        )
        logger.mesg(
            f"  * total_bytes   : {logstr.file(brk(int_bits(final_result['total_bytes'])))}"
        )
        logger.mesg(f"  * elapsed_sec   : {logstr.file(elapsed_str)}")
        logger.mesg(f"  * qps           : {logstr.file(qps_str)}")
        logger.mesg(f"  * throughput    : {logstr.file(throughput_str)}")

        return final_result

    def run(self):
        """Run all benchmarks and print summary."""
        logger.note(f"> RocksDB Benchmark")
        logger.mesg(f"  * db_path: {logstr.file(str(ROCKS_DB_PATH))}")

        seq_result = self.benchmark_sequential_read()
        rand_result = self.benchmark_random_read()

        # print summary
        logger.note(f"\n> Benchmark Summary:")
        logger.mesg(f"  {'':15s} {'Sequential':>15s} {'Random':>15s}")
        logger.mesg(f"  {'-'*47}")

        if seq_result and rand_result:
            logger.mesg(
                f"  {'read_count':15s} "
                f"{seq_result['read_count']:>15,d} "
                f"{rand_result['read_count']:>15,d}"
            )
            logger.mesg(
                f"  {'elapsed_sec':15s} "
                f"{seq_result['elapsed_sec']:>15.3f} "
                f"{rand_result['elapsed_sec']:>15.3f}"
            )
            logger.mesg(
                f"  {'qps (ops/s)':15s} "
                f"{seq_result['qps']:>15,.2f} "
                f"{rand_result['qps']:>15,.2f}"
            )
            logger.mesg(
                f"  {'throughput MB/s':15s} "
                f"{seq_result['throughput_mb_s']:>15.2f} "
                f"{rand_result['throughput_mb_s']:>15.2f}"
            )


class BenchmarkArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # benchmark options
        self.add_argument("-n", "--max-count", type=int, default=100_000)
        self.add_argument("-b", "--batch-size", type=int, default=1000)
        self.add_argument("-s", "--sequential", action="store_true")
        self.add_argument("-r", "--random", action="store_true")
        # cache test options
        self.add_argument(
            "-w",
            "--warmup",
            type=int,
            default=0,
            help="Number of warmup rounds before benchmark (fill cache)",
        )
        self.add_argument(
            "-R",
            "--repeat",
            type=int,
            default=1,
            help="Number of repeat rounds (test cache hit performance)",
        )
        # analyze options
        self.add_argument("-a", "--analyze", action="store_true", help="Run analysis")
        self.add_argument("-f", "--full", action="store_true", help="Full analysis")
        self.add_argument("-d", "--db-path", type=str, default=None)
        self.add_argument("-q", "--quiet", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    args = BenchmarkArgParser().args

    # run analysis mode
    if args.analyze:
        db_path = Path(args.db_path) if args.db_path else None
        analyzer = RocksStatsAnalyzer(db_path=db_path, verbose=not args.quiet)
        try:
            analyzer.run(full=args.full)
        finally:
            analyzer.close()
        return

    # run benchmark mode
    benchmark = RocksBenchmark(
        max_count=args.max_count,
        batch_size=args.batch_size,
        warmup_rounds=args.warmup,
        repeat_rounds=args.repeat,
    )

    # run specific benchmark or all
    if args.sequential and not args.random:
        benchmark.benchmark_sequential_read()
    elif args.random and not args.sequential:
        benchmark.benchmark_random_read()
    else:
        benchmark.run()


if __name__ == "__main__":
    main()

    # Run all benchmarks with 100k embeddings (default)
    # python -m models.tembed.benchmark

    # Run with custom count
    # python -m models.tembed.benchmark -n 50000

    # Run with custom batch size
    # python -m models.tembed.benchmark -n 100000 -b 500

    # Run only sequential read benchmark
    # python -m models.tembed.benchmark -s

    # Run only random read benchmark
    # python -m models.tembed.benchmark -r

    # Run with warmup (fill cache first, then benchmark)
    # python -m models.tembed.benchmark -r -w 1 -n 10000

    # Run with repeat (test cache hit performance)
    # python -m models.tembed.benchmark -r -R 3 -n 10000

    # Run with warmup and repeat (best for cache testing)
    # python -m models.tembed.benchmark -r -w 1 -R 3 -n 10000

    # Run quick analysis
    # python -m models.tembed.benchmark -a

    # Run full analysis with raw stats
    # python -m models.tembed.benchmark -a -f

    # Run analysis with custom db path
    # python -m models.tembed.benchmark -a -d /path/to/rocksdb
