import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

from pathlib import Path
from tclogger import logger, logstr, brk, int_bits
from typing import Union

PYARROW_TYPES = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    list[int]: pa.list_(pa.int64()),
    list[float]: pa.list_(pa.float64()),
    list[str]: pa.list_(pa.string()),
}


class VideoTextsParquetOperator:
    DATA_ROOT = Path(__file__).parents[2] / "data"
    COL_TYPES = {
        "bvid": str,
        "ptid": int,
        "tid": int,
        "sentence": str,
        "tokens": list[str],
    }

    def __init__(
        self,
        dataset_root: Union[Path, str] = None,
        dataset_name: str = "video_texts",
        parquet_prefix: str = "",
        col_types: dict[str, type] = None,
        dataset_max_rows: int = int(1e4 * 1e6),
        file_max_rows: int = int(1e4 * 100),
        buffer_max_rows: int = int(1e4 * 10),
        verbose: bool = False,
    ):
        self.data_root = Path(dataset_root or self.DATA_ROOT)
        self.dataset_name = dataset_name
        self.parquet_prefix = parquet_prefix
        self.dataset_max_rows = dataset_max_rows
        self.file_max_rows = file_max_rows
        self.buffer_max_rows = buffer_max_rows
        self.col_types = col_types or self.COL_TYPES
        self.verbose = verbose
        self.buffer: pa.Table = None
        self.init_paths()
        self.init_schema()

    def init_paths(self):
        self.dataset_dir = self.data_root / self.dataset_name
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.pq_idx_bits = int_bits(self.dataset_max_rows // self.file_max_rows)

    def init_schema(self) -> dict:
        self.schema = pa.schema(
            [
                pa.field(col_name, PYARROW_TYPES[col_type])
                for col_name, col_type in self.col_types.items()
            ]
        )

    def clear_dataset(self):
        dataset_name_str = logstr.note(brk(self.dataset_name))
        logger.warn(f"  ! WARNING: You are deleting dataset: {dataset_name_str}")
        confirmation = input(
            logstr.mesg(
                f'  > Type "{logstr.note(self.dataset_name)}" to confirm deletion: '
            )
        )
        if confirmation != self.dataset_name:
            logger.mesg(f"  * Skip clear dataset: {dataset_name_str}")
        else:
            logger.warn(f"  ! Deleting dataset: {dataset_name_str}")
            parquet_files = sorted(
                self.dataset_dir.glob(f"{self.parquet_prefix}*.parquet")
            )
            for parquet_file in parquet_files:
                parquet_file.unlink()

    def next_parquet_idx(self) -> int:
        parquet_files = sorted(self.dataset_dir.glob(f"{self.parquet_prefix}*.parquet"))
        if not parquet_files:
            pq_idx = 0
        else:
            last_file = parquet_files[-1]
            metadata = pq.read_metadata(last_file)
            if metadata.num_rows < self.file_max_rows:
                pq_idx = len(parquet_files) - 1
            else:
                pq_idx = len(parquet_files)
        return pq_idx

    def get_parquet_path(self, pq_idx: int) -> Path:
        return (
            self.dataset_dir
            / f"{self.parquet_prefix}{pq_idx:0>{self.pq_idx_bits}}.parquet"
        )

    def next_parquet_path(self) -> Path:
        return self.get_parquet_path(self.next_parquet_idx())

    def rows_to_table(self, rows: Union[list[dict], pd.DataFrame]) -> pa.Table:
        if isinstance(rows, list):
            return pa.Table.from_pylist(rows, schema=self.schema)
        elif isinstance(rows, pd.DataFrame):
            return pa.Table.from_pandas(rows, schema=self.schema)
        else:
            error_str = "× rows must be list[dict] or pd.DataFrame"
            logger.warn(error_str)
            raise ValueError(error_str)

    def append_buffer(self, rows: Union[list[dict], pd.DataFrame]):
        table = self.rows_to_table(rows)
        if not self.buffer:
            self.buffer = table
        else:
            self.buffer = pa.concat_tables([self.buffer, table])
        if self.buffer.num_rows >= self.buffer_max_rows:
            self.flush_buffer()

    def write_parquet(self, table: pa.Table = None):
        self.parquet_path = self.next_parquet_path()
        if self.verbose:
            print()
            row_count_str = logstr.mesg(brk(table.num_rows))
            logger.note(f"  > Write to parquet: {row_count_str}")
            logger.file(f"    * {self.parquet_path}")
        if not self.parquet_path.exists():
            pq.write_table(table, self.parquet_path)
        else:
            old_table = pq.read_table(self.parquet_path)
            new_table = pa.concat_tables([old_table, table])
            pq.write_table(new_table, self.parquet_path)

    def flush_buffer(self):
        if self.buffer:
            self.write_parquet(self.buffer)
            self.buffer = None


class ParquetOperatorArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-dr", "--dataset-root", type=str, default=None)
        self.add_argument("-dn", "--dataset-name", type=str, default="video_texts")
        self.add_argument("-pp", "--parquet-prefix", type=str, default="")
        self.add_argument("-dw", "--dataset-max-w-rows", type=int, default=1e6)
        self.add_argument("-fw", "--file-max-w-rows", type=int, default=100)
        self.add_argument("-bw", "--buffer-max-w-rows", type=int, default=10)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args
