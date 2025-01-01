import argparse
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

from collections.abc import Generator
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

DATA_ROOT = Path(__file__).parents[2] / "data"
COL_TYPES = {
    "bvid": str,
    "ptid": int,
    "tid": int,
    "sentence": str,
    "tokens": list[str],
}


class VideoTextsParquetWriter:
    def __init__(
        self,
        dataset_root: Union[Path, str] = None,
        dataset_name: str = "video_texts",
        parquet_prefix: str = "",
        col_types: dict[str, type] = None,
        dataset_max_rows: int = int(1e6 * 1e4),
        file_max_rows: int = int(100 * 1e4),
        buffer_max_rows: int = int(10 * 1e4),
        force_delete: bool = False,
        verbose: bool = False,
    ):
        self.data_root = Path(dataset_root or DATA_ROOT)
        self.dataset_name = dataset_name
        self.parquet_prefix = parquet_prefix
        self.dataset_max_rows = dataset_max_rows
        self.file_max_rows = file_max_rows
        self.buffer_max_rows = buffer_max_rows
        self.force_delete = force_delete
        self.col_types = col_types or COL_TYPES
        self.verbose = verbose
        self.buffer: pa.Table = None
        self.init_paths()
        self.init_schema()

    def init_paths(self):
        self.dataset_dir = self.data_root / self.dataset_name
        self.pq_idx_bits = int_bits(self.dataset_max_rows // self.file_max_rows)

    def init_schema(self) -> dict:
        self.schema = pa.schema(
            [
                pa.field(col_name, PYARROW_TYPES[col_type])
                for col_name, col_type in self.col_types.items()
            ]
        )

    def clear_dataset(self):
        def delete_files(files: list[Path]):
            logger.warn(f"  ! Deleting dataset: {len(files)} files")
            for file in files:
                file.unlink()

        dataset_name_str = logstr.file(self.dataset_name)
        if not self.dataset_dir.exists():
            logger.mesg(f"  * No existed dataset: [{dataset_name_str}]")
            return

        parquet_files = sorted(
            list(self.dataset_dir.glob(f"{self.parquet_prefix}*.parquet"))
        )
        if not parquet_files:
            logger.mesg(f"  * No existed parquets in dataset: [{dataset_name_str}]")
            return

        logger.warn(f"  ! WARNING: You are deleting dataset: [{dataset_name_str}]")

        if self.force_delete:
            delete_files(parquet_files)
            return
        else:
            confirmation = None
            while confirmation != self.dataset_name:
                confirmation = input(
                    logstr.mesg(f'  > Type "{dataset_name_str}" to confirm deletion: ')
                )
            delete_files(parquet_files)

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
        if not self.parquet_path.parent.exists():
            self.parquet_path.parent.mkdir(parents=True)
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


class VideoTextsParquetReader:
    def __init__(
        self,
        dataset_root: Union[Path, str] = None,
        dataset_name: str = "video_texts",
        parquet_prefix: str = "",
        verbose: bool = False,
    ):
        self.data_root = Path(dataset_root or DATA_ROOT)
        self.dataset_name = dataset_name
        self.parquet_prefix = parquet_prefix
        self.verbose = verbose
        self.init_paths()
        self.init_total()

    def init_paths(self):
        self.dataset_dir = self.data_root / self.dataset_name
        if self.parquet_prefix:
            pattern = f"{self.parquet_prefix}*.parquet"
        else:
            pattern = "*.parquet"
        self.parquet_files = sorted(self.dataset_dir.glob(pattern))

    def init_total(self):
        self.total_file_count = len(self.parquet_files)
        self.file_row_counts = {
            parquet_file: pq.read_metadata(parquet_file).num_rows
            for parquet_file in self.parquet_files
        }
        self.total_row_count = sum(self.file_row_counts.values())

    def table_generator(self) -> Generator[pa.Table, None, None]:
        for parquet_file in self.parquet_files:
            table = pq.read_table(parquet_file)
            self.current_table_row_count = table.num_rows
            if self.verbose:
                row_count_str = logstr.mesg(brk(table.num_rows))
                logger.note(f"  > Read parquet: {row_count_str}")
                logger.file(f"    * {parquet_file}")
            yield table
            del table
            gc.collect()

    def batch_generator(
        self, column: str = None, batch_size: int = 10000
    ) -> Generator[Union[list[dict], list, pa.Table], None, None]:
        for table in self.table_generator():
            for i in range(0, table.num_rows, batch_size):
                if not column:
                    yield table.slice(i, batch_size).to_pylist()
                else:
                    yield table.column(column).slice(i, batch_size)
            del table
            gc.collect()

    def row_generator(self) -> Generator[dict, None, None]:
        for table in self.table_generator():
            for row in table.to_pydict().values():
                yield row
            del table
            gc.collect()


class ParquetOperatorArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-dr", "--dataset-root", type=str, default=None)
        self.add_argument("-dn", "--dataset-name", type=str, default="video_texts")
        self.add_argument("-pp", "--parquet-prefix", type=str, default="")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


class ParquetWriterArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-dw", "--dataset-max-w-rows", type=int, default=1e6)
        self.add_argument("-fw", "--file-max-w-rows", type=int, default=200)
        self.add_argument("-bw", "--buffer-max-w-rows", type=int, default=100)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args
