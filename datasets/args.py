import argparse
import sys

from datasets.videos.data import CommonDataLoaderArgParser
from datasets.videos.data import SentencesDataLoaderArgParser
from datasets.videos.data import ParquetRowsDataLoaderArgParser
from datasets.videos.parquet import ParquetOperatorArgParser, ParquetWriterArgParser


class MergedArgParser:
    def __init__(self, *parser_classes):
        self.parser_classes = parser_classes
        self.construct_parsers()

    def construct_parsers(self):
        self.parsers = [cls(add_help=False) for cls in self.parser_classes]

    def add_parser_class(self, *args):
        self.parser_classes += args
        self.construct_parsers()

    def parse_args(self, args=None):
        self.args = argparse.ArgumentParser(parents=self.parsers).parse_args(
            sys.argv[1:]
        )
        return self.args


DATA_LOADER_ARG_PARSER = MergedArgParser(
    CommonDataLoaderArgParser,
    SentencesDataLoaderArgParser,
    ParquetRowsDataLoaderArgParser,
    ParquetOperatorArgParser,
    ParquetWriterArgParser,
)
