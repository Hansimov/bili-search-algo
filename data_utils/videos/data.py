import argparse
import sys

from sedb import MongoOperator
from tclogger import logger, logstr, TCLogbar, TCLogbarGroup, dict_to_str
from typing import Literal, Generator, Union

from configs.envs import MONGO_ENVS
from data_utils.videos.convert import DocSentenceConverter
from data_utils.videos.parquet import VideoTextsParquetReader


class VideosTagsAggregator:
    def __init__(
        self,
        videos_collect_name: str = "videos",
        tags_collect_name: str = "videos_tags",
        tags_join_name: str = "tagger",
        batch_size: int = 10000,
    ):
        self.videos_collect_name = videos_collect_name
        self.tags_collect_name = tags_collect_name
        self.tags_join_name = tags_join_name
        self.batch_size = batch_size
        self.init_pipeline()
        self.init_mongo()
        self.init_cursor()

    def init_pipeline(self):
        self.pipeline = [
            {
                "$lookup": {
                    "from": self.tags_collect_name,
                    "localField": "bvid",
                    "foreignField": "bvid",
                    "as": self.tags_join_name,
                }
            },
            {"$unwind": f"${self.tags_join_name}"},
            {
                "$project": {
                    "title": 1,
                    "desc": 1,
                    "owner.name": 1,
                    "tid": 1,
                    f"{self.tags_join_name}.tags": 1,
                    f"{self.tags_join_name}.region_tags": 1,
                }
            },
        ]

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS, connect_msg=f"from {self.__class__.__name__}", indent=2
        )
        self.videos_collect = self.mongo.db[self.videos_collect_name]
        self.videos_estimated_count = self.videos_collect.estimated_document_count()

    def init_cursor(self):
        self.cursor = self.videos_collect.aggregate(
            self.pipeline, allowDiskUse=True, batchSize=self.batch_size
        )

    def __iter__(self):
        for doc in self.cursor:
            yield doc


class SentencesDataloader:
    def __init__(
        self,
        dbname: str = None,
        collect_name: Literal["videos", "videos_tags", "users", "pages"] = "videos",
        data_fields: list[str] = None,
        mongo_filter: dict = {},
        batch_size: int = 10000,
        max_batch: int = None,
        estimate_count: bool = False,
        iter_val: Literal["doc", "sentence", "tokens"] = "sentence",
        task_type: Literal["fasttext", "freq", "sentencepiece"] = "fasttext",
        tokenizer=None,
        max_sentence_length: int = None,
        iter_epochs: int = None,
        show_at_init: bool = False,
        show_epoch_bar: bool = True,
        verbose: bool = False,
    ):
        self.dbname = dbname
        self.collect_name = collect_name
        self.data_fields = data_fields
        self.mongo_filter = mongo_filter
        self.batch_size = batch_size
        self.max_batch = max_batch
        self.estimate_count = estimate_count
        self.iter_val = iter_val
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length
        self.iter_epochs = iter_epochs
        self.show_at_init = show_at_init
        self.show_epoch_bar = show_epoch_bar
        self.verbose = verbose
        self.init_mongo()
        self.init_cursor()
        self.init_progress_bars()
        self.init_doc_converter()
        self.init_total()

    def init_mongo(self):
        self.mongo_envs = MONGO_ENVS
        if self.dbname:
            self.mongo_envs["dbname"] = self.dbname
        self.mongo = MongoOperator(
            self.mongo_envs,
            connect_msg=f"from {self.__class__.__name__}",
            indent=0,
            verbose=self.verbose,
        )
        self.samples_collect = self.mongo.db[self.collect_name]
        collect_str = f"* collection: {logstr.success(self.collect_name)}"
        logger.file(collect_str, indent=self.mongo.indent + 2, verbose=self.verbose)
        if self.collect_name == "pages":
            self.mongo_filter = {"ns": 0, "revision.text": {"$exists": True}}

    def init_cursor(self):
        self.cursor = self.samples_collect.find(self.mongo_filter)

    def init_progress_bars(self):
        if self.iter_epochs is None or self.iter_epochs <= 1:
            self.show_epoch_bar = False
        if self.show_epoch_bar:
            subbar_indent = 2
        else:
            subbar_indent = 0
        sample_bar_head_str = " " * subbar_indent + "* Sample"
        self.sample_bar = TCLogbar(head=logstr.note(sample_bar_head_str))
        if self.show_epoch_bar:
            self.epoch_bar = TCLogbar(head=logstr.note("> Epoch"))
            self.logbars = [self.epoch_bar, self.sample_bar]
        else:
            self.epoch_bar = None
            self.logbars = [self.sample_bar]
        if len(self.logbars) > 1:
            TCLogbarGroup(
                self.logbars, show_at_init=self.show_at_init, verbose=self.verbose
            )

    def init_doc_converter(self):
        if self.task_type in ["fasttext", "freq"]:
            converter_params = {
                "is_mask_atoz_ws": False,
                "is_replace_non_word": True,
                "is_replace_digits": False,
                "is_simplify_chinese": True,
                "is_multiply_sentence": False,
                "max_repeats": 4,
            }
        elif self.task_type == "sentencepiece":
            converter_params = {
                "is_mask_atoz_ws": True,
                "is_replace_non_word": True,
                "is_replace_digits": True,
                "is_simplify_chinese": True,
                "is_multiply_sentence": True,
                "max_repeats": 4,
            }
        else:
            converter_params = {}
        self.doc_converter = DocSentenceConverter(
            collect_name=self.collect_name,
            fields=self.data_fields,
            **converter_params,
        )

    def init_total(self):
        if self.estimate_count:
            logger.note(f"> Estimating docs:", end=" ")
            self.samples_count = self.samples_collect.estimated_document_count()
            logger.mesg(f"[{self.samples_count}]")
        else:
            logger.note(f"> Counting docs:", end=" ")
            self.samples_count = self.samples_collect.count_documents(self.mongo_filter)
            logger.mesg(f"[{self.samples_count}]")
            if self.mongo_filter:
                logger.file(dict_to_str(self.mongo_filter), indent=2)
        if self.epoch_bar:
            self.epoch_bar.total = self.iter_epochs or 1
        self.sample_bar.total = self.samples_count

    def __epoch_start__(self):
        # self.init_total()
        if self.epoch_bar:
            self.epoch_bar.update(0, flush=True)

    def __epoch_end__(self):
        if self.epoch_bar:
            self.epoch_bar.update(increment=1)
        if (
            self.iter_epochs
            and self.iter_epochs > 1
            and self.epoch_bar
            and self.epoch_bar.count < self.iter_epochs
        ):
            self.sample_bar.reset()
            self.init_cursor()
        else:
            print()

    def segment_sentence(self, sentence: str) -> Generator[str, None, None]:
        """generate concat sentence segs without exceeding max_sentence_length, and must break at whitespaces"""
        segs = sentence.split()
        tmp_segs = []
        sentence_len = -1
        for seg in segs:
            seg_len = len(seg)
            if tmp_segs and sentence_len + seg_len + 1 >= self.max_sentence_length:
                yield " ".join(tmp_segs)
                tmp_segs = []
                sentence_len = -1
            tmp_segs.append(seg)
            sentence_len += seg_len + 1
        if tmp_segs:
            yield " ".join(tmp_segs)

    def format_docs(
        self, docs: list[dict], doc_type: Literal["doc", "sentence"]
    ) -> Union[list[dict], list[str]]:
        if doc_type == "sentence":
            if self.max_sentence_length:
                res = []
                for doc in docs:
                    sentence = self.doc_converter.convert(doc)
                    if len(sentence) > self.max_sentence_length:
                        for seg in self.segment_sentence(sentence):
                            res.append(seg)
                    else:
                        res.append(sentence)
            else:
                res = [self.doc_converter.convert(doc) for doc in docs]
            return res
        else:
            return docs

    def doc_batch_generator(
        self, doc_type: Literal["doc", "sentence"] = "doc"
    ) -> Generator[Union[list[dict], list[str]], None, None]:
        batch = []
        for idx, doc in enumerate(self.cursor):
            batch.append(doc)
            if (idx + 1) % self.batch_size == 0:
                yield self.format_docs(batch, doc_type=doc_type)
                batch = []
        if batch:
            yield self.format_docs(batch, doc_type=doc_type)
            batch = []

    def __iter__(self) -> Generator[str, None, None]:
        self.__epoch_start__()
        for batch_idx, batch in enumerate(self.doc_batch_generator()):
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            for doc in batch:
                self.sample_bar.update(increment=1)
                if self.iter_val == "sentence" or self.iter_val == "tokens":
                    try:
                        sentence = self.doc_converter.convert(doc)
                    except:
                        logger.warn(f"× Cannot convert doc: {doc}")
                        continue
                    if (
                        self.max_sentence_length
                        and len(sentence) > self.max_sentence_length
                    ):
                        for seg in self.segment_sentence(sentence):
                            yield (
                                seg
                                if self.iter_val == "sentence"
                                else self.tokenizer.tokenize(seg)
                            )
                    else:
                        yield (
                            sentence
                            if self.iter_val == "sentence"
                            else self.tokenizer.tokenize(sentence)
                        )
                else:
                    yield doc
        self.__epoch_end__()


class ParquetRowsDataLoader:
    def __init__(
        self,
        parquet_reader: VideoTextsParquetReader = None,
        column: str = None,
        iter_epochs: int = None,
        batch_size: int = 10000,
        max_batch: int = None,
        max_rows: int = None,
        show_at_init: bool = False,
        show_epoch_bar: bool = True,
        verbose: bool = False,
    ):
        self.parquet_reader = parquet_reader
        self.column = column
        self.iter_epochs = iter_epochs
        self.batch_size = batch_size
        self.max_batch = max_batch
        self.max_rows = max_rows
        self.show_at_init = show_at_init
        self.show_epoch_bar = show_epoch_bar
        self.verbose = verbose
        self.init_progress_bars()
        self.init_generators()

    def init_progress_bars(self):
        if self.show_epoch_bar:
            subbar_indent = 2
        else:
            subbar_indent = 0
        batch_bar_head_str = " " * subbar_indent + "* Batch "
        sample_bar_head_str = " " * subbar_indent + "* Sample"
        self.batch_bar = TCLogbar(head=logstr.note(batch_bar_head_str))
        self.sample_bar = TCLogbar(head=logstr.note(sample_bar_head_str))
        if self.show_epoch_bar:
            self.epoch_bar = TCLogbar(head=logstr.note("> Epoch"))
            self.logbars = [self.epoch_bar, self.batch_bar, self.sample_bar]
        else:
            self.epoch_bar = None
            self.logbars = [self.batch_bar, self.sample_bar]
        TCLogbarGroup(
            self.logbars, show_at_init=self.show_at_init, verbose=self.verbose
        )

    def init_total(self):
        self.file_idx = 0
        if self.epoch_bar:
            self.epoch_bar.total = self.iter_epochs or 1
        if self.max_batch:
            self.batch_bar.total = self.max_batch
        else:
            self.batch_bar.total = (
                self.parquet_reader.total_row_count // self.batch_size + 1
            )

    def init_generators(self):
        self.table_generator = self.parquet_reader.table_generator()
        self.row_generator = self.parquet_reader.row_generator()
        self.batch_generator = self.parquet_reader.batch_generator(
            column=self.column, batch_size=self.batch_size
        )

    def __epoch_start__(self):
        self.init_total()
        if self.epoch_bar:
            self.epoch_bar.update(0, flush=True)

    def __epoch_end__(self):
        if self.epoch_bar and self.epoch_bar.total >= 1:
            self.epoch_bar.update(increment=1)
        if (
            self.iter_epochs
            and self.iter_epochs > 1
            and self.epoch_bar
            and self.epoch_bar.count < self.iter_epochs
        ):
            self.batch_bar.reset()
            self.sample_bar.reset()
            self.init_generators()
        else:
            print("\n\r\n")


class CommonDataLoaderArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-bs", "--batch-size", type=int, default=10000)
        self.add_argument("-mb", "--max-batch", type=int, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


class SentencesDataLoaderArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-db", "--dbname", type=str, default=None)
        self.add_argument(
            "-cn",
            "--collect-name",
            type=str,
            choices=["videos", "users", "pages"],
            default="videos",
        )
        self.add_argument("-td", "--tid", type=self.to_int, default=None)
        self.add_argument("-pd", "--ptid", type=self.to_int, default=None)
        self.add_argument("-rf", "--reverse-filter", action="store_true")
        self.add_argument("-fg", "--filter-group", type=str, default=None)
        self.add_argument("-df", "--data-fields", type=str, default=None)
        self.add_argument("-ec", "--estimate-count", action="store_true")

    def to_int(self, values: Union[str, int, list[int]], sep: str = ",") -> list[int]:
        if isinstance(values, str):
            res = list(map(int, values.split(sep)))
        elif isinstance(values, int):
            res = [values]
        else:
            res = list(map(int, values))

        if len(res) == 1:
            return res[0]
        else:
            return res

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


class ParquetRowsDataLoaderArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-mr", "--max-rows", type=int, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    args = SentencesDataLoaderArgParser().parse_args()
    data_loader_params = {
        "dbname": args.dbname,
        "collect_name": args.collect_name,
        "data_fields": args.data_fields.split(",") if args.data_fields else None,
        "batch_size": args.batch_size,
        "max_batch": args.max_batch,
        "estimate_count": args.estimate_count,
    }
    loader = SentencesDataloader(**data_loader_params, show_at_init=False, verbose=True)
    for doc in loader:
        continue

    # python -m data_utils.videos.data -db zhwiki -cn pages -bs 1000
    # python -m data_utils.videos.data -cn videos -bs 10000 -mb 200 -ec
