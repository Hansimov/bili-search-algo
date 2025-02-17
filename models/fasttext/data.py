import json

from gensim.models.doc2vec import TaggedDocument
from tclogger import logger, dict_to_str

from data_utils.videos.data import SentencesDataloader, ParquetRowsDataLoader


class FasttextDataLoader(SentencesDataloader):
    def __iter__(self):
        self.__epoch_start__()
        for batch_idx, batch in enumerate(
            self.doc_batch_generator(doc_type="sentence")
        ):
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            tokenize_results: list[dict[str, str]] = self.tokenizer.tokenize_list(
                batch, sort=False
            )
            for result in tokenize_results:
                yield result["tokens"]
        self.__epoch_end__()


class FasttextParquetDataLoader(ParquetRowsDataLoader):
    def check_use_checkpoint(self):
        self.use_checkpoint = (
            hasattr(self, "checkpoint_num")
            and hasattr(self, "checkpoint_func")
            and self.checkpoint_num
            and callable(self.checkpoint_func)
        )

    def process_checkpoint(self, batch_idx: int):
        if self.use_checkpoint and (batch_idx + 1) % self.checkpoint_num == 0:
            self.checkpoint_func()

    def __iter__(self):
        self.__epoch_start__()
        self.batch_bar.reset()
        self.batch_bar.update(flush=True)
        sample_idx = 0
        self.check_use_checkpoint()
        for batch_idx, tokens_batch in enumerate(self.batch_generator):
            self.batch_bar.update(increment=1, flush=True)
            self.process_checkpoint(batch_idx)
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            self.sample_bar.total = len(tokens_batch)
            self.sample_bar.update(flush=True)
            for tokens in tokens_batch:
                self.sample_bar.update(1)
                sample_idx += 1
                if getattr(self, "model_class", None) == "doc2vec":
                    yield TaggedDocument(tokens, [sample_idx])
                else:
                    yield tokens
            self.sample_bar.reset()
        self.__epoch_end__()

    def get_count_from_local(self):
        if self.parquet_reader.dataset_name:
            count_name = self.parquet_reader.dataset_name
        else:
            count_name = "_"
        self.count_json = (
            self.parquet_reader.data_root.parent
            / "parquets_count"
            / f"{count_name}.count.json"
        )
        if self.count_json.exists():
            with self.count_json.open("r") as f:
                self.json_data = json.load(f)
            count_dict = self.json_data.get(str(self.max_batch), {})
            return (
                count_dict.get("total_words", None),
                count_dict.get("corpus_count", None),
            )
        else:
            self.json_data = {}
            return None, None

    def get_corpus_count(self):
        _, corpus_count = self.get_count_from_local()
        if not corpus_count:
            # read number of rows from parquet file
            corpus_count = self.parquet_reader.total_row_count
        return corpus_count

    def save_count_info(self, count_info: dict):
        if not self.count_json.parent.exists():
            self.count_json.parent.mkdir(parents=True, exist_ok=True)
        with self.count_json.open("w") as f:
            self.json_data.update(count_info)
            json.dump(self.json_data, f, indent=4)

    def get_count(self) -> tuple[int, int]:
        logger.note("> Counting vocab:")
        is_newly_counted = False
        total_words, corpus_count = self.get_count_from_local()
        if not total_words or not corpus_count:
            total_words, corpus_count = 0, 0
            for tokens in self:
                total_words += len(tokens)
                corpus_count += 1
            is_newly_counted = True
        count_info = {
            str(self.max_batch): {
                "total_words": total_words,
                "corpus_count": corpus_count,
            }
        }
        # logger.mesg(dict_to_str(count_info), indent=2)
        if is_newly_counted:
            self.save_count_info(count_info)
        return total_words, corpus_count
