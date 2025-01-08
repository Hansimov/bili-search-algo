import json

from gensim.models.doc2vec import TaggedDocument
from tclogger import logger, dict_to_str

from datasets.videos.data import SentencesDataloader, ParquetRowsDataLoader


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
    def __iter__(self):
        self.__epoch_start__()
        self.batch_bar.reset()
        self.batch_bar.update(flush=True)
        sample_idx = 0
        for batch_idx, tokens_batch in enumerate(self.batch_generator):
            self.batch_bar.update(increment=1, flush=True)
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
        self.count_json = self.parquet_reader.data_root / (
            self.parquet_reader.dataset_name + ".count.json"
        )
        if self.count_json.exists():
            with self.count_json.open("r") as f:
                count_dict = json.load(f)
            return (
                count_dict.get("total_words", None),
                count_dict.get("corpus_count", None),
            )
        else:
            return None, None

    def get_corpus_count(self):
        _, corpus_count = self.get_count_from_local()
        if not corpus_count:
            # read number of rows from parquet file
            corpus_count = self.parquet_reader.total_row_count
        return corpus_count

    def get_count(self) -> tuple[int, int]:
        logger.note("> Counting vocab:")
        total_words, corpus_count = self.get_count_from_local()
        if not total_words or not corpus_count:
            total_words, corpus_count = 0, 0
            for tokens in self:
                total_words += len(tokens)
                corpus_count += 1
        count_info = {
            "total_words": total_words,
            "corpus_count": corpus_count,
        }
        with self.count_json.open("w") as f:
            json.dump(count_info, f, indent=4)
        logger.mesg(dict_to_str(count_info), indent=2)
        return total_words, corpus_count
