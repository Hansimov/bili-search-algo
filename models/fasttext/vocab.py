import pandas as pd
import pickle

from tclogger import logger, dict_to_str

from configs.envs import TOKEN_FREQS_ROOT


class FasttextVocabLoader:
    def __init__(
        self, vocab_prefix: str, vocab_max_count: int = None, verbose: bool = False
    ):
        self.vocab_prefix = vocab_prefix
        self.vocab_max_count = vocab_max_count
        self.verbose = verbose
        self.vocab_dict = {}

    def load_vocab_from_csv(self) -> dict[str, dict[str, int]]:
        csv_path = TOKEN_FREQS_ROOT / f"{self.vocab_prefix}.csv"
        if self.verbose:
            logger.note("> Loading vocab csv from file:")
            logger.file(f"  * {csv_path}")
        df = pd.read_csv(csv_path)
        # df = df.sort_values(by="doc_freq", ascending=False)
        if self.vocab_max_count:
            df = df.head(self.vocab_max_count)
        df = df.sort_values(by="term_freq", ascending=False)
        self.term_freqs = dict(zip(df["token"], df["term_freq"]))
        self.doc_freqs = dict(zip(df["token"], df["doc_freq"]))
        if self.verbose:
            vocab_info = {
                "vocab_size": df.shape[0],
                "min_term_freq": df.tail(1)["term_freq"].values[0],
                "min_doc_freq": df.tail(1)["doc_freq"].values[0],
            }
            logger.mesg(dict_to_str(vocab_info), indent=2)

        return {
            "term_freqs": self.term_freqs,
            "doc_freqs": self.doc_freqs,
        }

    def load_vocab_from_pickle(self) -> dict[str, dict[str, int]]:
        vocab_pickle_path = TOKEN_FREQS_ROOT / f"{self.vocab_prefix}.pickle"
        if self.verbose:
            logger.note("> Loading vocab pickle from file:")
            logger.file(f"  * {vocab_pickle_path}")
        with vocab_pickle_path.open("rb") as f:
            pickle_dict = pickle.load(f)
        term_freqs = pickle_dict["term_freqs"]
        doc_freqs = pickle_dict["doc_freqs"]
        # since the freqs are already sorted, no need to sort again
        if self.vocab_max_count:
            term_freqs = term_freqs[: self.vocab_max_count]
            doc_freqs = doc_freqs[: self.vocab_max_count]
        if self.verbose:
            vocab_info = {
                "vocab_size": len(term_freqs),
                "min_term_freq": min(term_freqs.values()),
                "min_doc_freq": min(doc_freqs.values()),
            }
            logger.mesg(dict_to_str(vocab_info), indent=2)

        return {
            "term_freqs": term_freqs,
            "doc_freqs": doc_freqs,
        }


class FasttextVocabMerger:
    def __init__(self) -> None:
        pass
