import pandas as pd
import pickle

from itertools import islice
from tclogger import Runtimer, TCLogbar, logger, logstr
from tclogger import chars_slice, dict_to_str, brk
from typing import Literal, Union

from configs.envs import TOKEN_FREQS_ROOT


class FasttextVocabLoader:
    def __init__(
        self,
        vocab_prefix: str,
        vocab_max_count: int = None,
        return_format: Literal["df", "dict"] = "dict",
        verbose: bool = False,
    ):
        self.vocab_prefix = vocab_prefix
        self.vocab_max_count = vocab_max_count
        self.return_format = return_format
        self.verbose = verbose
        self.vocab_dict = {}

    def load_vocab_from_csv(self) -> Union[dict[str, dict[str, int]], pd.DataFrame]:
        csv_path = TOKEN_FREQS_ROOT / f"{self.vocab_prefix}.csv"
        if self.verbose:
            logger.note("> Loading vocab from csv:")
            logger.file(f"  * {csv_path}")
        df_params = {"na_filter": False}
        if self.vocab_max_count:
            df_params["nrows"] = self.vocab_max_count
        df = pd.read_csv(csv_path, **df_params)
        df = df.drop_duplicates(subset=["token"], keep="first")
        df = df.sort_values(by=["doc_freq", "term_freq"], ascending=False)
        # df = df.head(self.vocab_max_count)
        # df = df.sort_values(by="term_freq", ascending=False)

        if self.verbose:
            vocab_info = {
                "vocab_size": df.shape[0],
                "min_term_freq": df.tail(1)["term_freq"].values[0],
                "min_doc_freq": df.tail(1)["doc_freq"].values[0],
            }
            logger.mesg(dict_to_str(vocab_info), indent=2)

        if self.return_format == "df":
            res = df
        else:
            term_freqs = dict(zip(df["token"], df["term_freq"]))
            doc_freqs = dict(zip(df["token"], df["doc_freq"]))

            res = {
                "vocab_size": df.shape[0],
                "term_freqs": term_freqs,
                "doc_freqs": doc_freqs,
            }

            # free up memory
            del df, term_freqs, doc_freqs

        return res

    def load_vocab_from_pickle(self) -> dict[str, dict[str, int]]:
        vocab_pickle_path = TOKEN_FREQS_ROOT / f"{self.vocab_prefix}.pickle"
        if self.verbose:
            logger.note("> Loading vocab from pickle:")
            logger.file(f"  * {vocab_pickle_path}")
        with vocab_pickle_path.open("rb") as f:
            pickle_dict = pickle.load(f)
        term_freqs = pickle_dict["term_freqs"]
        doc_freqs = pickle_dict["doc_freqs"]
        # since the freqs are already sorted (desc), no need to sort again
        # select the top `vocab_max_count` terms
        old_vocab_size = len(term_freqs)
        if self.vocab_max_count:
            term_freqs = dict(islice(term_freqs.items(), self.vocab_max_count))
            doc_freqs = dict(islice(doc_freqs.items(), self.vocab_max_count))
        new_vocab_size = len(term_freqs)

        if self.verbose:
            vocab_info = {
                "old_vocab_size": old_vocab_size,
                "new_vocab_size": new_vocab_size,
                "min_term_freq": min(term_freqs.values()),
                "min_doc_freq": min(doc_freqs.values()),
            }
            logger.mesg(dict_to_str(vocab_info), indent=2)

        res = {
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "term_freqs": term_freqs,
            "doc_freqs": doc_freqs,
        }

        # free up memory
        del pickle_dict, term_freqs, doc_freqs

        return res


class FasttextVocabMerger:
    def __init__(self) -> None:
        pass
