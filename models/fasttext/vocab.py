import argparse
import pandas as pd
import pickle
import sys

from itertools import islice
from tclogger import Runtimer, TCLogbar, logger, logstr
from tclogger import chars_slice, dict_to_str, brk
from typing import Literal, Union

from configs.envs import TOKEN_FREQS_ROOT
from models.sentencepiece.filter import REGION_MONGO_FILTERS


class FasttextVocabLoader:
    def __init__(
        self,
        vocab_prefix: str,
        vocab_max_count: int = None,
        return_format: Literal["df", "dict"] = "df",
        order: Literal["sort", "shuffle"] = "sort",
        verbose: bool = False,
    ):
        self.vocab_prefix = vocab_prefix
        self.vocab_max_count = vocab_max_count
        self.return_format = return_format
        self.order = order
        self.verbose = verbose
        self.vocab_dict = {}

    def load_vocab_from_csv(
        self,
        return_format: Literal["df", "dict"] = None,
        order: Literal["sort", "shuffle"] = None,
        sort_first: Literal["doc_freq", "term_freq"] = "doc_freq",
    ) -> Union[dict[str, dict[str, int]], pd.DataFrame]:
        csv_path = TOKEN_FREQS_ROOT / f"{self.vocab_prefix}.csv"
        if self.verbose:
            logger.note("> Loading vocab from csv:")
            logger.file(f"  * {csv_path}")
        df_params = {"na_filter": False}
        if self.vocab_max_count:
            df_params["nrows"] = self.vocab_max_count

        df = pd.read_csv(csv_path, **df_params)
        df = df.drop_duplicates(subset=["token"], keep="first")

        order = order or self.order
        if order == "sort":
            if sort_first == "doc_freq":
                sort_list = ["doc_freq", "term_freq"]
            else:
                sort_list = ["term_freq", "doc_freq"]
            df = df.sort_values(by=sort_list, ascending=False)
        elif order == "shuffle":
            df = df.sample(frac=1).reset_index(drop=True)
        else:
            pass
        # df = df.head(self.vocab_max_count)
        # df = df.sort_values(by="term_freq", ascending=False)

        if self.verbose:
            vocab_info = {
                "vocab_size": df.shape[0],
                "min_term_freq": df.tail(1)["term_freq"].values[0],
                "min_doc_freq": df.tail(1)["doc_freq"].values[0],
            }
            logger.mesg(dict_to_str(vocab_info), indent=2)

        return_format = return_format or self.return_format
        if return_format == "df":
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
    def __init__(
        self,
        vocab_prefixes: list[str],
        output_prefix: str = "merged_video_texts",
        merged_count: int = None,
        min_freq: int = None,
        divide_algo: Literal["average", "ratio"] = "average",
        verbose: bool = False,
    ) -> None:
        self.vocab_prefixes = vocab_prefixes
        self.output_prefix = output_prefix
        self.merged_count = merged_count
        self.min_freq = min_freq
        self.divide_algo = divide_algo
        self.verbose = verbose

    def adjust_vocab_count(self, corpus_count: int) -> int:
        """30w vocab ~ 6000w corpus -> 1w vocab per 200w corpus"""
        return corpus_count // 200

    def load_vocabs(self):
        self.dfs = []
        logger.note("> Loading vocabs:")
        for idx, vocab_prefix in enumerate(self.vocab_prefixes):
            idx_str = f"[{idx+1}/{len(self.vocab_prefixes)}]"
            logger.mesg(f"  * {idx_str} {vocab_prefix}", verbose=self.verbose)
            logger.store_indent()
            logger.indent(4)
            loader = FasttextVocabLoader(
                vocab_prefix,
                vocab_max_count=self.merged_count,
                return_format="df",
                order="sort",
                verbose=self.verbose,
            )
            df = loader.load_vocab_from_csv()
            logger.restore_indent()

            # df = df.sort_values(by=["doc_freq", "term_freq"], ascending=False)
            self.dfs.append(df)
            # free up memory
            del df, loader

    def token_freq_to_df(self, token_freqs: dict[str, dict[str, int]]) -> pd.DataFrame:
        token_freqs_list = [
            {"token": token, **freqs} for token, freqs in token_freqs.items()
        ]
        df = pd.DataFrame(token_freqs_list)
        df = df.sort_values(by=["doc_freq", "term_freq"], ascending=False)
        return df

    def merge_vocabs(self):
        logger.note("> Merging vocabs:")
        vocab_prefix_num = len(self.vocab_prefixes)
        line_idx_by_prefix = [0] * vocab_prefix_num
        token_freqs = {}
        unique_token_count = 0
        logbar = TCLogbar(
            head=logstr.mesg("  * token:"), total=self.merged_count, show_at_init=True
        )
        while unique_token_count < self.merged_count:
            for prefix_idx in range(vocab_prefix_num):
                # get the current row info for the prefix vocab
                line_idx = line_idx_by_prefix[prefix_idx]
                row = self.dfs[prefix_idx].iloc[line_idx]
                token = row["token"]
                doc_freq = row["doc_freq"]
                term_freq = row["term_freq"]

                if token in token_freqs:
                    token_freqs[token]["doc_freq"] += doc_freq
                    token_freqs[token]["term_freq"] += term_freq
                else:
                    token_freqs[token] = {
                        "doc_freq": doc_freq,
                        "term_freq": term_freq,
                    }
                    unique_token_count += 1
                    token_str = chars_slice(token, end=8)
                    logbar.update(increment=1, desc=f"[{token_str}]: {doc_freq:>6}")
                line_idx_by_prefix[prefix_idx] += 1

        merged_df = self.token_freq_to_df(token_freqs)

        if self.verbose:
            merge_info = {
                "vocab_size": merged_df.shape[0],
                "min_doc_freq": merged_df.tail(1)["doc_freq"].values[0],
                "min_term_freq": merged_df.tail(1)["term_freq"].values[0],
            }
            logger.mesg(dict_to_str(merge_info), indent=2)

        self.merged_df = merged_df

        # free up memory
        del token_freqs, merged_df

    def save_merged_vocab(self):
        logger.note("> Saving merged vocab:")
        merged_vocab_path = TOKEN_FREQS_ROOT / f"{self.output_prefix}.csv"
        self.merged_df.to_csv(merged_vocab_path, index=False)
        logger.file(f"  * {merged_vocab_path}")

    def run(self):
        self.load_vocabs()
        self.merge_vocabs()
        self.save_merged_vocab()


class VocabMergerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-o", "--output-prefix", type=str, default="merged_video_texts"
        )
        self.add_argument("-mv", "--merged-count", type=int, default=None)
        self.add_argument("-mf", "--min-freq", type=int, default=None)
        self.add_argument(
            "-da",
            "--divide-algo",
            type=str,
            choices=["average", "ratio"],
            default="ratio",
        )

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


def main(args):
    REGIONS = list(REGION_MONGO_FILTERS.keys())
    vocab_prefixes = [f"video_texts_{region}_nt" for region in REGIONS]

    merger = FasttextVocabMerger(
        vocab_prefixes,
        merged_count=args.merged_count,
        output_prefix=args.output_prefix,
        divide_algo=args.divide_algo,
        verbose=True,
    )
    merger.run()


if __name__ == "__main__":
    arg_parser = VocabMergerArgParser()
    args = arg_parser.parse_args()

    with Runtimer():
        main(args)

    # python -m models.fasttext.vocab -mv 800000 -o merged_video_texts
