import argparse
import os
import sentencepiece as spm
import sys


from tclogger import logger, logstr, Runtimer, dict_to_str
from typing import Literal

from datasets.videos.data import SentencesDataloader, DataLoaderArgParser
from models.sentencepiece.edit import SentencePieceModelVocabEditor
from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.test import TEST_SENTENCES
from models.hanlp.tokenize import HanlpTokenizer


class SentencePieceModelTrainer:
    """
    Training Options:
    * https://github.com/google/sentencepiece/blob/master/doc/options.md
    """

    def __init__(
        self,
        add_dummy_prefix: bool = False,
        character_coverage: float = 0.999,
        input_sentence_size: int = 500000,
        minloglevel: int = 2,
        model_prefix="sentencepiece",
        model_type: Literal["unigram", "bpe", "char", "word"] = "unigram",
        num_threads: int = 16,
        split_by_unicode_script: bool = False,
        split_by_number: bool = False,
        shrinking_factor: float = 0.9,
        treat_whitespace_as_suffix: bool = False,
        user_defined_symbols="▁",
        vocab_size: int = 32000,
        overwrite: bool = True,
    ):
        self.train_params = {
            "add_dummy_prefix": add_dummy_prefix,
            "character_coverage": character_coverage,
            "input_sentence_size": input_sentence_size,
            "minloglevel": minloglevel,
            "model_type": model_type,
            "model_prefix": model_prefix,
            "num_threads": num_threads,
            "split_by_unicode_script": split_by_unicode_script,
            "split_by_number": split_by_number,
            "shrinking_factor": shrinking_factor,
            "treat_whitespace_as_suffix": treat_whitespace_as_suffix,
            "user_defined_symbols": user_defined_symbols,
            "vocab_size": vocab_size,
        }
        self.model_prefix = model_prefix
        self.model_file = f"{model_prefix}.model"
        self.overwrite = overwrite

    def init_data_loader(self, data_loader: SentencesDataloader):
        self.data_loader = data_loader

    def delete_existed_model(self):
        model_prefix = str(self.model_prefix)
        logger.warn(f"  ! WARNING: You are deleting model:", end=" ")
        logger.note(f"[{model_prefix}]")
        confirmation = input(
            f'  > Type "{logstr.file(model_prefix)}" to confirm deletion: '
        )
        if confirmation != model_prefix:
            logger.mesg(f"  * Skip delete model file: [{model_prefix}]")
        else:
            os.remove(f"{model_prefix}.model")
            os.remove(f"{model_prefix}.vocab")
            logger.warn(f"  * Model file deleted: [{model_prefix}]")

    def train(self):
        logger.note("> Training ...")
        logger.mesg(dict_to_str(self.train_params), indent=2)

        if self.overwrite and os.path.exists(self.model_file):
            self.delete_existed_model()

        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter(self.data_loader),
            **self.train_params,
        )

    def test(self, test_sentences: list[str], compare_baseline: bool = False):
        logger.note("> Testing ...")
        tokenizer = SentenceFullTokenizer(self.model_file)
        if compare_baseline:
            hanlp_tokenizer = HanlpTokenizer()
        for sentence in test_sentences:
            tokens = tokenizer.tokenize(sentence)
            pretty_tokens = tokenizer.stringify(tokens)
            logger.file(f"  * {pretty_tokens}")
            if compare_baseline:
                hanlp_tokens = hanlp_tokenizer.tokenize(sentence)
                pretty_hanlp_tokens = tokenizer.stringify(hanlp_tokens)
                logger.success(f"  * {pretty_hanlp_tokens}")


class ModelTrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-cc", "--character-coverage", type=float, default=0.999)
        self.add_argument("-is", "--input-sentence-size", type=int, default=500000)
        self.add_argument("-mp", "--model-prefix", type=str, default="sentencepiece")
        self.add_argument("-mt", "--model-type", type=str, default="unigram")
        self.add_argument("-sf", "--shrinking-factor", type=float, default=0.9)
        self.add_argument("-vs", "--vocab-size", type=int, default=32000)
        self.add_argument("-k", "--keep-exist-model", action="store_true")
        self.add_argument("-t", "--test-only", action="store_true")
        self.add_argument("-cb", "--compare-baseline", action="store_true")
        self.add_argument("-e", "--edit-model", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    data_loader_parser = DataLoaderArgParser(add_help=False)
    model_trainer_parser = ModelTrainerArgParser(add_help=False)
    merged_parser = argparse.ArgumentParser(
        parents=[data_loader_parser, model_trainer_parser]
    )
    args, unknown_args = merged_parser.parse_known_args(sys.argv[1:])

    trainer = SentencePieceModelTrainer(
        character_coverage=args.character_coverage,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        shrinking_factor=args.shrinking_factor,
        vocab_size=args.vocab_size,
        overwrite=not args.keep_exist_model,
    )
    if not args.test_only:
        logger.note("> Initiating data loader ...")
        data_params = {
            "dbname": args.dbname,
            "collect_name": args.collect_name,
            "data_fields": args.data_fields.split(",") if args.data_fields else None,
            "max_batch": args.max_batch,
            "batch_size": args.batch_size,
            "estimate_count": args.estimate_count,
        }
        data_loader = SentencesDataloader(
            **data_params, show_at_init=False, verbose=True
        )
        logger.mesg(dict_to_str(data_params), indent=2)
        trainer.init_data_loader(data_loader)
        with Runtimer() as timer:
            trainer.train()
    if args.edit_model:
        editor = SentencePieceModelVocabEditor(trainer.model_file, verbose=True)
        editor.edit()
    trainer.test(TEST_SENTENCES, compare_baseline=args.compare_baseline)

    # python -m models.sentencepiece.train -mp sp_480m_400k_0.9995_0.9 -mb 48000 -vs 400000 -cc 0.9995 -sf 0.9 -e
    # python -m models.sentencepiece.train -mp sp_480m_400k_0.9995_0.9 -t
    # python -m models.sentencepiece.train -mp sp_users_1kw_10k -cn users -mb 1000 -vs 10000 -cc 1.0 -e
    # python -m models.sentencepiece.train -mp sp_wiki_1w_400k -db zhwiki -cn pages -bs 1000 -ec -mb 10 -vs 10000 -e
    # python -m models.sentencepiece.train -mp sp_wiki_all_400k_0.9995 -db zhwiki -cn pages -bs 1000 -vs 400000 -cc 0.9995 -e
    # python -m models.sentencepiece.train -mp sp_wiki_all_400k_0.9999 -t
    # python -m models.sentencepiece.train -mp sp_400k_merged -t
