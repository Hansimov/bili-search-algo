import argparse
import os
import sentencepiece as spm
import sys


from collections.abc import Iterable
from tclogger import logger, logstr, Runtimer, dict_to_str
from typing import Literal

from models.sentencepiece.data import SentencesDataloader
from models.sentencepiece.edit import SentencePieceModelVocabEditor
from models.sentencepiece.tokenize import SentenceFullTokenizer
from models.sentencepiece.test import TEST_SENTENCES


class SentencePieceModelTrainer:
    """
    Training Options:
    * https://github.com/google/sentencepiece/blob/master/doc/options.md
    """

    def __init__(
        self,
        add_dummy_prefix: bool = False,
        character_coverage: float = 0.999,
        input_sentence_size: int = 1000000,
        minloglevel: int = 2,
        model_prefix="sentencepiece",
        model_type: Literal["unigram", "bpe", "char", "word"] = "unigram",
        num_threads: int = 16,
        split_by_unicode_script: bool = False,
        split_by_number: bool = False,
        shrinking_factor: float = 0.75,
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

    def load_data(
        self,
        data_loader: Iterable[str] = None,
        data_fields: str = None,
        max_batch: int = 100,
        batch_size: int = 10000,
    ):
        logger.note("> Loading data ...")
        if data_loader:
            self.data_loader = data_loader
        else:
            if data_fields and isinstance(data_fields, str):
                data_fields = data_fields.split(",")

            self.data_params = {
                "data_fields": data_fields,
                "max_batch": max_batch,
                "batch_size": batch_size,
            }
            logger.mesg(dict_to_str(self.data_params), indent=2)

            self.data_loader = SentencesDataloader(
                max_batch=max_batch,
                batch_size=batch_size,
                show_at_init=False,
                data_fields=data_fields,
                verbose=True,
            )

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

    def test(self, test_sentences: list[str]):
        logger.note("> Testing ...")
        tokenizer = SentenceFullTokenizer(self.model_file)
        for sentence in test_sentences:
            tokens = tokenizer.tokenize(sentence)
            pretty_tokens = tokenizer.stringify(tokens)
            logger.mesg(f"  * {pretty_tokens}")


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-cc", "--character-coverage", type=float, default=0.999)
        self.add_argument("-mb", "--max-batch", type=int, default=20000)
        self.add_argument("-mp", "--model-prefix", type=str, default="sentencepiece")
        self.add_argument("-mt", "--model-type", type=str, default="unigram")
        self.add_argument("-sf", "--shrinking-factor", type=float, default=0.75)
        self.add_argument("-vs", "--vocab-size", type=int, default=32000)
        self.add_argument("-df", "--data-fields", type=str, default=None)
        self.add_argument("-t", "--test-only", action="store_true")
        self.add_argument("-k", "--keep-exist-model", action="store_true")
        self.add_argument("-e", "--edit-model", action="store_true")
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])


if __name__ == "__main__":
    args = ArgParser().args
    trainer = SentencePieceModelTrainer(
        character_coverage=args.character_coverage,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        shrinking_factor=args.shrinking_factor,
        vocab_size=args.vocab_size,
        overwrite=not args.keep_exist_model,
    )
    if not args.test_only:
        trainer.load_data(
            data_fields=args.data_fields, max_batch=args.max_batch, batch_size=10000
        )
        with Runtimer() as timer:
            trainer.train()
    if args.edit_model:
        editor = SentencePieceModelVocabEditor(trainer.model_file, verbose=True)
        editor.edit()
    trainer.test(TEST_SENTENCES)

    # python -m models.sentencepiece.train -mp sp_480m_400k_0.9995_0.9 -mb 48000 -vs 400000 -cc 0.9995 -sf 0.9 -e
    # python -m models.sentencepiece.train -mp sp_480m_400k_0.9995_0.9 -t
    # python -m models.sentencepiece.train -mp sp_48kw_40k_name -mb 48000 -vs 40000 -df owner.name -e
