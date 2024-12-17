import argparse
import Pyro5.api
import Pyro5.core
import Pyro5.server
import sys

from gensim.models import FastText, KeyedVectors
from pathlib import Path

from tclogger import logger, logstr, dict_to_str, brk, Runtimer
from typing import Union

from configs.envs import FASTTEXT_CKPT_ROOT, PYRO_ENVS
from models.sentencepiece.tokenizer import SentenceFullTokenizer

PYRO_NS = "fasttext_model_runner"


@Pyro5.server.behavior(instance_mode="single")
class FasttextModelRunner:
    def __init__(
        self,
        model_prefix: Union[str, Path] = "fasttext",
        restrict_vocab: int = 150000,
        verbose: bool = False,
    ):
        self.model_prefix = model_prefix
        self.restrict_vocab = restrict_vocab
        self.verbose = verbose

    @Pyro5.server.expose
    def list_models(self) -> dict:
        logger.file(f"> List local models:", verbose=self.verbose)
        model_files = list(FASTTEXT_CKPT_ROOT.glob(f"*.model"))
        model_files.sort()
        model_info_dict = {}
        for model_file in model_files:
            model_prefix = model_file.stem
            vocab_file = (
                FASTTEXT_CKPT_ROOT / f"{model_prefix}.model.wv.vectors_vocab.npy"
            )
            model_size_in_mb = round(model_file.stat().st_size / 1024 / 1024)
            try:
                vocab_size_in_mb = round(vocab_file.stat().st_size / 1024 / 1024)
            except Exception as e:
                vocab_size_in_mb = None
            model_info_dict[model_prefix] = (model_size_in_mb, vocab_size_in_mb)
        logger.file(dict_to_str(model_info_dict), verbose=self.verbose)
        return model_info_dict

    def load_model(self):
        self.model_path = FASTTEXT_CKPT_ROOT / f"{self.model_prefix}.model"
        if self.verbose:
            logger.note(f"> Loading model:")
            logger.mesg(f"  * {self.model_path}")
            logger.mesg(f"  * restrict_vocab: {logstr.file(brk(self.restrict_vocab))}")
        self.model = FastText.load(str(self.model_path))

    def preprocess_words(self, words: Union[str, list[str]]):
        if isinstance(words, list):
            return [w.lower() for w in words]
        else:
            return words.lower()

    @Pyro5.server.expose
    def wv_func(self, func: str, *args, **kwargs):
        if self.verbose:
            logger.note(f"> model.wv.{func}:")
            logger.mesg(f"  * args  : {args}")
            logger.mesg(f"  * kwargs: {kwargs}")
        return getattr(self.model.wv, func)(*args, **kwargs)

    @Pyro5.server.expose
    def most_similar(
        self,
        positive: list = None,
        negative: list = None,
        topn: int = 10,
        restrict_vocab: int = None,
    ):
        results = self.model.wv.most_similar(
            positive=positive,
            negative=negative,
            topn=topn,
            restrict_vocab=restrict_vocab or self.restrict_vocab,
        )
        return results

    @Pyro5.server.expose
    def test(self, test_words: list):
        logger.note(f"> Testing:")
        for word in test_words:
            word = self.preprocess_words(word)
            logger.mesg(f"  * [{logstr.file(word)}]:")
            results = self.most_similar(positive=word, topn=10)[:6]
            for result in results:
                res_word, res_score = result
                logger.success(f"    * {res_score:>.4f}: {res_word}")
        logger.file(f"* {self.model_prefix}")

    def test_func(self, test_words: list):
        logger.note(f"> Testing (func):")
        for word in test_words:
            word = self.preprocess_words(word)
            logger.mesg(f"  * [{logstr.file(word)}]:")
            results = self.wv_func("most_similar", positive=word, topn=10)[:6]
            for result in results:
                res_word, res_score = result
                logger.success(f"    * {res_score:>.4f}: {res_word}")
        logger.file(f"* {self.model_prefix}")


class FasttextModelRunnerServer:
    def __init__(
        self,
        nameserver: str = PYRO_NS,
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.nameserver = nameserver
        self.host = host
        self.port = port
        self.verbose = verbose

    def serve(self, runner: FasttextModelRunner):
        if self.verbose:
            logger.note(f"> Running as remote server:")
            logger.mesg(f"  * {self.host}:{self.port}")
        # Pyro5.server.serve({runner: self.nameserver}, host=self.host, port=self.port)
        daemon = Pyro5.server.Daemon(host=self.host, port=self.port)
        self.uri = daemon.register(runner, objectId=self.nameserver, weak=True)
        logger.file(f"  * {self.uri}", verbose=self.verbose)
        daemon.requestLoop()


class FasttextModelRunnerClient:
    def __init__(
        self,
        nameserver: str = PYRO_NS,
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.uri = f"PYRO:{nameserver}@{host}:{port}"
        self.runner = Pyro5.api.Proxy(self.uri)


class FasttextModelRunnerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-m", "--model-prefix", type=str, default="fasttext")
        self.add_argument("-v", "--restrict-vocab", type=int, default=150000)
        self.add_argument("-l", "--list-models", action="store_true")
        self.add_argument("-n", "--nameserver", type=str, default=PYRO_NS)
        self.add_argument("-s", "--host", type=str, default=PYRO_ENVS["host"])
        self.add_argument("-p", "--port", type=int, default=PYRO_ENVS["port"])
        self.add_argument("-r", "--remote", action="store_true")
        self.add_argument("-t", "--test", action="store_true")
        self.add_argument("-tc", "--test-client", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    from models.fasttext.test import TEST_KEYWORDS

    parser = FasttextModelRunnerArgParser()
    args = parser.parse_args()

    runner = FasttextModelRunner(
        model_prefix=args.model_prefix,
        restrict_vocab=args.restrict_vocab,
        verbose=True,
    )

    if args.list_models:
        runner.list_models()
    else:
        remote_args = {
            "nameserver": args.nameserver,
            "host": args.host,
            "port": args.port,
        }

        if not args.test_client:
            with Runtimer() as timer:
                runner.load_model()

        if args.test:
            # runner.test(TEST_KEYWORDS)
            runner.test_func(TEST_KEYWORDS)
        elif args.test_client:
            client = FasttextModelRunnerClient(**remote_args)
            client.runner.test(TEST_KEYWORDS)
            # res = client.runner.list_models()
            # logger.success(dict_to_str(res))
        elif args.remote:
            runner.verbose = False
            server = FasttextModelRunnerServer(**remote_args, verbose=True)
            server.serve(runner)
        else:
            raise ValueError("× Invalid running mode!")

    # List models:
    # python -m models.fasttext.run -l

    # Test models:
    # python -m models.fasttext.run -t -m fasttext_tid_all_mv_60w -v 150000
    # python -m models.fasttext.run -t -m fasttext_tid_all_mv_30w -v 150000

    # Run remote server:
    # python -m models.fasttext.run -r -m fasttext_tid_all_mv_30w -v 150000

    # Test remote client:
    # python -m models.fasttext.run -tc
