import hanlp
import hanlp.pretrained
import jieba

from collections import Counter
from tclogger import logger, dict_to_str
from typing import Literal, Union


class WordCounter:
    HANLP_TOKENIZE_MODELS = {
        "coarse": hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,
        "fine": hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH,
    }
    JIEBA_TOKENIZE_METHODS = {
        "cut": jieba.lcut,
        "cut_for_search": jieba.lcut_for_search,
    }

    def __init__(self):
        self.tokenizer = None

    def load_tokenizer(
        self,
        engine: Literal["jieba", "hanlp"] = "jieba",
        hanlp_level: Literal["coarse", "fine"] = "coarse",
        jieba_method: Literal["cut", "cut_for_search"] = "cut",
    ):
        self.engine = engine
        if engine == "hanlp":
            self.tokenizer = hanlp.load(
                WordCounter.HANLP_TOKENIZE_MODELS[hanlp_level.lower()]
            )
        else:
            self.tokenizer = WordCounter.JIEBA_TOKENIZE_METHODS[jieba_method.lower()]

    def tokenize(self, text: Union[str, list[str]], verbose: bool = False, **kwargs):
        logger.enter_quiet(not verbose)
        logger.note(f"> Tokenizing:", end=" ")
        logger.mesg(f"[{text}]")
        if isinstance(text, list):
            tokenize_res = [self.tokenizer(subtext, **kwargs) for subtext in text]
        else:
            tokenize_res = self.tokenizer(text, **kwargs)
        logger.success(tokenize_res)
        logger.exit_quiet(not verbose)
        return tokenize_res

    def count(self, words: list, threshold: int = 1, verbose: bool = False):
        logger.enter_quiet(not verbose)
        if any(isinstance(word, list) for word in words):
            words = [word for sublist in words for word in sublist]

        counter = Counter(words)
        word_counts = dict(
            sorted(counter.items(), key=lambda item: item[1], reverse=True)
        )
        word_counts = {
            word: count for word, count in word_counts.items() if count >= threshold
        }
        logger.success(dict_to_str(word_counts))
        logger.exit_quiet(not verbose)
        return word_counts


if __name__ == "__main__":
    texts = ["我爱北京天安门", "天安门上太阳升"]
    counter = WordCounter()
    # Counter.load_tokenizer(engine="hanlp", hanlp_level="coarse")
    counter.load_tokenizer(engine="jieba", jieba_method="cut_for_search")
    tokens_list = counter.tokenize(texts, verbose=False)
    word_counts = counter.count(tokens_list, verbose=True)

    # python -m models.word.counter
