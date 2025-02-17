import pypinyin
import re
import zhconv

from collections import defaultdict
from tclogger import logger, TCLogbar, chars_slice, decolored

from configs.envs import TOKEN_FREQS_ROOT, TOKEN_FREQ_PREFIX
from data_utils.videos.freq import read_token_freq_csv

RE_ALPHA_DIGITS = re.compile(r"^[a-zA-Z0-9]+$")


class ChinesePinyinizer:
    def to_pinyin_choices(self, text: str) -> list[list[str]]:
        text = zhconv.convert(text, "zh-cn")
        pinyin_choices = pypinyin.pinyin(
            text, style=pypinyin.STYLE_NORMAL, heteronym=True
        )
        return pinyin_choices

    def to_pinyin_segs(self, text: str) -> list[str]:
        pinyin_choices = self.to_pinyin_choices(text)
        pinyin_segs = [choice[0] for choice in pinyin_choices]
        return pinyin_segs

    def to_pinyin_str(self, text: str, sep: str = "") -> str:
        if RE_ALPHA_DIGITS.match(text):
            return text
        pinyin_segs = self.to_pinyin_segs(text)
        pinyin_str = sep.join(pinyin_segs)
        return pinyin_str

    def to_pinyin_str_and_short(self, text: str, sep: str = "") -> tuple[str, str]:
        if RE_ALPHA_DIGITS.match(text):
            return text, text
        pinyin_segs = self.to_pinyin_segs(text)
        pinyin_str = sep.join(pinyin_segs)
        pinyin_short = "".join([seg[0] for seg in pinyin_segs])
        return pinyin_str, pinyin_short


class TokenPinyinDumper:
    def __init__(self, verbose: bool = False):
        self.token_freq_path = TOKEN_FREQS_ROOT / f"{TOKEN_FREQ_PREFIX}.csv"
        self.output_path = TOKEN_FREQS_ROOT / f"{TOKEN_FREQ_PREFIX}_pinyin.csv"
        self.pinyinizer = ChinesePinyinizer()
        self.verbose = verbose
        self.load_token_freq()

    def load_token_freq(self):
        if self.verbose:
            logger.note(f"> Loading token freq csv:")
            logger.file(f"  * {self.token_freq_path}")
        self.tf_df = read_token_freq_csv(self.token_freq_path)

    def dump_pinyin_data(self):
        max_idx = len(self.tf_df)
        bar = TCLogbar(total=max_idx)
        for idx, row in self.tf_df.iterrows():
            if idx >= max_idx:
                break
            token = str(row["token"])
            try:
                pinyin, short = self.pinyinizer.to_pinyin_str_and_short(token)
                if pinyin and pinyin != token:
                    self.tf_df.at[idx, "pinyin"] = pinyin
                    self.tf_df.at[idx, "short"] = short
            except Exception as e:
                logger.warn(f"  * [{token}]: {e}")
                continue
            bar.update(increment=1, desc=chars_slice(decolored(token), end=20))
        print()
        logger.note(f"> Saving token pinyin csv:")
        self.tf_df.to_csv(self.output_path, index=False)
        logger.file(f"  * {self.output_path}")


if __name__ == "__main__":
    dumper = TokenPinyinDumper(verbose=True)
    dumper.dump_pinyin_data()

    # python -m models.word.pinyin
