from collections import defaultdict
from tclogger import logger, logstr, dict_to_str, TCLogbar, chars_slice, decolored

from configs.envs import TOKEN_FREQS_ROOT, TOKEN_FREQ_PREFIX
from data_utils.videos.freq import read_token_freq_csv
from models.hanlp.pos import HanlpPosTagger

INCLUDE_POS_NAMES = ["名词", "动词", "人名"]
MID_POS_NAMES = ["数词", "方位词", "形容词", "量词"]
EXCLUDE_POS_NAMES = [
    *["连词", "副语素", "副词", "叹词", "后接成分"],
    *["拟声词", "介词", "代语素", "代词"],
    *["助词", "非语素字", "语气语素", "语气词"],
    # *["标点符号"] # do not add this, otherwise would remove some meaningful non-chinese chars
]


class TokenFreqPosTagger:
    def __init__(self, verbose: bool = False):
        self.token_freq_path = TOKEN_FREQS_ROOT / f"{TOKEN_FREQ_PREFIX}.csv"
        self.token_pos_path = TOKEN_FREQS_ROOT / f"{TOKEN_FREQ_PREFIX}_pos.csv"
        self.tagger = HanlpPosTagger(model_name="pku")
        self.verbose = verbose
        self.load_token_freq()

    def load_token_freq(self):
        if self.verbose:
            logger.note(f"> Loading token freq csv:")
            logger.file(f"  * {self.token_freq_path}")
        self.tf_df = read_token_freq_csv(self.token_freq_path)

    def tag_tokens(self):
        stats = defaultdict(int)
        max_idx = len(self.tf_df)
        bar = TCLogbar(total=max_idx)
        for idx, row in self.tf_df.iterrows():
            if idx >= max_idx:
                break
            token = str(row["token"])
            try:
                tags = self.tagger.tag_pos(token)
            except Exception as e:
                logger.warn(f"  * [{token}]: {e}")
                continue
            tag = tags[0]
            tag_name = self.tagger.tags_to_names(tag)
            self.tf_df.at[idx, "pos"] = tag_name
            if tag_name in EXCLUDE_POS_NAMES:
                stats["exclude"] = stats["exclude"] + 1
                line = f"  * {token} -> {tag_name}"
                logger.warn(line, verbose=self.verbose)
            elif tag_name in MID_POS_NAMES:
                stats["mid"] = stats["mid"] + 1
                line = f"  * {token} -> {tag_name}"
                logger.note(line, verbose=self.verbose)
            else:
                stats["include"] = stats["include"] + 1
                line = f"  * {logstr.file(token)} -> {logstr.success(tag_name)}"
                logger.mesg(line, verbose=self.verbose)
            bar.update(increment=1, desc=chars_slice(decolored(line), end=20))
        print()
        logger.note(f"> Saving token pos csv:")
        self.tf_df.to_csv(self.token_pos_path, index=False)
        logger.file(f"  * {self.token_pos_path}")
        logger.success(dict_to_str(stats), indent=2)


if __name__ == "__main__":
    tagger = TokenFreqPosTagger(verbose=False)
    tagger.tag_tokens()

    # python -m models.fasttext.pos
