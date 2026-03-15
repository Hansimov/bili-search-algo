import re
import unicodedata


RE_MULTISPACE = re.compile(r"\s+")
RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
RE_DIGITS = re.compile(r"\d")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_ASCII_TOKEN_CHARS = re.compile(r"[a-z0-9\-.]")
RE_ASCII_MIXED = re.compile(r"[0-9A-Za-z\-.]")
RE_ALPHA = re.compile(r"^[a-z]+$")
RE_ALNUM = re.compile(r"^[a-z0-9]+$")
RE_VIDEO_ID = re.compile(r"^(?:av\d{4,}|bv[0-9a-z]{4,})$", re.IGNORECASE)
RE_CJK_COMMA_KEEP = re.compile(
    r"^[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{1,12}，启动$"
)
RE_TEMPLATE_NOISE = re.compile(
    r"(?:sharefrom|sourcefrom|fromsource|livefromsource|showredpacket|navhide)",
    re.IGNORECASE,
)
RE_CTA_NOISE = re.compile(
    r"(?:一键三连|求三连|求关注|关注我|关注我们|快来围观|账号已注销|up主激励计划)"
)
RE_CTA_CONTEXT = re.compile(
    r"(?:求个?|请|记得|欢迎|感谢|给个|给我|点个|点点|建议|多多|赶紧|喜欢|希望|支持|一下|吧|呀|哦|喔|联系|回复|留言|不迷路)"
)
RE_PROMO_CONTEXT = re.compile(
    r"(?:主页|首页|简介|评论区|链接|福利|惊喜|置顶|橱窗|下单|进群|传送门)"
)
RE_LIVEROOM_NOISE = re.compile(
    r"(?:直播间(?:地址|传送门|号)|[来在去上进回到].{0,2}直播间|(?:b站|抖音|斗鱼|虎牙|主播)直播间)"
)
RE_COPYRIGHT_NOISE = re.compile(
    r"(?:如侵删|侵删|未经允许|禁止搬运|禁止转载|非商用|素材来源)"
)
RE_REPOST_NOISE = re.compile(
    r"(?:仅搬运|勿转载|可转载|可搬运|接投稿|代投稿|搬运自|投稿请|投稿至|投稿见|原搬运|原投稿|转载自)"
)

COMMON_ASCII_BIGRAMS = {
    "al",
    "an",
    "ar",
    "at",
    "ch",
    "cl",
    "co",
    "de",
    "ea",
    "ed",
    "el",
    "en",
    "er",
    "es",
    "ge",
    "ha",
    "he",
    "hi",
    "ia",
    "ic",
    "ie",
    "in",
    "io",
    "is",
    "it",
    "ke",
    "la",
    "le",
    "li",
    "ll",
    "lo",
    "ma",
    "me",
    "mi",
    "na",
    "nd",
    "ne",
    "ng",
    "ni",
    "nt",
    "on",
    "oo",
    "or",
    "ou",
    "ov",
    "pa",
    "ph",
    "pl",
    "pr",
    "ra",
    "re",
    "ri",
    "ro",
    "rs",
    "rt",
    "sa",
    "se",
    "sh",
    "si",
    "so",
    "sp",
    "st",
    "ta",
    "te",
    "th",
    "ti",
    "to",
    "tr",
    "tu",
    "ua",
    "ud",
    "un",
    "ur",
    "ve",
    "vi",
    "wh",
    "wi",
    "wo",
    "ya",
    "yo",
}
COMMON_ASCII_TRIGRAMS = {
    "ack",
    "age",
    "air",
    "all",
    "ame",
    "and",
    "ani",
    "ard",
    "art",
    "ate",
    "ati",
    "ava",
    "awa",
    "ayo",
    "bel",
    "ble",
    "boo",
    "cha",
    "che",
    "chi",
    "com",
    "cro",
    "der",
    "ear",
    "edi",
    "end",
    "ent",
    "era",
    "ers",
    "est",
    "eve",
    "for",
    "fun",
    "ger",
    "ght",
    "har",
    "igh",
    "ill",
    "ing",
    "ion",
    "ita",
    "ive",
    "jav",
    "ker",
    "lan",
    "lay",
    "leg",
    "lic",
    "man",
    "men",
    "min",
    "mod",
    "nal",
    "nic",
    "nia",
    "ome",
    "one",
    "oni",
    "ord",
    "ork",
    "oud",
    "our",
    "out",
    "ove",
    "pac",
    "pho",
    "pla",
    "pro",
    "ran",
    "rea",
    "ring",
    "rod",
    "san",
    "sek",
    "shi",
    "son",
    "sou",
    "sta",
    "str",
    "tal",
    "ter",
    "the",
    "tra",
    "und",
    "ver",
    "with",
    "you",
}

WIKI_NOISE_EXACT = {
    "参考来源",
    "外部连结",
    "外部链接",
    "存档副本",
    "开放街图",
    "扩展阅读",
    "别名重定向",
    "移动重定向",
    "简繁重定向",
}
CTA_NOISE_EXACT = {
    "一键三连",
    "快来围观",
    "求三连",
    "求关注",
    "账号已注销",
}
CTA_NOISE_PREFIXES = ("up主激励计划", "关注我")
CTA_ACTION_WORDS = ("点赞", "投币", "收藏", "转发", "关注", "三连", "私信")
CTA_PHRASE_ACTION_WORDS = ("点赞", "投币", "转发", "关注", "三连", "私信")


def normalize_spaces(text: str) -> str:
    return RE_MULTISPACE.sub(" ", text).strip()


def normalize_common_token(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = normalize_spaces(text)
    if lowercase:
        text = text.lower()
    return text


def contains_cjk(text: str) -> bool:
    return bool(RE_CJK.search(text))


def count_cjk(text: str) -> int:
    return sum(1 for char in text if RE_CJK.match(char))


def count_digits(text: str) -> int:
    return len(RE_DIGITS.findall(text))


def count_latin(text: str) -> int:
    return len(RE_LATIN.findall(text))


def count_ascii_token_chars(text: str) -> int:
    return len(RE_ASCII_TOKEN_CHARS.findall(text))


def count_ascii_mixed_chars(text: str) -> int:
    return len(RE_ASCII_MIXED.findall(text))


def calc_token_units(text: str) -> int:
    return count_cjk(text) * 3 + count_ascii_mixed_chars(text)


def max_consecutive_non_vowels(text: str) -> int:
    max_run = 0
    current_run = 0
    for char in text.lower():
        if not char.isalpha() or char in "aeiou":
            current_run = 0
            continue
        current_run += 1
        if current_run > max_run:
            max_run = current_run
    return max_run


def uncommon_bigram_ratio(token: str) -> float:
    if len(token) < 2:
        return 0.0
    bigrams = [token[idx : idx + 2] for idx in range(len(token) - 1)]
    uncommon_count = sum(bigram not in COMMON_ASCII_BIGRAMS for bigram in bigrams)
    return uncommon_count / len(bigrams)


def count_common_trigrams(token: str) -> int:
    if len(token) < 3:
        return 0
    return sum(
        token[idx : idx + 3] in COMMON_ASCII_TRIGRAMS for idx in range(len(token) - 2)
    )


def looks_like_random_ascii(token: str) -> bool:
    token = token.lower()
    if not RE_ALPHA.fullmatch(token):
        return False
    if len(token) <= 4:
        return False
    vowel_count = sum(char in "aeiou" for char in token)
    consonant_run = max_consecutive_non_vowels(token)
    uncommon_ratio = uncommon_bigram_ratio(token)
    trigram_hits = count_common_trigrams(token)
    if len(token) >= 7 and vowel_count == 0 and "y" not in token:
        return True
    if len(token) >= 8 and vowel_count == 0:
        return True
    if len(token) >= 12 and consonant_run >= 5:
        return True
    if len(token) >= 12 and consonant_run >= 4 and uncommon_ratio >= 0.55:
        return True
    if len(token) >= 14 and vowel_count <= 2 and trigram_hits == 0:
        return True
    return False


def looks_like_random_mixed_ascii(token: str) -> bool:
    token = token.lower()
    if len(token) < 6:
        return False
    if not RE_ALNUM.fullmatch(token):
        return False
    letters = sum(char.isalpha() for char in token)
    digits = sum(char.isdigit() for char in token)
    if not letters or not digits:
        return False
    vowel_count = sum(char in "aeiou" for char in token)
    consonant_run = max_consecutive_non_vowels(token)
    letter_only = "".join(char for char in token if char.isalpha())
    letter_consonant_run = max_consecutive_non_vowels(letter_only) if letter_only else 0
    uncommon_ratio = uncommon_bigram_ratio(letter_only) if letter_only else 0.0
    trigram_hits = count_common_trigrams(letter_only)
    if letters >= 4 and digits <= 3 and vowel_count <= 1:
        return True
    if letters >= 6 and len(token) >= 10 and consonant_run >= 5:
        return True
    if letters >= 6 and len(token) >= 9 and letter_consonant_run >= 5:
        return True
    if (
        letters >= 6
        and len(token) >= 10
        and letter_consonant_run >= 4
        and uncommon_ratio >= 0.6
        and trigram_hits == 0
    ):
        return True
    return False


def should_keep_cjk_comma_phrase(token: str) -> bool:
    return bool(RE_CJK_COMMA_KEEP.fullmatch(token))


def is_video_id_token(token: str) -> bool:
    return bool(RE_VIDEO_ID.fullmatch(token))


def count_keyword_hits(token: str, keywords: tuple[str, ...]) -> int:
    return sum(keyword in token for keyword in keywords)


def is_cta_phrase(token: str) -> bool:
    action_hits = count_keyword_hits(token, CTA_ACTION_WORDS)
    if action_hits >= 2:
        return True
    if action_hits == 0:
        return False
    if RE_PROMO_CONTEXT.search(token):
        return True
    if RE_CTA_CONTEXT.search(token):
        return True
    if token.startswith(("求", "请", "记得", "欢迎", "感谢", "给个", "点个")):
        return True
    if token.endswith(("支持", "支持一下", "联系", "留言", "回复", "不迷路")):
        return True
    if "私信" in token and token != "私信":
        return True
    for action in CTA_PHRASE_ACTION_WORDS:
        if token.endswith(action):
            prefix = token[: -len(action)]
            if prefix and len(prefix) <= 4 and contains_cjk(prefix):
                return True
    return False


def is_title_template_noise(token: str) -> bool:
    if "直播间" in token and token != "直播间":
        return True
    if RE_COPYRIGHT_NOISE.search(token):
        return True
    if RE_REPOST_NOISE.search(token):
        return True
    if RE_LIVEROOM_NOISE.search(token):
        return True
    return False


def is_curated_noise_token(token: str) -> bool:
    if is_video_id_token(token):
        return True
    if token in WIKI_NOISE_EXACT or token in CTA_NOISE_EXACT:
        return True
    if token.endswith("重定向"):
        return True
    if token.startswith(CTA_NOISE_PREFIXES):
        return True
    if RE_CTA_NOISE.search(token):
        return True
    if is_cta_phrase(token):
        return True
    if RE_TEMPLATE_NOISE.search(token):
        return True
    if is_title_template_noise(token):
        return True
    return False
