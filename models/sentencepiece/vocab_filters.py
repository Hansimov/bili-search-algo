import re

from dataclasses import dataclass

from data_utils.videos.convert import CH_CJK

CH_MASK = "▂"
RE_CH_CJK = rf"[{CH_CJK}]"
RE_PURE_DIGITS = r"^\d+$"
RE_ASCII_TOKEN = r"^[A-Za-z0-9._=\-]+$"
RE_ASCII_ALPHA = r"^[A-Za-z]+$"
RE_ASCII_ALNUM = r"^[A-Za-z0-9]+$"
RE_CONNECTOR_RUN = r"[._=\-]{2,}"
RE_REPEATED_ASCII = r"(.)\1{2,}"
RE_BV_ID = r"^bv\d"

PT_CH_CJK = re.compile(RE_CH_CJK)
PT_PURE_DIGITS = re.compile(RE_PURE_DIGITS)
PT_ASCII_TOKEN = re.compile(RE_ASCII_TOKEN)
PT_ASCII_ALPHA = re.compile(RE_ASCII_ALPHA)
PT_ASCII_ALNUM = re.compile(RE_ASCII_ALNUM)
PT_CONNECTOR_RUN = re.compile(RE_CONNECTOR_RUN)
PT_REPEATED_ASCII = re.compile(RE_REPEATED_ASCII)
PT_BV_ID = re.compile(RE_BV_ID, re.IGNORECASE)


@dataclass(frozen=True)
class TokenProfile:
    token: str
    cjk_char_len: int
    has_cjk: bool
    is_ascii_token: bool
    is_ascii_alpha: bool
    is_ascii_alnum: bool
    is_pure_digits: bool
    has_connector: bool
    malformed: bool


def calc_cjk_char_len(token: str) -> int:
    return sum(1 for char in token if PT_CH_CJK.match(char))


def is_ascii_token(token: str) -> bool:
    return bool(PT_ASCII_TOKEN.fullmatch(token))


def is_ascii_alpha(token: str) -> bool:
    return bool(PT_ASCII_ALPHA.fullmatch(token))


def is_ascii_alnum(token: str) -> bool:
    return bool(PT_ASCII_ALNUM.fullmatch(token))


def is_malformed_token(token: str) -> bool:
    if len(token) >= 2 and "%" in token:
        return True
    if len(token) >= 2 and PT_CONNECTOR_RUN.search(token):
        return True
    if len(token) >= 2 and (
        token[0] in ".-_="
        or token[-1] in ".-_="
        or token[0] == CH_MASK
        or token[-1] == CH_MASK
    ):
        return True
    if PT_BV_ID.match(token):
        return True
    if is_ascii_token(token):
        connector_count = sum(1 for char in token if char in ".-_=")
        if connector_count * 2 >= len(token):
            return True
        if PT_REPEATED_ASCII.search(token):
            return True
    return False


def build_token_profile(token: str) -> TokenProfile:
    cjk_char_len = calc_cjk_char_len(token)
    ascii_token = is_ascii_token(token)
    return TokenProfile(
        token=token,
        cjk_char_len=cjk_char_len,
        has_cjk=bool(cjk_char_len),
        is_ascii_token=ascii_token,
        is_ascii_alpha=is_ascii_alpha(token),
        is_ascii_alnum=is_ascii_alnum(token),
        is_pure_digits=bool(PT_PURE_DIGITS.fullmatch(token)),
        has_connector=any(char in ".-_=" for char in token),
        malformed=is_malformed_token(token),
    )
