import re

from tclogger import dict_get


class DocSentenceConverter:
    """
    GB2312 编码表:
        - https://www.toolhelper.cn/Encoding/GB2312
        - A1A0~A3FE (JP), A6A0~A9FE (CN)
    CJK Unicode Tables:
        - https://www.khngai.com/chinese/charmap/tbluni.php
        - 0x4E00~0x9FFF (CN)
    Unicode Kanji Table:
        - http://www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
        - 0x3040~0x30FF (JP)
    """

    RE_CJK = r"[\u4E00-\u9FFF\u3040-\u30FF]"
    RE_CJK_SPACE = rf"(?<={RE_CJK})\s+(?={RE_CJK})"

    def doc_to_sentence(self, doc: dict) -> str:
        author = dict_get(doc, "owner.name", "")
        author_str = f"{author}:" if author else ""

        title = dict_get(doc, "title", "")
        title_str = f"{title}." if title else ""

        desc = dict_get(doc, "desc", "")
        desc_str = f"{desc}." if desc else ""

        rtags = dict_get(doc, "rtags", "")
        tags = dict_get(doc, "tags", "")
        tags_str = f"{rtags}, {tags}." if tags else f"{rtags}."
        tags_str = tags_str.replace(" ", "")

        sentence = f"{author_str}{title_str}{desc_str}{tags_str}"
        return sentence

    def lower_and_strip(self, sentence: str) -> str:
        return sentence.lower().strip()

    def remove_whitespaces_among_cjk(self, sentence: str) -> str:
        return re.sub(self.RE_CJK_SPACE, "", sentence)

    def convert(self, doc: dict) -> str:
        sentence = self.doc_to_sentence(doc)
        sentence = self.lower_and_strip(sentence)
        sentence = self.remove_whitespaces_among_cjk(sentence)
        return sentence


if __name__ == "__main__":
    from tclogger import logger

    sentence = " 这是 一段 中文。这是日语：これは 日本語 です。 Here is some English. "
    logger.note(sentence)
    converter = DocSentenceConverter()
    sentence = converter.lower_and_strip(sentence)
    sentence = converter.remove_whitespaces_among_cjk(sentence)
    logger.success(sentence)

    # python -m models.sentencepiece.convert
