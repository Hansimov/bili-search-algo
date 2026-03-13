import unittest

from models.sentencepiece.vocab_filters import build_token_profile, is_malformed_token


class TokenFilterTests(unittest.TestCase):
    def test_malformed_ascii_patterns(self):
        self.assertTrue(is_malformed_token("bv123456"))
        self.assertTrue(is_malformed_token("aaabbbb"))
        self.assertTrue(is_malformed_token("abc__def"))
        self.assertTrue(is_malformed_token("-hello"))
        self.assertTrue(is_malformed_token("hello-"))

    def test_valid_tokens(self):
        self.assertFalse(is_malformed_token("youtube"))
        self.assertFalse(is_malformed_token("gta5"))
        self.assertFalse(is_malformed_token("王者荣耀"))

    def test_token_profile(self):
        profile = build_token_profile("王者youtube")
        self.assertTrue(profile.has_cjk)
        self.assertFalse(profile.is_ascii_token)
        self.assertEqual(profile.cjk_char_len, 2)


if __name__ == "__main__":
    unittest.main()
