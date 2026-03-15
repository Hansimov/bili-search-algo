import unittest

from models.sentencepiece.vocab_filters import build_token_profile, is_malformed_token


class TokenFilterTests(unittest.TestCase):
    def test_malformed_ascii_patterns(self):
        self.assertTrue(is_malformed_token("bv123456"))
        self.assertTrue(is_malformed_token("aaabbbb"))
        self.assertTrue(is_malformed_token("abc__def"))
        self.assertTrue(is_malformed_token("-hello"))
        self.assertTrue(is_malformed_token("hello-"))

    def test_malformed_separator_and_template_noise(self):
        self.assertTrue(is_malformed_token("生活，记录"))
        self.assertTrue(is_malformed_token("fps：瓦"))
        self.assertTrue(is_malformed_token("外部链接"))
        self.assertTrue(is_malformed_token("sourcefrom"))
        self.assertTrue(is_malformed_token("关注我"))
        self.assertTrue(is_malformed_token("请关注我"))
        self.assertTrue(is_malformed_token("一键三连支持一下"))
        self.assertTrue(is_malformed_token("点赞收藏"))
        self.assertTrue(is_malformed_token("点个关注"))
        self.assertTrue(is_malformed_token("请私信我"))
        self.assertTrue(is_malformed_token("直播间传送门"))
        self.assertTrue(is_malformed_token("虎牙直播间"))
        self.assertTrue(is_malformed_token("仅搬运"))
        self.assertTrue(is_malformed_token("搬运自"))
        self.assertTrue(is_malformed_token("如侵删"))
        self.assertFalse(is_malformed_token("原神，启动"))
        self.assertFalse(is_malformed_token("黑神话：悟空"))

    def test_valid_tokens(self):
        self.assertFalse(is_malformed_token("youtube"))
        self.assertFalse(is_malformed_token("gta5"))
        self.assertFalse(is_malformed_token("王者荣耀"))
        self.assertFalse(is_malformed_token("收藏家"))
        self.assertFalse(is_malformed_token("收藏夹"))
        self.assertFalse(is_malformed_token("投稿者"))
        self.assertFalse(is_malformed_token("搬运工"))
        self.assertFalse(is_malformed_token("直播间"))

    def test_token_profile(self):
        profile = build_token_profile("王者youtube")
        self.assertTrue(profile.has_cjk)
        self.assertFalse(profile.is_ascii_token)
        self.assertEqual(profile.cjk_char_len, 2)


if __name__ == "__main__":
    unittest.main()
