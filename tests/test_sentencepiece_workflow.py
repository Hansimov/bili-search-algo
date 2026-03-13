import unittest

from models.sentencepiece.workflow import build_input_prefix, expand_targets


class SentencePieceWorkflowTests(unittest.TestCase):
    def test_build_input_prefix(self):
        self.assertEqual(build_input_prefix("sp_908m"), "sp_908m_")
        self.assertEqual(build_input_prefix("sp_908m_"), "sp_908m_")

    def test_expand_targets(self):
        targets = expand_targets(["1", "r"], include_wiki=False)
        self.assertEqual(
            targets,
            ["cine_movie", "douga_anime", "tech_sports", "recent"],
        )

    def test_expand_targets_with_wiki(self):
        targets = expand_targets(["all"], include_wiki=True)
        self.assertIn("zhwiki", targets)
        self.assertIn("recent", targets)


if __name__ == "__main__":
    unittest.main()
