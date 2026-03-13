import unittest

from argparse import Namespace

from models.sentencepiece.workflow import (
    build_input_prefix,
    build_word_commands,
    expand_targets,
)


class SentencePieceWorkflowTests(unittest.TestCase):
    def test_build_input_prefix(self):
        self.assertEqual(build_input_prefix("sp_908m"), "sp_908m_")
        self.assertEqual(build_input_prefix("sp_908m_"), "sp_908m_")

    def test_expand_targets(self):
        targets = expand_targets(["cine_movie", "recent"], include_wiki=False)
        self.assertEqual(
            targets,
            ["cine_movie", "recent"],
        )

    def test_expand_legacy_alias_targets(self):
        targets = expand_targets(["1", "r"], include_wiki=False)
        self.assertEqual(
            targets,
            ["cine_movie", "douga_anime", "tech_sports", "recent"],
        )

    def test_expand_targets_with_wiki(self):
        targets = expand_targets(["all"], include_wiki=True)
        self.assertIn("zhwiki", targets)
        self.assertIn("recent", targets)

    def test_build_word_commands(self):
        args = Namespace(
            skip_english=False,
            skip_chinese=False,
            word_min_freq=6,
            word_max_count=50000,
            word_log_dir=__import__("pathlib").Path("/tmp/word_logs"),
        )
        jobs = build_word_commands(args)
        self.assertEqual(len(jobs), 2)
        self.assertIn("-mn", jobs[0].command)
        self.assertIn("50000", jobs[0].command)


if __name__ == "__main__":
    unittest.main()
