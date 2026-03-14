import unittest

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from models.sentencepiece.workflow import (
    SanitizedLogWriter,
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

    def test_sanitized_log_writer_collapses_progress_updates(self):
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "word.log"
            writer = SanitizedLogWriter(log_path)
            writer.write(
                "\033[95mstart\033[0m\n"
                "\033[1G\033[2K\033[96m* Doc:\033[0m 10%\r"
                "\033[1G\033[2K\033[96m* Doc:\033[0m 20%\r"
                "\033[1G\033[2K\033[96m* Doc:\033[0m 30%"
            )
            writer.close()

            self.assertEqual(
                log_path.read_text(encoding="utf-8"), "start\n* Doc: 30%\n"
            )


if __name__ == "__main__":
    unittest.main()
