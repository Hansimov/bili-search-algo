import argparse
import codecs
import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tclogger import brk, decolored, logger, logstr

from configs.envs import REPO_ROOT, SENTENCEPIECE_CKPT_ROOT
from models.sentencepiece.convert import (
    SentencePieceConverter,
    VocabsMerger,
    WordRecordsConverter,
)
from models.sentencepiece.merge import SentencePieceModelMerger
from models.word.eng import get_dump_path

DEFAULT_SP_PREFIX = "sp_908m"
DEFAULT_WIKI_PREFIX = "sp_wiki_8m_400k"
DEFAULT_WIKI_VOCAB_SIZE = 400000
DEFAULT_DOC_COUNT = 908000000
TRAIN_STATUS_DIR = SENTENCEPIECE_CKPT_ROOT / "status"
TRAIN_LOG_DIR = SENTENCEPIECE_CKPT_ROOT / "logs"
WORDS_LOG_DIR = TRAIN_LOG_DIR / "words"
LOG_SNAPSHOT_INTERVAL = 20.0
ANSI_LINE_RESET_RE = re.compile(r"(?:\x1b\[[0-9;?]*[EFGK]|\x1b\[[0-9;?]*2K)+")

VIDEO_REGIONS = [
    "cine_movie",
    "douga_anime",
    "tech_sports",
    "music_dance",
    "fashion_ent",
    "know_info",
    "daily_life",
    "other_life",
    "mobile_game",
    "other_game",
    "recent",
]
REGION_GROUPS = {
    **{region: [region] for region in VIDEO_REGIONS},
    "zhwiki": ["zhwiki"],
    "test": ["test"],
}
TARGET_ALIASES = {
    "all": VIDEO_REGIONS,
    "videos": VIDEO_REGIONS,
    "wiki": ["zhwiki"],
    "w": ["zhwiki"],
    "x": ["test"],
    "1": ["cine_movie", "douga_anime", "tech_sports"],
    "2": ["music_dance", "fashion_ent", "know_info"],
    "3": ["daily_life", "other_life"],
    "4": ["mobile_game", "other_game"],
    "r": ["recent"],
}


@dataclass
class TrainJob:
    name: str
    model_prefix: str
    command: list[str]
    status_path: Path
    log_path: Path
    process: subprocess.Popen | None = None
    log_writer: object = None
    log_thread: threading.Thread | None = None


@dataclass
class CommandJob:
    name: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen | None = None
    log_writer: object = None
    log_thread: threading.Thread | None = None


class SanitizedLogWriter:
    def __init__(
        self,
        log_path: Path,
        snapshot_interval: float = LOG_SNAPSHOT_INTERVAL,
        now_func=time.monotonic,
    ):
        self.log_path = log_path
        self.handle = open(log_path, "w", encoding="utf-8")
        self.current_line = ""
        self.snapshot_interval = snapshot_interval
        self.now_func = now_func
        self.last_snapshot_at = self.now_func()
        self.pending_escape = ""

    def _extract_incomplete_escape(self, text: str) -> tuple[str, str]:
        last_escape_idx = text.rfind("\x1b")
        if last_escape_idx < 0:
            return text, ""
        suffix = text[last_escape_idx:]
        if re.fullmatch(r"\x1b(?:\[[0-9;?]*[ -/]*)?", suffix):
            return text[:last_escape_idx], suffix
        return text, ""

    def _flush_line(self):
        self.handle.write(self.current_line.rstrip() + "\n")
        self.handle.flush()
        self.last_snapshot_at = self.now_func()
        self.current_line = ""

    def _flush_snapshot_if_due(self):
        if not self.current_line:
            return
        if self.snapshot_interval is None:
            return
        if self.now_func() - self.last_snapshot_at < self.snapshot_interval:
            return
        self.handle.write(self.current_line.rstrip() + "\n")
        self.handle.flush()
        self.last_snapshot_at = self.now_func()

    def write(self, text: str):
        if not text:
            return
        if self.pending_escape:
            text = self.pending_escape + text
            self.pending_escape = ""
        text, self.pending_escape = self._extract_incomplete_escape(text)
        text = ANSI_LINE_RESET_RE.sub("\r", text)
        text = decolored(text)
        if not text:
            return
        for char in text:
            if char == "\r":
                self.current_line = ""
            elif char == "\n":
                self._flush_line()
            else:
                self.current_line += char
        self._flush_snapshot_if_due()

    def close(self):
        if self.pending_escape:
            self.current_line += decolored(self.pending_escape)
            self.pending_escape = ""
        if self.current_line:
            self._flush_line()
        self.handle.close()


def start_logged_process(command: list[str], cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_writer = SanitizedLogWriter(log_path)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    def forward_output():
        stdout = process.stdout
        selector = selectors.DefaultSelector()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            if stdout is None:
                return
            selector.register(stdout, selectors.EVENT_READ)
            while True:
                events = selector.select(timeout=1.0)
                if not events:
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(stdout.fileno(), 4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    log_writer.write(text)
            tail = decoder.decode(b"", final=True)
            if tail:
                log_writer.write(tail)
        finally:
            selector.close()
            if stdout is not None:
                stdout.close()
            log_writer.close()

    log_thread = threading.Thread(target=forward_output, daemon=True)
    log_thread.start()
    return process, log_writer, log_thread


def finalize_logged_process(job: TrainJob | CommandJob):
    if job.log_thread:
        job.log_thread.join(timeout=5)
    if job.log_writer:
        job.log_writer = None
    job.log_thread = None


def build_input_prefix(prefix_base: str) -> str:
    return prefix_base if prefix_base.endswith("_") else f"{prefix_base}_"


def describe_exit_code(returncode: int) -> str:
    if returncode < 0:
        return f"signal {-returncode}"
    return f"exit code {returncode}"


def run_command_jobs(
    jobs: list[CommandJob],
    parallel: int,
    description: str,
    cwd: Path = REPO_ROOT,
) -> int:
    pending = list(jobs)
    active: list[CommandJob] = []
    parallel = max(1, parallel)

    while pending or active:
        while pending and len(active) < parallel:
            job = pending.pop(0)
            job.process, job.log_writer, job.log_thread = start_logged_process(
                command=job.command,
                cwd=cwd,
                log_path=job.log_path,
            )
            logger.note(
                f"> Started {description}: {logstr.mesg(job.name)} -> {logstr.file(job.log_path.name)}"
            )
            active.append(job)

        next_active = []
        for job in active:
            if job.process.poll() is None:
                next_active.append(job)
                continue
            finalize_logged_process(job)
            if job.process.returncode != 0:
                logger.warn(
                    f"× {description} failed: {job.name} ({describe_exit_code(job.process.returncode)})"
                )
                if job.log_path.exists():
                    tail = job.log_path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()[-20:]
                    if tail:
                        logger.file("\n".join(tail), indent=2)
                for active_job in next_active:
                    if active_job.process and active_job.process.poll() is None:
                        active_job.process.terminate()
                return job.process.returncode
        active = next_active

        if pending or active:
            time.sleep(1)

    return 0


def expand_targets(targets: list[str], include_wiki: bool = False) -> list[str]:
    expanded = []
    targets = targets or ["all"]
    for target in targets:
        if target in REGION_GROUPS:
            expanded.extend(REGION_GROUPS[target])
        elif target in TARGET_ALIASES:
            expanded.extend(TARGET_ALIASES[target])
        else:
            expanded.append(target)
    if include_wiki:
        expanded.append("zhwiki")

    deduped = []
    seen = set()
    for item in expanded:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_train_command(args: argparse.Namespace, region: str) -> tuple[str, list[str]]:
    if region == "zhwiki":
        model_prefix = args.wiki_prefix
        command = [
            sys.executable,
            "-u",
            "-m",
            "models.sentencepiece.train",
            "-m",
            model_prefix,
            "-db",
            "zhwiki",
            "-cn",
            "pages",
            "-bs",
            str(args.batch_size),
            "-vs",
            str(args.wiki_vocab_size),
        ]
    else:
        model_prefix = f"{args.prefix_base}_{region}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "models.sentencepiece.train",
            "-m",
            model_prefix,
            "-fg",
            region,
            "-bs",
            str(args.batch_size),
        ]

    if args.max_batch is not None:
        command.extend(["-mb", str(args.max_batch)])
    if args.character_coverage is not None:
        command.extend(["-cc", str(args.character_coverage)])
    if args.input_sentence_size is not None:
        command.extend(["-is", str(args.input_sentence_size)])
    if args.max_sentencepiece_length is not None:
        command.extend(["-ml", str(args.max_sentencepiece_length)])
    if args.num_threads is not None:
        command.extend(["-nt", str(args.num_threads)])
    if args.model_type:
        command.extend(["-mt", args.model_type])
    if args.shrinking_factor is not None:
        command.extend(["-sf", str(args.shrinking_factor)])
    if region != "zhwiki" and args.vocab_size is not None and not args.auto_vocab_size:
        command.extend(["-vs", str(args.vocab_size)])
    if args.auto_vocab_size and region != "zhwiki":
        command.append("-av")
    if args.keep_exist_model:
        command.append("-k")
    if args.force_delete:
        command.append("-fd")
    if args.estimate_count:
        command.append("-ec")
    if args.edit_model:
        command.append("-e")
    if args.split_alphanum:
        command.append("-sa")
    if args.split_by_number:
        command.append("-sn")
    if args.split_by_unicode_script:
        command.append("-su")

    return model_prefix, command


def build_word_commands(args: argparse.Namespace) -> list[CommandJob]:
    jobs = []
    word_workers = getattr(args, "word_workers", 10)
    if not args.skip_english:
        command = [
            sys.executable,
            "-u",
            "-m",
            "models.word.eng",
            "-ec",
            "-en",
            "-mf",
            str(args.word_min_freq),
            "-j",
            str(word_workers),
        ]
        if args.word_max_count is not None:
            command.extend(["-mn", str(args.word_max_count)])
        jobs.append(
            CommandJob(
                name="english",
                command=command,
                log_path=args.word_log_dir / "word_eng.log",
            )
        )
    if not args.skip_chinese:
        command = [
            sys.executable,
            "-u",
            "-m",
            "models.word.eng",
            "-ec",
            "-zh",
            "-mf",
            str(args.word_min_freq),
            "-j",
            str(word_workers),
        ]
        if args.word_max_count is not None:
            command.extend(["-mn", str(args.word_max_count)])
        jobs.append(
            CommandJob(
                name="chinese",
                command=command,
                log_path=args.word_log_dir / "word_zh.log",
            )
        )
    return jobs


class SentencePieceTrainOrchestrator:
    def __init__(self, jobs: list[TrainJob], parallel: int = 1, interval: float = 5.0):
        self.jobs = jobs
        self.parallel = max(1, parallel)
        self.interval = max(0.5, interval)

    def load_status(self, job: TrainJob) -> dict:
        if not job.status_path.exists():
            return {}
        try:
            return json.loads(job.status_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def start_job(self, job: TrainJob):
        job.status_path.parent.mkdir(parents=True, exist_ok=True)
        if job.status_path.exists():
            job.status_path.unlink()
        job.process, job.log_writer, job.log_thread = start_logged_process(
            command=job.command,
            cwd=REPO_ROOT,
            log_path=job.log_path,
        )
        logger.note(
            f"> Started: {logstr.mesg(job.name)} -> {logstr.file(job.model_prefix)}"
        )

    def format_status(self, job: TrainJob) -> str:
        status = self.load_status(job)
        stage = status.get("stage", "pending")
        total = status.get("total_samples")
        processed = status.get("samples_processed", 0)
        progress = status.get("progress")
        if progress is not None:
            progress_str = f"{progress * 100:6.2f}%"
        else:
            progress_str = "   n/a"
        total_str = f"{processed}/{total}" if total is not None else str(processed)
        state = "done"
        if job.process and job.process.poll() is None:
            state = "run"
        elif job.process and job.process.poll() not in [None, 0]:
            state = "fail"
        elif not job.process:
            state = "wait"
        return f"[{state:>4}] {job.name:<14} {stage:<10} {progress_str} {total_str:<18} {job.log_path.name}"

    def tail_log(self, job: TrainJob, lines: int = 20) -> str:
        if not job.log_path.exists():
            return ""
        content = job.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(content[-lines:])

    def stop_all(self):
        for job in self.jobs:
            if job.process and job.process.poll() is None:
                job.process.terminate()
        time.sleep(1)
        for job in self.jobs:
            if job.process and job.process.poll() is None:
                job.process.kill()
            finalize_logged_process(job)

    def run(self) -> int:
        pending = list(self.jobs)
        active: list[TrainJob] = []
        try:
            while pending or active:
                while pending and len(active) < self.parallel:
                    job = pending.pop(0)
                    self.start_job(job)
                    active.append(job)

                logger.note("> Train status snapshot:")
                for job in self.jobs:
                    logger.line(f"  * {self.format_status(job)}")

                next_active = []
                for job in active:
                    if job.process.poll() is None:
                        next_active.append(job)
                        continue
                    finalize_logged_process(job)
                    if job.process.returncode != 0:
                        logger.warn(
                            f"× Train job failed: {job.name} ({describe_exit_code(job.process.returncode)})"
                        )
                        tail = self.tail_log(job)
                        if tail:
                            logger.file(tail, indent=2)
                        self.stop_all()
                        return job.process.returncode
                active = next_active

                if pending or active:
                    time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.warn("! Interrupted, stopping all training jobs ...")
            self.stop_all()
            return 130

        return 0


def run_words(args: argparse.Namespace) -> int:
    jobs = build_word_commands(args)
    if not jobs:
        logger.warn("× No word extraction jobs requested")
        return 0
    if getattr(args, "dry_run", False):
        for job in jobs:
            logger.line(f"  * {' '.join(job.command)}")
        return 0
    return run_command_jobs(
        jobs=jobs,
        parallel=args.word_parallel,
        description="word job",
    )


def run_train(args: argparse.Namespace) -> int:
    targets = expand_targets(args.targets, include_wiki=args.include_wiki)
    jobs = []
    for region in targets:
        model_prefix, command = build_train_command(args, region)
        status_path = args.status_dir / f"{model_prefix}.json"
        log_path = args.log_dir / f"{model_prefix}.log"
        command.extend(["-sp", str(status_path), "-si", str(args.status_interval)])
        jobs.append(
            TrainJob(
                name=region,
                model_prefix=model_prefix,
                command=command,
                status_path=status_path,
                log_path=log_path,
            )
        )

    target_names = ", ".join(job.name for job in jobs)
    logger.note(f"> Training targets: {logstr.mesg(brk(target_names))}")
    if args.dry_run:
        for job in jobs:
            logger.line(f"  * {' '.join(job.command)}")
        return 0

    orchestrator = SentencePieceTrainOrchestrator(
        jobs=jobs,
        parallel=args.parallel,
        interval=args.monitor_interval,
    )
    return orchestrator.run()


def resolve_merge_model_paths(args: argparse.Namespace) -> list[Path]:
    input_model_prefixes = [args.wiki_prefix]
    input_model_prefixes.extend(
        [f"{build_input_prefix(args.input_prefix)}{region}" for region in VIDEO_REGIONS]
    )
    return [
        SENTENCEPIECE_CKPT_ROOT / f"{prefix}.model"
        for prefix in input_model_prefixes
        if (SENTENCEPIECE_CKPT_ROOT / f"{prefix}.model").exists()
    ]


def run_merge(args: argparse.Namespace) -> int:
    model_paths = resolve_merge_model_paths(args)
    if not model_paths:
        logger.warn("× No model paths found for merge")
        return 1
    merger = SentencePieceModelMerger(
        model_paths=model_paths,
        output_path=SENTENCEPIECE_CKPT_ROOT / f"{args.output_prefix}.model",
        max_vocab_size=args.merge_vocab_size,
        trunc_ratio=args.trunc_ratio,
        max_cjk_char_len=args.max_cjk_char_len,
        min_ascii_video_support=args.min_ascii_video_support,
        min_ascii_source_support=args.min_ascii_source_support,
        min_ascii_len=args.min_ascii_len,
        export_stats=args.export_stats,
        verbose=True,
    )
    merger.merge()
    return 0


def run_convert(args: argparse.Namespace) -> int:
    txt_paths = []
    model_path = SENTENCEPIECE_CKPT_ROOT / f"{args.output_prefix}.model"
    if args.convert_sentencepiece:
        sp_converter = SentencePieceConverter(model_path=model_path)
        sp_converter.to_txt()
        txt_paths.append(sp_converter.txt_path)

    if args.convert_record:
        word_converter = WordRecordsConverter(
            min_doc_freq=args.min_doc_freq,
            max_char_len=args.max_char_len,
        )
        for lang in ["en", "zh"]:
            csv_path = get_dump_path(args.doc_count, lang=lang)
            word_converter.set_csv_path(csv_path)
            word_converter.to_txt()
            txt_paths.append(word_converter.txt_path)

    if args.convert_merge:
        merger = VocabsMerger()
        merger.merge(txt_paths, save_path=args.save_path)

    return 0


def run_all(args: argparse.Namespace) -> int:
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    handlers = {
        "words": run_words,
        "train": run_train,
        "merge": run_merge,
        "convert": run_convert,
    }

    pending_steps = list(steps)
    run_words_and_train_together = (
        not args.serial_steps and "words" in pending_steps and "train" in pending_steps
    )

    if run_words_and_train_together:
        logger.note(
            "> Workflow step: run words and train in parallel to avoid idle CPU before merge"
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                "words": executor.submit(run_words, args),
                "train": executor.submit(run_train, args),
            }
            for step_name in ["words", "train"]:
                exit_code = futures[step_name].result()
                if exit_code != 0:
                    return exit_code
        pending_steps = [
            step for step in pending_steps if step not in {"words", "train"}
        ]

    for step in pending_steps:
        logger.note(f"> Workflow step: {logstr.success(step)}")
        exit_code = handlers[step](args)
        if exit_code != 0:
            return exit_code
    return 0


def add_words_args(parser: argparse.ArgumentParser):
    parser.add_argument("--word-min-freq", type=int, default=6)
    parser.add_argument("--skip-english", action="store_true")
    parser.add_argument("--skip-chinese", action="store_true")
    parser.add_argument("--word-max-count", type=int, default=None)
    parser.add_argument("--word-parallel", type=int, default=2)
    parser.add_argument("--word-workers", type=int, default=10)
    parser.add_argument("--word-log-dir", type=Path, default=WORDS_LOG_DIR)


def add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "targets",
        nargs="*",
        default=["all"],
        help="regions or aliases: all, videos, wiki, zhwiki, test; legacy 1/2/3/4/r/w/x still work",
    )
    parser.add_argument("-p", "--prefix-base", type=str, default=DEFAULT_SP_PREFIX)
    parser.add_argument("-wp", "--wiki-prefix", type=str, default=DEFAULT_WIKI_PREFIX)
    parser.add_argument(
        "-wv", "--wiki-vocab-size", type=int, default=DEFAULT_WIKI_VOCAB_SIZE
    )
    parser.add_argument("-iw", "--include-wiki", action="store_true")
    parser.add_argument("-j", "--parallel", type=int, default=1)
    parser.add_argument("-mi", "--monitor-interval", type=float, default=5.0)
    parser.add_argument("-sd", "--status-dir", type=Path, default=TRAIN_STATUS_DIR)
    parser.add_argument("-ld", "--log-dir", type=Path, default=TRAIN_LOG_DIR)
    parser.add_argument("-si", "--status-interval", type=int, default=2000)
    parser.add_argument("-dr", "--dry-run", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=1000)
    parser.add_argument("-mb", "--max-batch", type=int, default=None)
    parser.add_argument("-cc", "--character-coverage", type=float, default=0.9995)
    parser.add_argument("-is", "--input-sentence-size", type=int, default=500000)
    parser.add_argument("-ml", "--max-sentencepiece-length", type=int, default=16)
    parser.add_argument("-nt", "--num-threads", type=int, default=16)
    parser.add_argument("-mt", "--model-type", type=str, default="unigram")
    parser.add_argument("-sf", "--shrinking-factor", type=float, default=0.75)
    parser.add_argument("-vs", "--vocab-size", type=int, default=32000)
    parser.add_argument("-av", "--auto-vocab-size", action="store_true")
    parser.add_argument("-k", "--keep-exist-model", action="store_true")
    parser.add_argument("-fd", "--force-delete", action="store_true")
    parser.add_argument("-ec", "--estimate-count", action="store_true")
    parser.add_argument("-e", "--edit-model", action="store_true")
    parser.add_argument("-sa", "--split-alphanum", action="store_true")
    parser.add_argument("-sn", "--split-by-number", action="store_true")
    parser.add_argument("-su", "--split-by-unicode-script", action="store_true")


def add_merge_args(parser: argparse.ArgumentParser):
    parser.add_argument("-i", "--input-prefix", type=str, default=DEFAULT_SP_PREFIX)
    parser.add_argument("-o", "--output-prefix", type=str, default="sp_merged")
    parser.add_argument("-mvs", "--merge-vocab-size", type=int, default=1000000)
    parser.add_argument("-tr", "--trunc-ratio", type=float, default=0.9)
    parser.add_argument("-mc", "--max-cjk-char-len", type=int, default=8)
    parser.add_argument("-avs", "--min-ascii-video-support", type=int, default=1)
    parser.add_argument("-ass", "--min-ascii-source-support", type=int, default=2)
    parser.add_argument("-al", "--min-ascii-len", type=int, default=4)
    parser.add_argument("-es", "--export-stats", action="store_true")


def add_convert_args(parser: argparse.ArgumentParser):
    parser.add_argument("-cs", "--convert-sentencepiece", action="store_true")
    parser.add_argument("-cr", "--convert-record", action="store_true")
    parser.add_argument("-cm", "--convert-merge", action="store_true")
    parser.add_argument("-dc", "--doc-count", type=int, default=DEFAULT_DOC_COUNT)
    parser.add_argument("-n", "--min-doc-freq", type=int, default=20)
    parser.add_argument("-l", "--max-char-len", type=int, default=32)
    parser.add_argument("-sp", "--save-path", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage the SentencePiece vocabulary workflow: extract word vocabs, "
            "train regional models, merge models, convert outputs, or run the full pipeline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    words_parser = subparsers.add_parser(
        "words", help="extract English/Chinese word vocabs"
    )
    add_words_args(words_parser)

    train_parser = subparsers.add_parser(
        "train",
        help="train one or more regional SentencePiece models with optional parallel orchestration",
    )
    add_train_args(train_parser)

    merge_parser = subparsers.add_parser(
        "merge", help="merge trained SentencePiece models"
    )
    add_merge_args(merge_parser)

    convert_parser = subparsers.add_parser(
        "convert",
        help="convert merged SentencePiece outputs and optionally merge with word vocabs",
    )
    add_merge_args(convert_parser)
    add_convert_args(convert_parser)

    all_parser = subparsers.add_parser(
        "all",
        help="run multiple workflow steps in order; default steps are words,train,merge,convert",
    )
    all_parser.add_argument(
        "--steps",
        type=str,
        default="words,train,merge,convert",
        help="comma-separated subset of: words,train,merge,convert",
    )
    all_parser.add_argument(
        "--serial-steps",
        action="store_true",
        help="disable words/train overlap and run steps strictly in sequence",
    )
    add_words_args(all_parser)
    add_train_args(all_parser)
    add_merge_args(all_parser)
    add_convert_args(all_parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "words": run_words,
        "train": run_train,
        "merge": run_merge,
        "convert": run_convert,
        "all": run_all,
    }
    exit_code = handlers[args.command](args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
