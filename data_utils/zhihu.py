from huggingface_hub import snapshot_download
from pathlib import Path
from tclogger import logger

DATA_ROOT = Path(__file__).parents[1] / "data"


class ZhihuDataDownloader:
    def __init__(self):
        self.download_path = DATA_ROOT / "zhihu"
        self.repo_id = "wangrui6/Zhihu-KOL"
        self.allow_patterns = ["*.parquet"]
        if not self.download_path.exists():
            self.download_path.mkdir(parents=True)

    def download(self):
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            allow_patterns=self.allow_patterns,
            local_dir=self.download_path,
            cache_dir=DATA_ROOT,
        )

    def move(self):
        logger.note("> Move files to:")
        logger.file(f"  * {self.download_path}")
        for file in self.download_path.rglob("*.parquet"):
            file.rename(self.download_path / file.name)


if __name__ == "__main__":
    downloader = ZhihuDataDownloader()
    downloader.download()
    downloader.move()

    # python -m data_utils.zhihu
