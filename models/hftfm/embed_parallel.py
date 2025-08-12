import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Dict, Any, Literal, Generator
from dataclasses import dataclass

from tclogger import logger, logstr, dict_to_str, Runtimer, PathType
from models.hftfm.embed import HFTransformersEmbedder
from models.hftfm.tokenizers_fix import setup_worker_tokenizers_parallelism


@dataclass
class EmbedTask:
    """嵌入任务数据结构"""

    task_id: int
    sentences: Union[str, List[str]]
    batch_idx: int = 0


@dataclass
class EmbedResult:
    """嵌入结果数据结构"""

    task_id: int
    embeddings: np.ndarray
    batch_idx: int = 0


class HFTransformersEmbedderWorker:
    """独立的 embedder 工作进程"""

    def __init__(self, embedder_config: Dict[str, Any]):
        self.embedder_config = embedder_config
        self.embedder = None

    def initialize(self):
        """在工作进程中初始化 embedder"""
        if self.embedder is None:
            # 在工作进程中设置tokenizers并行
            setup_worker_tokenizers_parallelism(verbose=False)

            self.embedder = HFTransformersEmbedder(**self.embedder_config)
            self.embedder.load_model()

    def embed_batch(self, task: EmbedTask) -> EmbedResult:
        """执行单个嵌入任务"""
        if self.embedder is None:
            self.initialize()

        embeddings = self.embedder.embed(task.sentences)
        return EmbedResult(
            task_id=task.task_id, embeddings=embeddings, batch_idx=task.batch_idx
        )


class ParallelHFTransformersEmbedder:
    """支持多核并行的 HuggingFace Transformers Embedder"""

    def __init__(
        self,
        model_name: str = None,
        model_path: PathType = None,
        device: Literal["cuda", "cpu"] = "cpu",
        use_quantize: bool = True,
        use_layer_prune: bool = True,
        layer_prune_ratio: float = 0.5,
        use_attention_prune: bool = True,
        attention_prune_ratio: float = 0.5,
        model_kwargs: dict = None,
        verbose: bool = False,
        # 并行参数
        num_workers: int = None,
        batch_size: int = 32,
        max_queue_size: int = 100,
        parallel_mode: Literal["thread"] = "thread",
    ):
        # 基础参数
        self.embedder_config = {
            "model_name": model_name,
            "model_path": model_path,
            "device": device,
            "use_quantize": use_quantize,
            "use_layer_prune": use_layer_prune,
            "layer_prune_ratio": layer_prune_ratio,
            "use_attention_prune": use_attention_prune,
            "attention_prune_ratio": attention_prune_ratio,
            "model_kwargs": model_kwargs,
            "verbose": verbose,
        }

        # 并行参数
        self.num_workers = num_workers or min(os.cpu_count(), 4)
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.parallel_mode = parallel_mode
        self.verbose = verbose

        # 运行时变量
        self.workers = []
        self.input_queue = None
        self.output_queue = None
        self.is_initialized = False

        # 如果是 GPU 模式，调整工作进程数量
        if device == "cuda":
            self.num_workers = min(self.num_workers, 2)  # GPU 模式限制工作进程数量

    def log_init_info(self):
        """记录初始化信息"""
        if self.verbose:
            logger.note(f"> Initializing ParallelHFTransformersEmbedder:")

            info_dict = {
                "parallel_mode": self.parallel_mode,
                "num_workers": self.num_workers,
                "batch_size": self.batch_size,
                "device": self.embedder_config["device"],
            }

            logger.line(dict_to_str(info_dict), indent=2)

    def initialize(self):
        """初始化并行环境"""
        if self.is_initialized:
            return
        self.log_init_info()
        self.init_multithread()
        self.is_initialized = True

    def init_multithread(self):
        """初始化多线程模式"""
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.base_embedder = HFTransformersEmbedder(**self.embedder_config)
        self.base_embedder.load_model()

        if self.verbose:
            logger.mesg(f"  * Started {self.num_workers} worker threads")

    def split_to_batches(self, sentences: List[str]) -> List[List[str]]:
        """将句子列表分割成批次"""
        batches = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            batches.append(batch)
        return batches

    def embed_batch_multithread(self, sentences: List[str]) -> np.ndarray:
        """多线程模式的批量嵌入"""
        if not sentences:
            return np.array([])

        # 分批次
        batches = self.split_to_batches(sentences)

        # 提交任务到线程池
        futures = []
        for batch_idx, batch in enumerate(batches):
            future = self.executor.submit(self.base_embedder.embed, batch)
            futures.append((batch_idx, future))

        # 收集结果
        results = {}
        for batch_idx, future in futures:
            embeddings = future.result()
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            results[batch_idx] = embeddings

        # 按批次顺序合并结果
        all_embeddings = []
        for batch_idx in range(len(batches)):
            all_embeddings.append(results[batch_idx])

        return np.vstack(all_embeddings)

    def embed(
        self, sentences: Union[str, List[str]], show_progress: bool = False
    ) -> np.ndarray:
        """
        主要的嵌入方法

        Args:
            sentences: 要嵌入的句子（单个字符串或字符串列表）
            show_progress: 是否显示进度信息

        Returns:
            嵌入向量数组
        """
        if not self.is_initialized:
            self.initialize()

        if isinstance(sentences, str):
            sentences = [sentences]
            return_single = True
        else:
            return_single = False

        if not sentences:
            return np.array([])

        if show_progress and self.verbose:
            logger.note(
                f"> Embedding {len(sentences)} sentences in {len(self.split_to_batches(sentences))} batches"
            )

        embeddings = self.embed_batch_multithread(sentences)

        if return_single and embeddings.shape[0] == 1:
            return embeddings[0]

        return embeddings

    def embed_iter(
        self,
        sentences_iter: Union[List[str], Generator[str, None, None]],
        yield_batch_size: int = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        流式嵌入，适合处理大量数据

        Args:
            sentences_iter: 句子迭代器
            yield_batch_size: 每次返回的批次大小

        Yields:
            批次嵌入结果
        """
        if not self.is_initialized:
            self.initialize()

        yield_batch_size = yield_batch_size or self.batch_size

        current_batch = []
        for sentence in sentences_iter:
            current_batch.append(sentence)

            if len(current_batch) >= yield_batch_size:
                embeddings = self.embed(current_batch)
                yield embeddings
                current_batch = []

        # 处理剩余的句子
        if current_batch:
            embeddings = self.embed(current_batch)
            yield embeddings

    def terminate(self):
        """清理资源"""
        if not self.is_initialized:
            return

        self.executor.shutdown(wait=True)

        self.is_initialized = False

        if self.verbose:
            logger.mesg(f"  * All workers terminated")

    def __del__(self):
        """析构函数，确保资源清理"""
        self.terminate()

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.terminate()


# 兼容性包装器
class BatchHFTransformersEmbedder(HFTransformersEmbedder):
    """
    在原有 HFTransformersEmbedder 基础上增加批处理优化
    这个类不使用多进程，但优化了批处理逻辑
    """

    def __init__(self, *args, **kwargs):
        # 提取批处理参数
        self.optimal_batch_size = kwargs.pop("optimal_batch_size", 32)
        super().__init__(*args, **kwargs)

    def embed_large_batch(
        self, sentences: List[str], batch_size: int = None, show_progress: bool = False
    ) -> np.ndarray:
        """
        优化的大批量嵌入方法

        Args:
            sentences: 句子列表
            batch_size: 批次大小，如果为None则使用optimal_batch_size
            show_progress: 是否显示进度

        Returns:
            嵌入向量数组
        """
        if not sentences:
            return np.array([])

        batch_size = batch_size or self.optimal_batch_size

        if len(sentences) <= batch_size:
            return self.embed(sentences)

        if show_progress and self.verbose:
            logger.note(
                f"> Embedding {len(sentences)} sentences in batches of {batch_size}"
            )

        # 分批处理
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            batch_embeddings = self.embed(batch)

            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)

            all_embeddings.append(batch_embeddings)

            if show_progress and self.verbose:
                progress = min(i + batch_size, len(sentences))
                logger.mesg(f"  * Progress: {progress}/{len(sentences)}")

        return np.vstack(all_embeddings)


def test_parallel_embedder():
    """测试并行嵌入器"""
    logger.note("> Testing ParallelHFTransformersEmbedder")

    # 测试数据
    test_sentences = [
        "这是一个测试句子。",
        "人工智能是未来的发展方向。",
        "深度学习在自然语言处理中有重要应用。",
        "向量化表示是文本理解的基础。",
        "多核并行可以显著提高计算效率。",
    ] * 10  # 复制多份以测试批处理

    # 测试多线程模式
    logger.note(f"  * Testing multi-thread mode with {len(test_sentences)} sentences")
    with Runtimer():
        with ParallelHFTransformersEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu",
            use_quantize=True,
            num_workers=4,
            batch_size=8,
            parallel_mode="thread",
            verbose=False,
        ) as embedder:
            embeddings = embedder.embed(test_sentences, show_progress=True)
            logger.mesg(f"  * Result shape: {embeddings.shape}")


def test_batch_embedder():
    """测试批处理嵌入器"""
    logger.note("> Testing BatchHFTransformersEmbedder")

    test_sentences = [
        "这是一个测试句子。",
        "人工智能是未来的发展方向。",
        "深度学习在自然语言处理中有重要应用。",
    ] * 20

    with Runtimer():
        embedder = BatchHFTransformersEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu",
            use_quantize=True,
            optimal_batch_size=16,
            verbose=True,
        )
        embedder.load_model()

        embeddings = embedder.embed_large_batch(
            test_sentences, batch_size=8, show_progress=True
        )
        logger.mesg(f"  * Result shape: {embeddings.shape}")


if __name__ == "__main__":
    # 测试所有实现
    test_batch_embedder()
    test_parallel_embedder()

    # python -m models.hftfm.embed_parallel
