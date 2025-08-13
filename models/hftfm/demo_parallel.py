"""
实际使用示例：展示如何使用并行化的 embedding 来提高吞吐率

这个脚本演示了几种不同的使用场景：
1. 批量处理大量文本
2. 流式处理数据
3. 性能对比测试


> Performance Summary: (10000 samples)
  * baseline           :  37.02s (1.00x)
  * batch_16           :  33.39s (1.11x)
  * batch_32           :  35.90s (1.03x)
  * batch_48           :  38.50s (0.96x)
  * multi_thread_16x16 :  23.01s (1.61x)
  * multi_thread_16x32 :  21.50s (1.72x)
  * multi_thread_32x32 :  18.52s (2.00x)
  * multi_thread_32x48 :  26.79s (1.38x)
  * multi_thread_48x48 :  31.48s (1.18x)
"""

import time
import numpy as np

from typing import List, Generator
from tclogger import logger, logstr, Runtimer

from models.hftfm.embed import HFTransformersEmbedder
from models.hftfm.embed_parallel import (
    ParallelHFTransformersEmbedder,
    BatchHFTransformersEmbedder,
)


def generate_test_sentences(count: int = 1000) -> List[str]:
    """生成测试句子"""
    base_sentences = [
        "人工智能是未来科技发展的重要方向，深度学习技术不断突破传统的认知边界。",
        "自然语言处理技术在搜索引擎、智能客服、机器翻译等领域有着广泛的应用前景。",
        "向量化表示是现代文本理解和语义匹配的核心技术，为信息检索提供了强有力的支撑。",
        "多核并行计算可以显著提高大规模文本处理的效率，是优化系统性能的重要手段。",
        "嵌入模型的质量直接影响下游任务的效果，选择合适的预训练模型至关重要。",
        "批处理和并行处理是处理大规模数据的两种重要优化策略，可以根据具体场景选择。",
        "GPU加速和CPU多核处理各有优势，需要根据模型大小和数据量进行合理配置。",
        "文本检索系统的性能瓶颈往往在于特征提取阶段，优化这一环节能带来显著提升。",
    ]

    sentences = []
    for i in range(count):
        # 添加一些变化以模拟真实数据
        base_sentence = base_sentences[i % len(base_sentences)]
        sentences.append(f"[{i+1:04d}] {base_sentence}")

    return sentences


def log_timer_embeddings(timer: Runtimer, embeddings: np.ndarray):
    logger.okay(f"  ✓ Time: {timer.elapsed_seconds:.2f}s,", end=" ")
    logger.mesg(f"Shape: {embeddings.shape}")


def benchmark():
    """性能对比测试：单线程 vs 并行"""
    logger.note("> Benchmarking ...")

    # 测试数据
    test_sentences = generate_test_sentences(10000)  # 样本数
    logger.mesg(f"  * Test sentences: {len(test_sentences)}")

    results = {}

    # 1. 原始单线程版本
    logger.hint("> Testing baseline version:")
    with Runtimer(False) as timer:
        embedder = HFTransformersEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu",
            use_quantize=True,
            verbose=False,
        )
        embedder.load_model()
        embeddings = embedder.embed(test_sentences)

    results["baseline"] = {
        "time": timer.elapsed_seconds,
        "shape": embeddings.shape,
    }
    log_timer_embeddings(timer, embeddings)

    # 2. 批处理优化版本
    for batch_size in [16, 32, 48]:
        logger.hint(f"> Testing batch version: ({batch_size})")
        with Runtimer(False) as timer:
            batch_embedder = BatchHFTransformersEmbedder(
                model_name="BAAI/bge-base-zh-v1.5",
                device="cpu",
                use_quantize=True,
                optimal_batch_size=batch_size,
                verbose=False,
            )
            batch_embedder.load_model()
            embeddings = batch_embedder.embed_large_batch(
                test_sentences, batch_size=batch_size
            )

        results[f"batch_{batch_size}"] = {
            "time": timer.elapsed_seconds,
            "shape": embeddings.shape,
        }
        log_timer_embeddings(timer, embeddings)

    # 3. 多线程版本
    for num_workers, batch_size in [(16, 16), (16, 32), (32, 32), (32, 48), (48, 48)]:
        stat_str = f"{num_workers}x{batch_size}"
        logger.hint(f"> Testing multi-thread version: ({stat_str})")
        with Runtimer(False) as timer:
            with ParallelHFTransformersEmbedder(
                model_name="BAAI/bge-base-zh-v1.5",
                device="cpu",
                use_quantize=True,
                num_workers=num_workers,
                batch_size=batch_size,
                parallel_mode="thread",
                verbose=False,
            ) as parallel_embedder:
                embeddings = parallel_embedder.embed(test_sentences)

        results[f"multi_thread_{stat_str}"] = {
            "time": timer.elapsed_seconds,
            "shape": embeddings.shape,
        }
        log_timer_embeddings(timer, embeddings)

    # 计算加速比
    logger.note("> Performance Summary:")
    baseline_time = results["baseline"]["time"]
    for method, result in results.items():
        speedup = baseline_time / result["time"]
        speedup_str = f"{speedup:.2f}x"
        if abs(speedup - 1) < 0.05:
            speedup_str = logstr.file(speedup_str)
        elif speedup > 1:
            speedup_str = logstr.okay(f"{speedup:.2f}x")
        else:
            speedup_str = logstr.warn(f"{speedup:.2f}x")
        logger.mesg(f"  * {method:18} : {result['time']:6.2f}s ({speedup_str})")


def demo_streaming_processing():
    """演示流式处理大规模数据"""
    logger.note("> Demo: Streaming Processing")

    def sentence_generator(count: int) -> Generator[str, None, None]:
        """模拟大规模数据流"""
        base_sentences = generate_test_sentences(10)
        for i in range(count):
            yield base_sentences[i % len(base_sentences)]

    # 模拟处理 1000 条数据的流
    sentence_stream = sentence_generator(1000)

    with Runtimer():
        with ParallelHFTransformersEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu",
            use_quantize=True,
            num_workers=4,
            batch_size=32,
            parallel_mode="thread",
            verbose=True,
        ) as embedder:

            total_processed = 0
            for batch_embeddings in embedder.embed_iter(
                sentence_stream, yield_batch_size=64
            ):
                total_processed += batch_embeddings.shape[0]
                logger.mesg(
                    f"  * Processed batch: {batch_embeddings.shape[0]} sentences, Total: {total_processed}"
                )

                # 这里可以进行后续处理，比如存储到数据库
                # process_embeddings(batch_embeddings)

    logger.success(f"  * Total processed: {total_processed} sentences")


def demo_real_world_usage():
    """演示真实世界的使用场景"""
    logger.note("> Demo: Real-world Usage Scenarios")

    # 场景1：批量处理B站视频标题和描述
    logger.note("  * Scenario 1: Batch processing video titles and descriptions")

    # 模拟B站视频数据
    video_data = [
        {
            "bvid": f"BV{i:010d}",
            "title": f"视频标题 {i}",
            "desc": f"这是第{i}个视频的描述，包含了丰富的内容信息。",
        }
        for i in range(1, 101)
    ]

    # 提取所有文本
    all_texts = []
    for video in video_data:
        all_texts.append(video["title"])
        all_texts.append(video["desc"])

    logger.mesg(
        f"    * Processing {len(all_texts)} texts from {len(video_data)} videos"
    )

    with Runtimer():
        with ParallelHFTransformersEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu",
            use_quantize=True,
            num_workers=4,
            batch_size=16,
            parallel_mode="thread",
            verbose=True,
        ) as embedder:
            embeddings = embedder.embed(all_texts, show_progress=True)

    logger.success(f"    * Generated embeddings: {embeddings.shape}")

    # 场景2：实时处理新增数据
    logger.note("  * Scenario 2: Real-time processing of new data")

    # 模拟新增数据流
    def new_data_stream():
        """模拟实时新增的数据"""
        for i in range(50):
            yield f"新增数据 {i+1}：这是实时产生的内容，需要及时进行向量化处理。"
            time.sleep(0.01)  # 模拟数据产生间隔

    processed_count = 0
    with ParallelHFTransformersEmbedder(
        model_name="BAAI/bge-base-zh-v1.5",
        device="cpu",
        use_quantize=True,
        num_workers=2,
        batch_size=8,
        parallel_mode="thread",
        verbose=True,
    ) as embedder:
        current_batch = []
        for new_text in new_data_stream():
            current_batch.append(new_text)

            # 当批次大小达到阈值时进行处理
            if len(current_batch) >= 10:
                with Runtimer() as timer:
                    batch_embeddings = embedder.embed(current_batch)
                    processed_count += len(current_batch)
                    logger.mesg(
                        f"    * Processed batch of {len(current_batch)} in {timer.elapsed_seconds:.3f}s"
                    )
                    current_batch = []

        # 处理剩余数据
        if current_batch:
            batch_embeddings = embedder.embed(current_batch)
            processed_count += len(current_batch)
            logger.mesg(f"    * Processed final batch of {len(current_batch)}")

    logger.success(f"    * Total processed in real-time: {processed_count} items")


def demo_memory_efficient_processing():
    """演示内存高效的处理方式"""
    logger.note("> Demo: Memory-efficient Processing")

    # 模拟大规模数据（这里只是示例，实际可能是从数据库或文件读取）
    total_count = 5000
    logger.mesg(f"  * Simulating processing of {total_count} sentences")

    def large_data_generator():
        """大数据生成器，模拟从数据库或文件逐步读取"""
        base_sentences = generate_test_sentences(20)
        for i in range(total_count):
            yield f"[{i+1:05d}] {base_sentences[i % len(base_sentences)]}"

    # 使用流式处理，避免一次性加载所有数据到内存
    processed_count = 0
    total_embedding_size = 0

    with Runtimer():
        with ParallelHFTransformersEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            device="cpu",
            use_quantize=True,
            num_workers=4,
            batch_size=32,
            parallel_mode="thread",
            verbose=True,
        ) as embedder:

            for batch_embeddings in embedder.embed_iter(
                large_data_generator(), yield_batch_size=64
            ):
                processed_count += batch_embeddings.shape[0]
                total_embedding_size += batch_embeddings.nbytes

                # 这里可以进行批次处理，如保存到数据库、计算相似度等
                # save_embeddings_to_database(batch_embeddings)

                if processed_count % 500 == 0:
                    logger.mesg(
                        f"    * Progress: {processed_count}/{total_count} sentences"
                    )

    logger.success(f"  * Completed: {processed_count} sentences")
    logger.mesg(
        f"  * Total embedding size: {total_embedding_size / 1024 / 1024:.1f} MB"
    )


def main():
    """主函数：运行所有演示"""
    logger.note("=" * 60)
    logger.note("HFTransformers Parallel Embedding Demo")
    logger.note("=" * 60)

    try:
        # 1. 性能对比测试
        benchmark()
        print()

        # # 2. 流式处理演示
        # demo_streaming_processing()
        # print()

        # # 3. 真实场景演示
        # demo_real_world_usage()
        # print()

        # # 4. 内存高效处理演示
        # demo_memory_efficient_processing()
        # print()

        logger.success("✓ All demos completed successfully!")

    except KeyboardInterrupt:
        logger.warn("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

    # python -m models.hftfm.demo_parallel
