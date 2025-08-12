"""
HuggingFace Tokenizers Fork Warning 修复方案

解决问题：
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. 
Disabling parallelism to avoid deadlocks...

这个警告出现的原因：
1. HuggingFace tokenizers 使用内部并行机制
2. 当已经使用了tokenizers后再进行进程fork，会出现死锁风险
3. tokenizers库为了安全会禁用内部并行

解决方案：
1. 在fork前禁用tokenizers并行
2. 在工作进程中重新设置tokenizers并行
3. 使用环境变量控制tokenizers行为
"""

import os
import multiprocessing as mp
from contextlib import contextmanager


class TokenizersParallelismManager:
    """管理 HuggingFace Tokenizers 并行设置的类"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.original_env_value = None

    def disable_tokenizers_parallelism(self):
        """禁用 tokenizers 并行"""
        self.original_env_value = os.environ.get("TOKENIZERS_PARALLELISM")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.verbose:
            print("  * Disabled TOKENIZERS_PARALLELISM to prevent fork warnings")

    def enable_tokenizers_parallelism(self):
        """启用 tokenizers 并行"""
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        if self.verbose:
            print("  * Enabled TOKENIZERS_PARALLELISM for worker process")

    def restore_tokenizers_parallelism(self):
        """恢复原始的 tokenizers 并行设置"""
        if self.original_env_value is None:
            # 如果原来没有设置，则删除环境变量
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            # 恢复原始值
            os.environ["TOKENIZERS_PARALLELISM"] = self.original_env_value

        if self.verbose:
            print("  * Restored original TOKENIZERS_PARALLELISM setting")


@contextmanager
def safe_multiprocessing_context(verbose: bool = False):
    """
    安全的多进程上下文管理器
    在进入多进程环境前禁用tokenizers并行，退出时恢复
    """
    manager = TokenizersParallelismManager(verbose=verbose)

    try:
        # 进入时禁用tokenizers并行
        manager.disable_tokenizers_parallelism()
        yield manager
    finally:
        # 退出时恢复原始设置
        manager.restore_tokenizers_parallelism()


def setup_worker_tokenizers_parallelism(verbose: bool = False):
    """
    在工作进程中设置tokenizers并行
    这个函数应该在每个工作进程的初始化阶段调用
    """
    # 在worker中可以安全地启用tokenizers并行
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if verbose:
        print(f"  * Worker {mp.current_process().name}: Enabled TOKENIZERS_PARALLELISM")


def get_safe_multiprocessing_start_method() -> str:
    """
    获取安全的多进程启动方法

    Returns:
        str: 推荐的启动方法
    """
    system = os.name

    if system == "posix":  # Linux/Unix
        # 在Linux上使用 'spawn' 或 'forkserver' 更安全
        # spawn会创建完全新的进程，避免fork问题
        available_methods = mp.get_all_start_methods()

        if "spawn" in available_methods:
            return "spawn"
        elif "forkserver" in available_methods:
            return "forkserver"
        else:
            return "fork"
    else:  # Windows
        return "spawn"  # Windows默认使用spawn


def configure_multiprocessing_for_tokenizers(verbose: bool = False):
    """
    为tokenizers配置多进程环境

    Args:
        verbose: 是否显示详细信息
    """
    # 获取推荐的启动方法
    recommended_method = get_safe_multiprocessing_start_method()
    current_method = mp.get_start_method(allow_none=True)

    if current_method != recommended_method:
        try:
            mp.set_start_method(recommended_method, force=True)
            if verbose:
                print(
                    f"  * Changed multiprocessing start method: {current_method} -> {recommended_method}"
                )
        except RuntimeError as e:
            if verbose:
                print(f"  * Warning: Could not change start method: {e}")
                print(f"  * Using current method: {current_method}")
    else:
        if verbose:
            print(f"  * Using multiprocessing start method: {current_method}")


# 简化的API函数
def disable_tokenizers_parallelism():
    """简单的函数来禁用tokenizers并行"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def enable_tokenizers_parallelism():
    """简单的函数来启用tokenizers并行"""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


# 装饰器版本
def with_safe_tokenizers(func):
    """
    装饰器：在函数执行前后管理tokenizers并行设置
    """

    def wrapper(*args, **kwargs):
        with safe_multiprocessing_context():
            return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    # 测试代码
    import sys

    print("=== HuggingFace Tokenizers 并行管理测试 ===")

    # 测试环境变量管理
    print("\n1. 测试环境变量管理:")
    manager = TokenizersParallelismManager(verbose=True)

    print(
        f"   初始 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'None')}"
    )

    manager.disable_tokenizers_parallelism()
    print(
        f"   禁用后 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}"
    )

    manager.enable_tokenizers_parallelism()
    print(
        f"   启用后 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}"
    )

    manager.restore_tokenizers_parallelism()
    print(
        f"   恢复后 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'None')}"
    )

    # 测试上下文管理器
    print("\n2. 测试上下文管理器:")
    print(
        f"   进入前 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'None')}"
    )

    with safe_multiprocessing_context(verbose=True) as ctx:
        print(
            f"   上下文中 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}"
        )

    print(
        f"   退出后 TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'None')}"
    )

    # 测试多进程配置
    print("\n3. 测试多进程配置:")
    configure_multiprocessing_for_tokenizers(verbose=True)

    print("\n测试完成!")

    # python -m models.hftfm.tokenizers_fix
