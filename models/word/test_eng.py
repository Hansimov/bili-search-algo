"""
测试 RE_WORD 正则表达式
"""

import re

from tclogger import logger

from models.word.eng import REP_ENG


def test_re_word():
    # 测试用例
    test_cases = [
        # 应该匹配的情况
        ("Hello world", ["Hello world"]),  # 包含空格的应该视为一个单词
        ("test-case", ["test-case"]),  # 包含连字符
        ("API key", ["API key"]),  # 包含空格的词组
        ("3D model", ["3D model"]),  # 数字开头但包含字母，包含空格
        ("iOS app", ["iOS app"]),  # 数字在中间，包含空格
        ("HTTP 2.0", ["HTTP 2.0"]),  # 包含点号和空格
        ("user123", ["user123"]),  # 字母开头包含数字
        ("A", []),  # 单个字母
        ("x86-64", ["x86-64"]),  # 字母开头，包含数字和连字符
        ("version 1.2.3", ["version 1.2.3"]),  # 包含多个点号
        ("Node.js app", ["Node.js app"]),  # 包含点号和空格
        # 不应该匹配的情况
        ("123", []),  # 纯数字
        ("456.78", []),  # 数字开头无字母
        ("", []),  # 空字符串
        ("中文ABC", ["ABC"]),  # 中文后的英文
        ("test@email.com", ["test", "email.com"]),  # 邮箱格式，@分隔
        ("hello, world", ["hello", "world"]),  # 逗号分隔
        # 中英文夹杂测试用例
        ("我喜欢Python编程", ["Python"]),  # 中文中间的英文单词
        ("使用JavaScript和CSS", ["JavaScript", "CSS"]),  # 多个英文单词
        ("Node.js很好用", ["Node.js"]),  # 包含点号的英文
        ("iPhone 15 Pro很棒", ["iPhone 15 Pro"]),  # 包含空格和数字的英文
        ("学习AI和ML技术", ["AI", "ML"]),  # 缩写词
        ("GitHub上的project", ["GitHub", "project"]),  # 前后都有中文
        ("用React开发web应用", ["React", "web"]),  # 分散的英文单词
        ("版本v2.1.0发布了", ["v2.1.0"]),  # 版本号格式
        ("这是test-case测试", ["test-case"]),  # 包含连字符
        ("支持UTF-8编码", ["UTF-8"]),  # 数字开头包含字母的情况
        ("API接口很重要", ["API"]),  # 纯字母缩写
        ("用户user123登录", ["user123"]),  # 字母开头包含数字
        ("配置config.json文件", ["config.json"]),  # 文件名格式
        ("HTTP 2.0协议", ["HTTP 2.0"]),  # 包含空格和点号
        ("中文123数字", []),  # 纯数字不应匹配
        ("价格$99.99美元", []),  # 纯数字金额不应匹配
        # 边界情况测试
        ("123abc456", ["123abc456"]),  # 数字开头包含字母，前后有数字边界
        ("abc123def", ["abc123def"]),  # 字母开头，前后非字母数字
        ("开始a结束", []),  # 单个字母被中文包围
        ("test1test", ["test1test"]),  # 应该作为一个完整单词
        ("a1b2c3", ["a1b2c3"]),  # 字母数字交替
        ("中文A-B英文", ["A-B"]),  # 包含连字符的短单词
        ("前缀1a后缀", ["1a"]),  # 数字开头包含字母的最小情况
    ]

    logger.note("测试 RE_ENG 正则表达式:")

    for text, expected in test_cases:
        matches = REP_ENG.findall(text)
        is_matched = matches == expected
        status = "✓" if is_matched else "✗"
        if is_matched:
            logger.okay(f"{status} 输入: '{text}'")
        else:
            logger.warn(f"{status} 输入: '{text}'")
            logger.file(f"  实际: {matches}")
        logger.line(f"  期望: {expected}")


if __name__ == "__main__":
    test_re_word()

    # python -m models.word.test_eng
