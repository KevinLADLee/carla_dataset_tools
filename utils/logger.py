#!/usr/bin/env python3
"""
CARLA Dataset Tools - 统一日志配置
提供跨模块的结构化日志记录，支持彩色终端输出和文件日志
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class LogColors:
    """终端日志颜色代码"""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅用于终端）"""

    COLORS = {
        'DEBUG': LogColors.CYAN,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.MAGENTA,
    }

    def format(self, record):
        # 保存原始levelname
        original_levelname = record.levelname

        # 添加颜色（仅当输出到终端时）
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"
                )

        # 格式化消息
        result = super().format(record)

        # 恢复原始levelname（避免影响其他handler）
        record.levelname = original_levelname

        return result


def setup_logger(
    name: str = "carla_dataset_tools",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    配置并返回logger实例

    Args:
        name: Logger名称（通常使用 __name__）
        log_file: 日志文件路径（可选）
        level: 日志级别（默认INFO）
        console_output: 是否输出到控制台（默认True）

    Returns:
        配置好的logger实例

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("开始录制数据")
        >>> logger.error("传感器 camera_01 失败")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler（带颜色）
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # 文件handler（无颜色，包含更多信息）
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件中记录所有级别
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# 全局默认logger
_default_logger = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例（便捷函数）

    Args:
        name: Logger名称，如果为None则使用全局logger

    Returns:
        Logger实例

    Example:
        >>> # 在模块中使用
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条信息日志")
        >>>
        >>> # 使用全局logger
        >>> logger = get_logger()
        >>> logger.warning("这是一条警告日志")
    """
    global _default_logger

    if name:
        return logging.getLogger(name)

    if _default_logger is None:
        _default_logger = setup_logger()

    return _default_logger


def configure_global_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    Configure global logging settings for all loggers in the application

    This function configures Python's root logger, ensuring all child loggers
    (created with logging.getLogger(__name__)) inherit the configuration.
    Should be called once at application startup.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Optional log file path

    Example:
        >>> from utils.logger import configure_global_logging
        >>> import logging
        >>>
        >>> # Configure DEBUG level with file output
        >>> configure_global_logging(
        ...     level=logging.DEBUG,
        ...     log_file='logs/recording.log'
        ... )
    """
    global _default_logger

    # Configure the root logger - all child loggers will inherit this configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Also update the default logger reference for backward compatibility
    _default_logger = root_logger


# 便捷函数，用于快速日志记录
def debug(msg: str):
    """快速记录DEBUG日志"""
    get_logger().debug(msg)


def info(msg: str):
    """快速记录INFO日志"""
    get_logger().info(msg)


def warning(msg: str):
    """快速记录WARNING日志"""
    get_logger().warning(msg)


def error(msg: str):
    """快速记录ERROR日志"""
    get_logger().error(msg)


def critical(msg: str):
    """快速记录CRITICAL日志"""
    get_logger().critical(msg)
