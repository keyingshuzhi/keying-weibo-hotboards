# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午3:23
# @Author: 柯影数智
# @File: logutil.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm


"""
weibo_hotsearch.logutil — 运维友好型日志模块

特性
- 控制台 + 滚动文件双通道；ERROR 级别单独文件
- JSON/纯文本两种格式（可通过 .env 切换）
- 自动创建日志目录；与包内 CONFIG 集成
- 提供便捷方法：
  * setup_logging() 一次化配置
  * get_logger(name) 获取命名 logger
  * audit_config() 记录脱敏后的关键配置
  * install_excepthook() 捕获未处理异常写日志
  * log_duration() 代码片段耗时上下文
环境变量（写在 .env）
- LOG_DIR=logs
- LOG_LEVEL=INFO  （DEBUG/INFO/WARNING/ERROR）
- LOG_TO_CONSOLE=1
- LOG_JSON=0
- LOG_ROTATE_BYTES=10485760    # 10MB
- LOG_BACKUP_COUNT=10
- LOG_NAME=weibo_hotsearch
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo

# 避免与 stdlib logging 同名，模块名使用 logutil
from .config import CONFIG


# ------------------------
# 环境读取（带默认）
# ------------------------
def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if (v is not None and v.strip() != "") else default


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    return v.strip() not in ("0", "false", "False", "no", "NO")


# ------------------------
# 格式化器
# ------------------------
class _JSONFormatter(logging.Formatter):
    def __init__(self, tz: ZoneInfo):
        super().__init__()
        self.tz = tz

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, self.tz).isoformat(timespec="seconds")
        payload: Dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": record.process,
            "tid": record.thread,
            "file": record.pathname,
            "line": record.lineno,
            "func": record.funcName,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def __init__(self, tz: ZoneInfo):
        super().__init__()
        self.tz = tz
        # 例：2025-11-07 15:02:31+08:00 | INFO  | weibo_hotsearch.fetch:123 | message
        self._fmt = "%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s"

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        # 带时区的紧凑格式
        return dt.isoformat(timespec="seconds")

    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(self._fmt)
        formatter.formatTime = self.formatTime  # 绑定本地化时间
        return formatter.format(record)


# ------------------------
# 初始化与获取 logger
# ------------------------
_initialized = False
_root_logger: Optional[Logger] = None


def setup_logging(
        *,
        log_dir: Optional[str] = None,
        level: Optional[str] = None,
        to_console: Optional[bool] = None,
        use_json: Optional[bool] = None,
        rotate_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
        app_name: Optional[str] = None,
) -> Logger:
    """
    初始化日志系统（幂等）。可显式传参或走 .env 默认。
    返回根 logger（可直接 logger.info(...) 使用）
    """
    global _initialized, _root_logger
    if _initialized and _root_logger:
        return _root_logger

    tz = CONFIG.tzinfo()
    log_dir = log_dir or _env_str("LOG_DIR", "logs")
    level = (level or _env_str("LOG_LEVEL", "INFO")).upper()
    to_console = _env_bool("LOG_TO_CONSOLE", True) if to_console is None else to_console
    use_json = _env_bool("LOG_JSON", False) if use_json is None else use_json
    rotate_bytes = rotate_bytes or _env_int("LOG_ROTATE_BYTES", 10 * 1024 * 1024)  # 10MB
    backup_count = backup_count or _env_int("LOG_BACKUP_COUNT", 10)
    app_name = app_name or _env_str("LOG_NAME", "weibo_hotsearch")

    # 目录
    p = Path(log_dir)
    if not p.is_absolute():
        # 基于项目根目录创建
        p = Path(__file__).resolve().parents[1] / p
    p.mkdir(parents=True, exist_ok=True)

    # 根 logger
    root = logging.getLogger(app_name)
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False  # 阻止向上冒泡重复打印

    # 清理旧 handler（避免重复添加）
    for h in list(root.handlers):
        root.removeHandler(h)

    # 格式化器
    fmt = _JSONFormatter(tz) if use_json else _TextFormatter(tz)

    # 文件 handler（全部日志）
    file_path = p / f"{app_name}.log"
    fh = RotatingFileHandler(file_path, maxBytes=rotate_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setLevel(root.level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 错误单独文件
    err_path = p / f"{app_name}.error.log"
    eh = RotatingFileHandler(err_path, maxBytes=rotate_bytes, backupCount=backup_count, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    root.addHandler(eh)

    # 控制台
    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(root.level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    _initialized = True
    _root_logger = root
    return root


def get_logger(name: Optional[str] = None) -> Logger:
    """
    获取命名 logger（会确保已初始化）。
    """
    if not _initialized or _root_logger is None:
        setup_logging()
    # 使用根 logger 名字空间的子 logger
    base = _root_logger.name if _root_logger else "weibo_hotsearch"
    return logging.getLogger(f"{base}.{name}" if name else base)


# ------------------------
# 运维辅助：配置审计/异常钩子/计时
# ------------------------
def audit_config(logger: Optional[Logger] = None) -> None:
    """
    记录一次脱敏配置，便于排障。
    """
    lg = logger or get_logger(__name__)
    lg.info("CONFIG snapshot: %s", json.dumps(CONFIG.safe_dump(), ensure_ascii=False))


def install_excepthook(logger: Optional[Logger] = None) -> None:
    """
    捕获未处理异常并写入 error 日志（不吞异常）。
    """
    lg = logger or get_logger(__name__)
    _orig_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        lg.exception("Uncaught exception", exc_info=(exc_type, exc, tb))
        _orig_hook(exc_type, exc, tb)

    sys.excepthook = _hook  # type: ignore


class log_duration:
    """
    代码片段耗时记录：
        with log_duration(get_logger(__name__), "fetch_hot_search"):
            ... do work ...
    """

    def __init__(self, logger: Optional[Logger], label: str, level: int = logging.INFO):
        self.logger = logger or get_logger(__name__)
        self.label = label
        self.level = level
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.logger.log(self.level, "[%s] start", self.label)
        return self

    def __exit__(self, exc_type, exc, tb):
        cost = (time.perf_counter() - self._t0) * 1000.0
        if exc:
            self.logger.exception("[%s] failed (%.1f ms)", self.label, cost)
        else:
            self.logger.log(self.level, "[%s] done (%.1f ms)", self.label, cost)
        # 不吞异常
        return False
