# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午1:12
# @Author: 柯影数智
# @File: config.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

# ---------- 加载 .env ----------
try:
    from dotenv import load_dotenv  # type: ignore

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------- 工具 ----------
def _env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if (v is not None and v.strip() != "") else default


def _env_str_any(keys: List[str], default: str = "") -> str:
    for k in keys:
        v = os.getenv(k)
        if v is not None and v.strip() != "":
            return v
    return default


def _env_int(key: str, default: int) -> int:
    try:
        val = os.getenv(key, "").strip()
        return int(val) if val != "" else default
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_csv(key: str) -> Optional[List[str]]:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return None
    # 兼容中英文逗号
    parts = [p.strip() for p in v.replace("，", ",").split(",")]
    return [p for p in parts if p]


def _read_text_file(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return ""


def _abs_from_root(path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


# ---------- 常量 ----------
WEEKDAY_CN: List[str] = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
TECH_KEYWORDS_DEFAULT: List[str] = [
    "科技", "数码", "互联网", "AI", "大模型", "人工智能", "深度学习", "机器学习", "算法",
    "芯片", "半导体", "英伟达", "NVIDIA", "华为", "苹果", "Apple", "小米", "鸿蒙",
    "开源", "Linux", "编程", "Python", "Java", "Go", "Rust", "数据库", "云", "算力",
    "自动驾驶", "特斯拉", "机器人", "量子", "卫星", "火箭", "光刻", "操作系统"
]


# ---------- 配置主体 ----------
@dataclass
class Config:
    # 时区
    tz: str = field(default_factory=lambda: _env_str("TZ", "Asia/Shanghai"))

    # 接口/UA
    weibo_api: str = field(default_factory=lambda: _env_str("WEIBO_API", "https://weibo.com/ajax/side/hotSearch"))
    weibo_summary: str = field(
        default_factory=lambda: _env_str("WEIBO_SUMMARY", "https://s.weibo.com/top/summary?cate=realtimehot"))
    head_ua: str = field(default_factory=lambda: _env_str(
        "HEAD_UA",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ))

    # 路径资源
    outdir: str = field(default_factory=lambda: _abs_from_root(_env_str("OUTDIR", "archive")))
    screenshot_dir: str = field(default_factory=lambda: _abs_from_root(_env_str("SCREENSHOT_DIR", "screenshot")))
    bg: str = field(default_factory=lambda: _abs_from_root(_env_str("BG", "resource/hot_research.jpg")))
    font: str = field(default_factory=lambda: _abs_from_root(_env_str("FONT", "resource/heiti.ttf")))
    numfont: str = field(default_factory=lambda: _abs_from_root(_env_str("NUMFONT", "resource/SmileySans.ttf")))
    chromedriver: str = field(default_factory=lambda: _abs_from_root(
        _env_str("CHROMEDRIVER", "resource/chromedriver-mac-arm64/chromedriver")))

    # DeepSeek
    deepseek_api_key: str = field(default_factory=lambda: _env_str("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: _env_str("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_timeout_sec: int = field(default_factory=lambda: _env_int("DEEPSEEK_TIMEOUT_SEC", 75))
    deepseek_retries: int = field(default_factory=lambda: _env_int("DEEPSEEK_RETRIES", 1))

    # 企业微信（支持填完整 URL 或仅 key）
    wecom_webhook_key_or_url: str = field(
        default_factory=lambda: _env_str_any(["WECOM_WEBHOOK_KEY", "WECOM_WEBHOOK_URL"], ""))

    # 微博 Cookie（优先读文件）
    weibo_cookie_file: str = field(default_factory=lambda: _env_str("WEIBO_COOKIE_FILE", ""))
    weibo_cookie: str = field(default_factory=lambda: _env_str("WEIBO_COOKIE", ""))

    # 抓取层 HTTP
    request_timeout_sec: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT_SEC", 8))
    request_retries: int = field(default_factory=lambda: _env_int("REQUEST_RETRIES", 2))

    # 文案/筛选
    weekday_cn: List[str] = field(default_factory=lambda: WEEKDAY_CN)
    tech_keywords: List[str] = field(default_factory=lambda: (
            (_env_csv("TECH_KEYWORDS")) or
            ([l for l in _read_text_file(_env_str("TECH_KEYWORDS_FILE", "")).splitlines() if l.strip()] if _env_str(
                "TECH_KEYWORDS_FILE", "") else None) or
            TECH_KEYWORDS_DEFAULT
    ))

    # 自动化/默认行为
    default_topn: int = field(default_factory=lambda: _env_int("DEFAULT_TOPN", 10))
    default_tech_only: bool = field(default_factory=lambda: _env_bool("DEFAULT_TECH_ONLY", True))
    auto_summary: bool = field(default_factory=lambda: _env_bool("AUTO_SUMMARY", True))
    auto_image: bool = field(default_factory=lambda: _env_bool("AUTO_IMAGE", True))
    auto_screenshot: bool = field(default_factory=lambda: _env_bool("AUTO_SCREENSHOT", True))
    auto_wechat: bool = field(default_factory=lambda: _env_bool("AUTO_WECHAT", True))
    multi_boards: bool = field(default_factory=lambda: _env_bool("MULTI_BOARDS", True))
    screenshot_desktop: bool = field(default_factory=lambda: _env_bool("SCREENSHOT_DESKTOP", False))
    no_headless: bool = field(default_factory=lambda: _env_bool("NO_HEADLESS", False))
    screenshot_timeout: int = field(default_factory=lambda: _env_int("SCREENSHOT_TIMEOUT", 20))
    no_cookie_inject: bool = field(default_factory=lambda: _env_bool("NO_COOKIE_INJECT", False))

    # 便捷
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def now_local(self) -> datetime:
        return datetime.now(self.tzinfo())

    @property
    def wecom_webhook_url(self) -> str:
        v = (self.wecom_webhook_key_or_url or "").strip()
        if not v:
            return ""
        if v.startswith(("http://", "https://")):
            return v
        return f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={v}"

    @property
    def wecom_webhook_key(self) -> str:
        # 兼容旧命名
        return self.wecom_webhook_url

    def _resolve_cookie(self) -> str:
        if self.weibo_cookie_file:
            txt = _read_text_file(self.weibo_cookie_file)
            if txt:
                return txt
        return self.weibo_cookie

    def weibo_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": self.head_ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        cookie = self._resolve_cookie()
        if cookie:
            headers["Cookie"] = cookie
        return headers

    def safe_dump(self) -> Dict[str, Any]:
        data = asdict(self)

        def _mask(token: str) -> str:
            if not token:
                return ""
            if len(token) <= 8:
                return "***"
            return f"{token[:4]}***{token[-4:]}"

        # 敏感脱敏
        data["deepseek_api_key"] = _mask(self.deepseek_api_key)
        raw = self.wecom_webhook_key_or_url or ""
        data["wecom_webhook_key_or_url"] = (raw if raw.startswith("http") else _mask(raw))
        # Cookie 仅显示长度
        cookie_txt = self._resolve_cookie()
        data["weibo_cookie"] = f"<set: len={len(cookie_txt)}>" if cookie_txt else ""
        # 路径保留（已是绝对路径）
        return data


# 单例
CONFIG = Config()


# 便捷函数（兼容旧代码）
def tzinfo() -> ZoneInfo:
    return CONFIG.tzinfo()


def now_local() -> datetime:
    return CONFIG.now_local()
