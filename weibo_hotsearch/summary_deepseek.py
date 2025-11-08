# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午4:32
# @Author: 柯影数智
# @File: summary_deepseek.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

"""
weibo_hotsearch.summary_deepseek — 用 DeepSeek 生成“微博热搜（科技类）”摘要（可观测增强版）

改进要点
- 可观测：明确记录 SDK/HTTP 调用路径、耗时、状态码与重试；可用环境变量强制选路 & 打开调试日志
- 稳健：SDK 调用自动兼容不同 openai 版本的 timeout 传参；HTTP 支持 Retry-After、指数退避
- 结构：固定分节锚点；若模型漏“原文链接”，自动补齐；字数保护尽量不破坏 Markdown 结构
- 清洁：NFKC 宽窄转换（全角→半角），轻度空白压缩，适合企业微信群阅读
- CLI：新增 --force-http/--force-sdk/--debug/--probe 快速排障
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import CONFIG

# ------------------------
# 日志配置
# ------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    lvl = logging.DEBUG if os.getenv("DEEPSEEK_DEBUG") else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s")

# =========================
#  提示词与后处理
# =========================

_SYS_PROMPT = (
    "你是一名资深中文科技媒体编辑。你的任务：将给定的微博热搜条目（JSON）整理成结构化中文摘要。\n"
    "硬性要求：\n"
    "1) 只能依据入参 JSON 中的字段（title/url/hot/label）；不得添加或猜测 JSON 之外的信息、数字或链接。\n"
    "2) 客观、中性、克制，不渲染情绪，不使用夸张修辞或营销话术；不夹带主观判断和预测。\n"
    "3) 用简体中文；避免 Emoji；使用全角中文标点；保证可直接粘贴到企业微信群。\n"
    "4) 严格按照用户提示的格式输出，不要输出代码块或多余前后缀。"
)

_FMT_GUIDE = (
    "输出格式（必须严格遵守）：\n"
    "【要点】（最多 10 条，视实际条目数而定，每条 7–20 字，以“• ”开头）\n"
    "• ...\n"
    "...\n"
    "【趋势点评】（1 段，50–80 字，保持中性克制）\n"
    "...\n"
    "【话题标签】（3–6 个；以 # 开头；空格分隔；不加其他符号）\n"
    "#标签1 #标签2 #标签3\n"
    "【原文链接】（使用“序号. 标题（链接）”；标题来自 JSON；不得修改标题含义）\n"
    "1. 标题（https://...）\n"
    "2. 标题（https://...）"
)


def _build_user_prompt(items: List[Dict[str, Any]], max_words: int) -> str:
    return (
            "请基于下列 JSON（字段包含 title、url、hot、label）生成摘要。"
            "要求：\n"
            "• 优先覆盖热度更高、与科技/互联网/数码相关的条目；如非科技类也可简洁覆盖，但避免延展。\n"
            "• 【要点】每条 7–20 字，使用“• ”开头；语言简洁、事实导向；不要重复表述；不要带链接。\n"
            "• 【趋势点评】一段（50–80 字）；提炼共性与趋势，避免价值判断与夸张词。\n"
            "• 【话题标签】3–6 个，#开头；以主题/领域/关键词为主，不要空洞词语；不带标点。\n"
            "• 【原文链接】严格使用输入 JSON 的标题与链接；不要虚构或替换为其他链接。\n"
            f"• 总体字数不超过 {max_words} 字；如条目不足 10 条，按实际数量生成要点。\n\n"
            f"{_FMT_GUIDE}\n\n"
            "以下为 JSON：\n"
            + json.dumps(items, ensure_ascii=False, indent=2)
    )


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    t = t.strip("`")
    if "\n" in t:
        t = t.split("\n", 1)[-1]
        if "\n" in t:
            t = t.rsplit("\n", 1)[0]
    return t.strip()


def _safe_trim_markdown(text: str, limit_chars: int) -> str:
    if len(text) <= limit_chars:
        return text
    # 优先在“安全断点”处截断，尽量保护列表与链接
    safe_breaks = ["\n\n", "\n- ", "\n• ", "]()", "）\n", ")\n", "\n【原文链接】\n"]
    cut = limit_chars
    window = text[: limit_chars + 80]
    best = -1
    for b in safe_breaks:
        idx = window.rfind(b, 0, limit_chars)
        if idx > best:
            best = idx
    if best >= 0:
        cut = best + 1
    return text[:cut].rstrip() + "…"


def _build_links_block(items: List[Dict[str, Any]]) -> str:
    lines = ["【原文链接】"]
    for i, it in enumerate(items, 1):
        title = str(it.get("title") or "").strip()
        url = str(it.get("url") or "").strip()
        if not title or not url:
            continue
        lines.append(f"{i}. {title}（{url}）")
    return "\n".join(lines)


def _ensure_sections_and_links(t: str, items: List[Dict[str, Any]]) -> str:
    """
    模型有时漏写“原文链接”或写成别名，这里统一兜底补全。
    约定锚点：『【要点】』『【趋势点评】』『【话题标签】』『【原文链接】』
    """
    text = _strip_code_fence(t)

    # 若缺失链接块，追加
    if "【原文链接】" not in text:
        link_block = _build_links_block(items)
        if not text.endswith("\n"):
            text += "\n"
        text += ("\n" + link_block)

    # 若缺分节标题，温和补齐（不强造内容，仅补标题）
    if "【要点】" not in text:
        text = "【要点】\n" + text
    if "【趋势点评】" not in text:
        text = text.replace("【原文链接】", "【趋势点评】\n（略）\n\n【原文链接】", 1)
    if "【话题标签】" not in text:
        text = text.replace("【原文链接】", "【话题标签】\n#热点\n\n【原文链接】", 1)
    return text


def _normalize_readability(text: str) -> str:
    """
    统一宽窄（NFKC 全角→半角），并轻度空白收敛。
    """
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _postprocess(text: str, max_words: int, items_for_links: Optional[List[Dict[str, Any]]] = None) -> str:
    if not text:
        return ""
    t = _ensure_sections_and_links(text, items_for_links or [])
    t = _normalize_readability(t)
    hard_cap = max_words * 2
    return _safe_trim_markdown(t, hard_cap)


# =========================
#  DeepSeek 基础配置
# =========================

def _api_base() -> str:
    base = getattr(CONFIG, "deepseek_base_url", "") or "https://api.deepseek.com"
    return base.rstrip("/")


def _http_chat_endpoints() -> List[str]:
    b = _api_base()
    return [f"{b}/v1/chat/completions", f"{b}/chat/completions"]  # 兼容历史端点


def _effective_timeout_retries(timeout: Optional[int], retries: Optional[int]) -> Tuple[int, int]:
    cfg_timeout = getattr(CONFIG, "deepseek_timeout_sec", None)
    cfg_retries = getattr(CONFIG, "deepseek_retries", None)
    if cfg_timeout is None:
        base_rt = int(getattr(CONFIG, "request_timeout_sec", 8))
        cfg_timeout = max(12, base_rt + 4)
    if cfg_retries is None:
        cfg_retries = int(getattr(CONFIG, "request_retries", 2))
    t = int(timeout if timeout is not None else cfg_timeout)
    r = int(retries if retries is not None else cfg_retries)
    return max(1, t), max(1, r)


def _connect_read_timeouts(effective_timeout: int) -> Tuple[int, int] | int:
    conn = getattr(CONFIG, "deepseek_connect_timeout_sec", None)
    read = getattr(CONFIG, "deepseek_read_timeout_sec", None)
    if isinstance(conn, int) and isinstance(read, int) and conn > 0 and read > 0:
        return (conn, read)
    return effective_timeout


# =========================
#  DeepSeek 调用实现（SDK / HTTP）
# =========================

def _call_deepseek_http(
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        timeout: int = 30,
        retries: int = 1,
        max_tokens: int = 640,
) -> Optional[str]:
    session = requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "stop": ["```"],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }

    non_retry_status = {400, 401, 403, 422}
    endpoints = _http_chat_endpoints()
    attempt = 0
    backoff = 0.6
    req_timeout = _connect_read_timeouts(timeout)

    while attempt < max(1, retries):
        for url in endpoints:
            t0 = time.time()
            try:
                resp = session.post(url, headers=headers, json=payload, timeout=req_timeout)
                dt = (time.time() - t0) * 1000
                logger.debug(f"[DeepSeek HTTP] url={url} status={resp.status_code} dt={dt:.0f}ms")
                if resp.status_code == 200:
                    data = resp.json()
                    return (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if resp.status_code in non_retry_status:
                    sys.stderr.write(f"[DeepSeek HTTP] non-retry status={resp.status_code} body={resp.text[:200]}\n")
                    return None
                sys.stderr.write(f"[DeepSeek HTTP] status={resp.status_code} body={resp.text[:200]}\n")
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        time.sleep(min(int(ra), 8))
                    except Exception:
                        pass
            except requests.RequestException as e:
                sys.stderr.write(f"[DeepSeek HTTP] error: {e}\n")
        attempt += 1
        if attempt < retries:
            time.sleep(backoff)
            backoff = min(backoff * 2, 4.0)
    return None


def _call_deepseek_sdk(
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        timeout: int = 30,
        max_tokens: int = 640,
) -> Optional[str]:
    """
    兼容 openai Python SDK 的不同版本：
    - 新版本支持 create(..., timeout=秒)
    - 若抛出 TypeError（不支持该 kw），自动降级为不传 timeout
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        logger.debug("[DeepSeek SDK] openai 包不可用，跳过 SDK 路径")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=_api_base())
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_tokens=max_tokens,
                timeout=timeout,  # 新版
                stop=["```"],
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        except TypeError:
            # 老版 SDK 没有 timeout 参数
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_tokens=max_tokens,
                stop=["```"],
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        dt = (time.time() - t0) * 1000
        logger.debug(f"[DeepSeek SDK] chat.completions.create ok dt={dt:.0f}ms")
        return r.choices[0].message.content
    except Exception as e:
        sys.stderr.write(f"[DeepSeek SDK] error: {e}\n")
        return None


# =========================
#  对外主函数
# =========================

def _dedupe_and_compact_items(items: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    seen = set()
    compact: List[Dict[str, Any]] = []
    for it in items:
        title = str(it.get("title", "")).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        hot_raw = it.get("hot", 0)
        try:
            hot_val = int(hot_raw) if (isinstance(hot_raw, int) or str(hot_raw).isdigit()) else 0
        except Exception:
            hot_val = 0
        compact.append({
            "title": title[:120],
            "url": str(it.get("url", ""))[:512],
            "hot": hot_val,
            "label": str(it.get("label", ""))[:12],
        })
        if len(compact) >= limit:
            break
    # 让高热度优先（提示词已说明“优先覆盖热度更高”）
    compact.sort(key=lambda d: (-int(d.get("hot", 0)), d.get("title", "")))
    return compact


def summarize_items(
        items: List[Dict[str, Any]],
        *,
        max_words: int = 400,
        model: str = None,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        max_tokens: int = 640,
) -> Optional[str]:
    """
    对 items 生成结构化中文摘要。成功返回字符串，失败返回 None。
    """
    api_key = (CONFIG.deepseek_api_key or "").strip()
    if not api_key:
        sys.stderr.write("[DeepSeek] 未配置 DEEPSEEK_API_KEY，跳过摘要。\n")
        return None

    model = model or getattr(CONFIG, "deepseek_model", "deepseek-chat")
    eff_timeout, eff_retries = _effective_timeout_retries(timeout, retries)

    compact = _dedupe_and_compact_items(items, limit=12)
    if not compact:
        return ""

    messages = [
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": _build_user_prompt(compact, max_words=max_words)},
    ]

    # 路径选择：环境变量或默认先 SDK 再 HTTP
    force_http = os.getenv("DEEPSEEK_FORCE_HTTP") == "1"
    force_sdk = os.getenv("DEEPSEEK_FORCE_SDK") == "1"
    text: Optional[str] = None

    if force_sdk and force_http:
        logger.warning("同时设置了 DEEPSEEK_FORCE_SDK 与 DEEPSEEK_FORCE_HTTP，忽略强制，使用默认策略。")
        force_http = force_sdk = False

    if force_sdk:
        logger.debug("[DeepSeek] 强制 SDK 路径")
        text = _call_deepseek_sdk(api_key, model, messages,
                                  temperature=temperature, timeout=eff_timeout, max_tokens=max_tokens)
    elif force_http:
        logger.debug("[DeepSeek] 强制 HTTP 路径")
        text = _call_deepseek_http(api_key, model, messages,
                                   temperature=temperature, timeout=eff_timeout, retries=eff_retries,
                                   max_tokens=max_tokens)
    else:
        logger.debug("[DeepSeek] 默认策略：SDK → HTTP")
        text = _call_deepseek_sdk(api_key, model, messages,
                                  temperature=temperature, timeout=eff_timeout, max_tokens=max_tokens)
        if not text:
            text = _call_deepseek_http(api_key, model, messages,
                                       temperature=temperature, timeout=eff_timeout, retries=eff_retries,
                                       max_tokens=max_tokens)

    if not text:
        sys.stderr.write("[DeepSeek] 所有路径均未成功，返回 None。\n")
        return None

    return _postprocess(text, max_words=max_words, items_for_links=compact)


# =========================
#  简易兜底（无 Key 或超时重试失败时）
# =========================

def _local_fallback_summary(items: List[Dict[str, Any]], max_words: int = 400) -> str:
    items = _dedupe_and_compact_items(items, limit=10)

    bullets = []
    for it in items:
        t = it["title"]
        bullets.append("• " + (t[:28] + ("…" if len(t) > 28 else "")))

    trend = "整体关注集中于科技与民生资讯，具体议题分散；建议结合权威渠道核验后再扩展阅读。"
    tags = "#科技 #互联网 #趋势 #微博"

    links = _build_links_block(items)

    out = []
    out.append("【要点】")
    out.append("\n".join(bullets))
    out.append("\n【趋势点评】\n" + trend)
    out.append("\n【话题标签】\n" + tags)
    out.append("\n" + links)
    text = "\n".join(out).strip()
    hard_cap = max_words * 2
    return _safe_trim_markdown(text, hard_cap)


# =========================
#  连接连通性探测（可选）
# =========================

def probe_deepseek(api_key: str, model: str, timeout: int = 12) -> Tuple[bool, str]:
    """
    发送一条极小请求验证连通性/鉴权；不计入业务流。
    """
    messages = [{"role": "user", "content": "回复OK"}]
    try:
        txt = _call_deepseek_http(api_key, model, messages, timeout=timeout, retries=1, max_tokens=4)
        if txt and "OK" in txt.upper():
            return True, "HTTP OK"
        # 再试 SDK
        txt2 = _call_deepseek_sdk(api_key, model, messages, timeout=timeout, max_tokens=4)
        if txt2 and "OK" in txt2.upper():
            return True, "SDK OK"
        return False, "no OK signature"
    except Exception as e:
        return False, f"probe error: {e}"


# =========================
#  CLI
# =========================

def _read_items_from_file(path: str) -> List[Dict[str, Any]]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    data = json.loads(raw)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("无法从输入 JSON 解析出 items 列表")


def _cli() -> None:
    cfg_timeout, cfg_retries = _effective_timeout_retries(None, None)
    ap = argparse.ArgumentParser(description="用 DeepSeek 生成微博热搜摘要（含可观测/排障选项）")
    ap.add_argument("--input", default="", help="输入 JSON 文件路径（items 列表或含 items 的对象）；传 - 从 STDIN 读取")
    ap.add_argument("--use-fetch", action="store_true", help="忽略 --input，直接抓取热搜后摘要")
    ap.add_argument("--topn", type=int, default=10, help="抓取/摘要条数（默认10）")
    ap.add_argument("--tech-only", action="store_true", help="仅科技类（由上游抓取阶段控制）")
    ap.add_argument("--selenium-fallback", action="store_true", help="抓取失败时启用 Selenium 兜底")
    ap.add_argument("--driver", default=getattr(CONFIG, "chromedriver", ""), help="chromedriver 路径")
    ap.add_argument("--max-words", type=int, default=400, help="摘要最大字数（默认400）")
    ap.add_argument("--model", default=getattr(CONFIG, "deepseek_model", "deepseek-chat"), help="DeepSeek 模型名")
    ap.add_argument("--temperature", type=float, default=0.2, help="生成温度（默认0.2）")
    ap.add_argument("--timeout", type=int, default=cfg_timeout,
                    help="接口超时秒数（优先 deepseek_timeout_sec；否则 request_timeout_sec+4）")
    ap.add_argument("--retries", type=int, default=cfg_retries,
                    help="请求重试次数（优先 deepseek_retries；否则 request_retries）")
    ap.add_argument("--max-tokens", type=int, default=640, help="模型生成最大 tokens（默认640）")

    # 新增：调试/强制路径/连通性探测
    ap.add_argument("--force-http", action="store_true", help="强制走 HTTP 直连（忽略 SDK）")
    ap.add_argument("--force-sdk", action="store_true", help="强制走 SDK 路径（忽略 HTTP）")
    ap.add_argument("--debug", action="store_true", help="启用 DEBUG 日志")
    ap.add_argument("--probe", action="store_true", help="仅探测 DeepSeek 连通性，不生成摘要")

    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.probe:
        key = (CONFIG.deepseek_api_key or "").strip()
        if not key:
            print("未配置 DEEPSEEK_API_KEY")
            return
        ok, msg = probe_deepseek(key, args.model, timeout=min(12, args.timeout))
        print(f"probe: {ok} ({msg})")
        return

    # 允许 CLI 层覆盖环境策略
    if args.force_http:
        os.environ["DEEPSEEK_FORCE_HTTP"] = "1"
        os.environ.pop("DEEPSEEK_FORCE_SDK", None)
    if args.force_sdk:
        os.environ["DEEPSEEK_FORCE_SDK"] = "1"
        os.environ.pop("DEEPSEEK_FORCE_HTTP", None)

    if args.use_fetch:
        from .fetch import fetch_hot_search  # 延迟导入，避免循环依赖
        payload = fetch_hot_search(
            topn=args.topn,
            tech_only=args.tech_only,
            selenium_fallback=args.selenium_fallback,
            driver_path=args.driver,
        )
        items = payload.get("items", [])
    else:
        if not args.input:
            ap.error("请提供 --input 文件路径或使用 --use-fetch")
            return
        items = _read_items_from_file(args.input)

    if not items:
        print("[WARN] 没有可用于摘要的 items。")
        return

    text = summarize_items(
        items,
        max_words=args.max_words,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
    )

    if not text:
        text = _local_fallback_summary(items, max_words=args.max_words)

    print(text)


if __name__ == "__main__":
    _cli()
