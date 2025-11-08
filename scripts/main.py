# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午5:20
# @Author: 柯影数智
# @File: main.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm


from __future__ import annotations

import sys as _sys
from pathlib import Path as _PTH

_PRJ = _PTH(__file__).resolve().parents[1]
if str(_PRJ) not in _sys.path:
    _sys.path.insert(0, str(_PRJ))
# ----------------------------------------------------

import os
import sys
import json
import argparse
import re
import html
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from urllib.parse import urlparse, parse_qs, urlencode

from weibo_hotsearch.config import CONFIG, tzinfo, WEEKDAY_CN
from weibo_hotsearch.fetch import fetch_hot_search
from weibo_hotsearch.utils import (
    build_markdown,
    make_session,
    fetch_weibo_summary_html,
    parse_weibo_html,
    dedupe_and_sort,
)

# ========= 可选图像合成能力探测 =========
try:
    from weibo_hotsearch.render_image import (
        render_image,
        render_three_boards_image,  # 兼容旧导入
        render_two_boards_image,  # 若存在则用官方两榜合成
    )  # type: ignore

    _HAS_MERGED = True
    _HAS_TWO_MERGED = True
except Exception:
    try:
        from weibo_hotsearch.render_image import render_image  # type: ignore
    except Exception:
        # 若项目旧版本无此模块，后续调用会失败；此处保持占位
        def render_image(*args, **kwargs):
            raise RuntimeError("render_image is not available")
    render_three_boards_image = None  # type: ignore
    render_two_boards_image = None  # type: ignore
    _HAS_MERGED = False
    _HAS_TWO_MERGED = False

# 兜底：若无官方两榜合成，尝试用 Pillow 拼接两张单榜图
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

from weibo_hotsearch.summary_deepseek import summarize_items
from weibo_hotsearch.screenshot import capture_summary_screenshot
from weibo_hotsearch.wecom import send_markdown, send_image_from_path
from weibo_hotsearch.logutil import get_logger

logger = get_logger("weibo_hotsearch.bootstrap")


# ========== 轻量进度条 ==========
class _Progress:
    def __init__(self, total: int, width: int = 26):
        self.total = max(1, total)
        self.width = max(10, width)
        self.cur = 0
        self._term_is_tty = sys.stderr.isatty() and (os.getenv("NO_TTY") != "1")

    def _bar(self) -> str:
        filled = int(self.width * self.cur / self.total)
        return "█" * filled + "░" * (self.width - filled)

    def step(self, msg: str) -> None:
        self.cur = min(self.cur + 1, self.total)
        prefix = f"[{self.cur:02d}/{self.total:02d}]"
        if self._term_is_tty:
            sys.stderr.write(f"\r{prefix} {self._bar()}  {msg.ljust(36)[:36]}")
            sys.stderr.flush()
            if self.cur == self.total:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(f"{prefix} {msg}\n")

    def note(self, msg: str) -> None:
        sys.stderr.write(f"    └─ {msg}\n")


# ========== 安全日志与掩码 ==========
def _mask_middle(txt: str, visible: int = 3) -> str:
    """将字符串中间替换为 **，仅保留前后若干字符。"""
    s = str(txt or "")
    if len(s) <= visible * 2:
        return "**"
    return s[:visible] + "**" + s[-visible:]


def _mask_url(url: str) -> str:
    """仅保留主机，路径与查询以 ** 替代。"""
    u = str(url or "").strip()
    try:
        pr = urlparse(u)
        if not pr.scheme or not pr.netloc:
            return _mask_middle(u)
        base = f"{pr.scheme}://{pr.netloc}"
        if pr.query:
            q = parse_qs(pr.query, keep_blank_values=True)
            masked = {k: ["**"] * len(v) for k, v in q.items()}
            return base + pr.path.rstrip("/") + "?" + urlencode(masked, doseq=True)
        if pr.path and pr.path != "/":
            return base + "/**"
        return base
    except Exception:
        return _mask_middle(u)


def _safe_dump_config() -> Dict[str, Any]:
    """对 CONFIG.safe_dump() 的敏感字段做二次掩码。"""
    try:
        raw = CONFIG.safe_dump()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    masked = {}
    for k, v in raw.items():
        kl = str(k).lower()
        if isinstance(v, str):
            if any(t in kl for t in ["key", "token", "secret", "password", "cookie", "webhook"]):
                masked[k] = "**"
            elif any(t in kl for t in ["url", "endpoint", "base"]):
                masked[k] = _mask_url(v)
            else:
                masked[k] = v
        else:
            masked[k] = v
    return masked


# ========== env ==========
def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default


def _str_env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return (v if v is not None else default).strip()


# ========== wecom webhook 解析 ==========
def _resolve_wecom_url() -> str:
    v = ""
    try:
        attr = getattr(CONFIG, "wecom_webhook_url", "")
        v = attr() if callable(attr) else (attr or "")
    except Exception:
        v = ""
    if not v:
        raw = getattr(CONFIG, "wecom_webhook_key_or_url", "") or ""
        raw = raw.strip()
        if raw:
            if raw.startswith(("http://", "https://")):
                v = raw
            else:
                v = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={raw}"
    return v.strip()


# ========== 关键词：解析与过滤 ==========
def _normalize_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s;/、，；]+", raw)
    ks = [p.strip().lower() for p in parts if p and p.strip()]
    # 去重且保序
    seen = set()
    out = []
    for k in ks:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _load_keywords_for_boards(args: argparse.Namespace) -> List[str]:
    """
    优先级：
      --keywords > --keywords-file > 环境变量 KEYWORDS > --tech-only 用 CONFIG.tech_keywords > []
    """
    # 1) --keywords
    if getattr(args, "keywords", ""):
        return _normalize_keywords(args.keywords)

    # 2) --keywords-file
    if getattr(args, "keywords_file", ""):
        p = Path(args.keywords_file)
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            return _normalize_keywords(txt)

    # 3) 环境变量 KEYWORDS
    env_k = _str_env("KEYWORDS", "")
    if env_k:
        return _normalize_keywords(env_k)

    # 4) --tech-only => CONFIG.tech_keywords
    if getattr(args, "tech_only", False):
        ks = getattr(CONFIG, "tech_keywords", []) or []
        return [str(x).strip().lower() for x in ks if str(x).strip()]

    # 5) 默认：不筛选
    return []


def _filter_items_by_keywords(items: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    if not keywords:
        return items
    out = []
    for it in items:
        title = (it.get("title") or "").lower()
        label = (it.get("label") or "").lower()
        txt = f"{title} {label}"
        if any(k in txt for k in keywords):
            out.append(it)
    return out


# ========== 工具 ==========
def _timestamp_filename(ext: str = "png", suffix: str = "") -> str:
    """
    生成时间戳文件名；可选 suffix（如 "热搜榜" / "社会榜" / "两大榜"）
    """
    now = datetime.now(tzinfo())
    base = now.strftime(f"20%y年%m月%d日%H:%M")
    if suffix:
        return f"{base}.{suffix}.{ext}"
    return f"{base}.{ext}"


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    _ensure_dir(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(text: str, path: Path) -> None:
    _ensure_dir(path)
    path.write_text(text, encoding="utf-8")


def _render_plain(pinned: Optional[str], items: List[Dict[str, Any]], source: str | None = None) -> str:
    now = datetime.now(tzinfo())
    ts = now.strftime("微博热搜榜 20%y年%m月%d日 %H:%M ")
    wd = WEEKDAY_CN[(now.isoweekday() - 1) % 7]
    head = f"{ts}{wd}"
    if source:
        head += f"（来源：{source}）"
    lines = [head]
    if pinned:
        lines.append(f"置顶: {pinned}")
    for i, it in enumerate(items, 1):
        tag = f" {it.get('label')}" if it.get("label") else ""
        lines.append(f"{i}. {it.get('title')}{tag} 链接：{it.get('url')}")
    return "\n".join(lines)


# ========== 渲染兼容包装 ==========
def _render_image_compat(items: List[Dict[str, Any]], out_path: Path, board_title: Optional[str] = None) -> str:
    """
    兼容不同版本的 render_image 签名，统一返回“绝对路径字符串”
    新版：render_image(items=..., out_path=..., board_title=...)
    旧版：render_image(items, out_path)
    极老：render_image(items, bg, out_path, font, numfont)
    """
    try:
        p = render_image(items=items, out_path=out_path, board_title=board_title)  # 新版
    except TypeError:
        try:
            p = render_image(items, out_path)  # 旧版
        except TypeError:
            bg = str(getattr(CONFIG, "bg", "") or "")
            font = str(getattr(CONFIG, "font", "") or "")
            numfont = str(getattr(CONFIG, "numfont", "") or "")
            p = render_image(items, bg, out_path, font, numfont)  # 极老兜底
    return str(p)


def _compose_two_images_vert(top_img_path: str, bottom_img_path: str, out_path: Path) -> str:
    """
    用 Pillow 将上下两张榜单图合并，并在顶部绘制时间戳 + 「热搜榜 · 社会榜」标签。
    若 Pillow 不可用，抛异常由上层回退。
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is not available for composing images")

    img_top = Image.open(top_img_path).convert("RGB")
    img_bottom = Image.open(bottom_img_path).convert("RGB")

    W = max(img_top.width, img_bottom.width)
    gap = 16  # 中间留白
    header_h = 112
    H = header_h + img_top.height + gap + img_bottom.height

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 顶部头条条
    bar_h = header_h
    draw.rectangle([(0, 0), (W, bar_h)], fill=(250, 250, 250))

    # 时间戳 + 标签
    now = datetime.now(tzinfo())
    ts = now.strftime("20%y年%m月%d日 %H:%M")
    header_text = f"微博两大榜｜{ts}｜热搜榜 · 社会榜"
    try:
        font_path = str(getattr(CONFIG, "font", "") or "")
        font = ImageFont.truetype(font_path, 36) if font_path and Path(font_path).exists() else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    tw, th = draw.textsize(header_text, font=font)
    draw.text(((W - tw) // 2, (bar_h - th) // 2), header_text, fill=(34, 34, 34), font=font)

    # 粘贴两张图
    x_top = (W - img_top.width) // 2
    x_bottom = (W - img_bottom.width) // 2
    canvas.paste(img_top, (x_top, header_h))
    canvas.paste(img_bottom, (x_bottom, header_h + img_top.height + gap))

    _ensure_dir(out_path)
    canvas.save(out_path, format="PNG")
    return str(out_path)


def _render_two_boards_merged(
        boards: "OrderedDict[str, List[Dict[str, Any]]]",
        out_path: Path,
) -> Tuple[bool, List[str]]:
    """
    优先使用 render_two_boards_image；无则生成两张单图再用 Pillow 合并。
    返回 (has_single_merged_image, paths)：
      - 若 True：paths=[merged_png]
      - 若 False：paths=[每榜各一张]（回退）
    """
    # 官方两榜合成
    if callable(render_two_boards_image):
        try:
            p = render_two_boards_image(boards, out_path)  # type: ignore
            return True, [str(p)]
        except Exception as e:
            logger.warning("render_two_boards_image 失败，将尝试 Pillow 合并：%s", e)

    # Pillow 合成
    try:
        # 先各自出图（确保文件名带“热搜榜/社会榜”）
        tmp_dir = out_path.parent
        top_name = _timestamp_filename("png", "热搜榜")
        bottom_name = _timestamp_filename("png", "社会榜")
        p_top = _render_image_compat(boards.get("热搜榜", []), tmp_dir / top_name, board_title="热搜榜")
        p_bottom = _render_image_compat(boards.get("社会榜", []), tmp_dir / bottom_name, board_title="社会榜")
        merged = _compose_two_images_vert(p_top, p_bottom, out_path)
        return True, [merged]
    except Exception as e:
        logger.warning("Pillow 合并失败，回退为两张单图：%s", e)
        # 回退为多张
        paths: List[str] = []
        for name, items in boards.items():
            if not items:
                continue
            p = _render_image_compat(items, out_path.parent / _timestamp_filename("png", name), board_title=name)
            paths.append(p)
        return False, paths


# ========== 多榜抓取（支持可选关键词过滤） ==========
def _find_tabs_from_summary(html_text: str) -> Dict[str, str]:
    tabs: Dict[str, str] = {}
    for m in re.finditer(r'href="[^"]*?/top/summary\?cate=([^"&]+)".*?>(.*?)</a>', html_text, re.I | re.S):
        cate = html.unescape(m.group(1)).strip()
        text = re.sub(r"<.*?>", "", html.unescape(m.group(2))).strip()
        if cate and text:
            tabs[text] = cate
    return tabs


def _fetch_board_by_cate(cate: str, topn: int) -> Tuple[List[Dict[str, Any]], str]:
    sess = make_session(timeout=getattr(CONFIG, "request_timeout_sec", 8),
                        retries=getattr(CONFIG, "request_retries", 2))
    url = f"https://s.weibo.com/top/summary?cate={cate}"
    try:
        r = sess.get(url, headers=CONFIG.weibo_headers())
        if r.status_code != 200 or not r.text:
            return [], f"抓取 {cate} 失败（HTTP {r.status_code}）"
        items = parse_weibo_html(r.text)
        items = dedupe_and_sort(items)
        return items[: max(0, len(items))], ""  # 先不截断，过滤后再截
    except Exception as e:
        return [], f"抓取 {cate} 失败：{e}"


def _fetch_two_boards(topn: int, keywords: Optional[List[str]] = None) -> Tuple[
    OrderedDict[str, List[Dict[str, Any]]], List[str]]:
    """
    返回：OrderedDict([("热搜榜", [...]), ("社会榜", [...])])，可按关键词过滤后再截取 topn
    """
    sess = make_session(timeout=getattr(CONFIG, "request_timeout_sec", 8),
                        retries=getattr(CONFIG, "request_retries", 2))
    root_html = fetch_weibo_summary_html(sess)
    tabs = _find_tabs_from_summary(root_html or "")

    cate_map: Dict[str, str] = {"热搜榜": "realtimehot", "社会榜": "socialevent"}
    for k, v in tabs.items():
        if ("社会" in k or "要闻" in k) and v:
            cate_map["社会榜"] = v
            break

    boards: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    hints: List[str] = []
    kw = keywords or []

    for name, cate in [("热搜榜", cate_map["热搜榜"]), ("社会榜", cate_map["社会榜"])]:
        if not cate:
            hints.append(f"未在页面导航中找到“{name}”，已跳过。")
            boards[name] = []
            continue
        items, hint = _fetch_board_by_cate(cate, topn)
        if hint:
            hints.append(f"{name}：{hint}")
        # 关键词过滤（如指定）
        items = _filter_items_by_keywords(items, kw)
        # 再截取 topn
        boards[name] = items[:topn]

    return boards, hints


# ========== AI 摘要（带 DeepSeek 调用状态回传） ==========
def _summarize_boards_with_retry(boards: Dict[str, List[Dict[str, Any]]], *, max_words=400, retries=1) -> Tuple[
    dict, bool]:
    out: dict[str, str] = {}
    used_api = False

    def _one(items):
        nonlocal used_api
        try:
            txt = summarize_items(items, max_words=max_words)
            if txt:
                used_api = True
            return txt or ""
        except Exception:
            return ""

    for _ in range(retries + 1):
        for name, items in boards.items():
            if name not in out:
                txt = _one(items)
                if txt:
                    out[name] = txt
        if "_overall" not in out:
            head_items = []
            for v in boards.values():
                head_items.extend(v[:3])
            txt = _one(head_items)
            if txt:
                out["_overall"] = txt
        if len(out) >= len(boards) + 1:
            break
    return out, used_api


# ========== Markdown 预览（仅摘要 + 两榜 Top3 超链接） ==========
def _strip_ai_links_block(s: str) -> str:
    if not s:
        return s
    return re.sub(r"\n*【原文链接】[\s\S]*$", "", s).rstrip()


def _md_summary_only(
        board_names: List[str],
        summary_text: str | None,
        boards: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        per_board_top: int = 3,
) -> str:
    now = datetime.now(tzinfo())
    head = f"**微博两大榜｜{now.strftime('20%y年%m月%d日 %H:%M')} {WEEKDAY_CN[(now.isoweekday() - 1) % 7]}**"
    sub = "、".join([f"**{n}**" for n in board_names if n])

    core = summary_text or "（摘要生成失败）"
    core = _strip_ai_links_block(core)

    md = f"{head}\n\n> 本期包含：{sub}\n\n> **AI 摘要**\n\n{core}"

    if boards:
        for name in ("热搜榜", "社会榜"):
            items = (boards.get(name) or [])[:per_board_top]
            if not items:
                continue
            md += f"\n\n**{name} Top {len(items)} 原文链接**\n"
            for i, it in enumerate(items, 1):
                title = (it.get("title") or "").strip()
                url = (it.get("url") or "").strip()
                if title and url:
                    md += f"{i}. [{title}]({url})\n"

    if len(md) > 3900:
        md = md[:3880].rstrip() + "…"
    return md


# ========== 子命令 ==========
def cmd_fetch(args: argparse.Namespace) -> int:
    pb = _Progress(total=2)
    pb.step("抓取热搜（单榜）")
    payload = fetch_hot_search(
        topn=args.topn,
        tech_only=args.tech_only,
        selenium_fallback=args.selenium_fallback,
        driver_path=args.driver,
    )
    pinned = payload.get("pinned") or None
    items = payload.get("items") or []
    source = payload.get("source")
    pb.step("输出结果")

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.format == "md":
        md = build_markdown(items, title_prefix="微博热搜", summary_text=None, tech_only=args.tech_only)
        if payload.get("hint"):
            md += f"\n> 提示：{payload['hint']}"
        print(md)
    else:
        text = _render_plain(pinned, items, source=source)
        if payload.get("hint"):
            text += f"\n\n提示：{payload['hint']}"
        print(text)

    if args.save_json:
        outp = Path(args.save_json)
        _save_json(payload, outp)
        print(f"\n[OK] 已保存 JSON：{outp}")
    return 0


def cmd_image(args: argparse.Namespace) -> int:
    pb = _Progress(total=3)
    pb.step("准备条目")
    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        items = data.get("items") or []
    else:
        payload = fetch_hot_search(
            topn=args.topn, tech_only=args.tech_only,
            selenium_fallback=args.selenium_fallback,
            driver_path=args.driver,
        )
        items = payload.get("items") or []
        if payload.get("hint"):
            sys.stderr.write(f"[提示] {payload['hint']}\n")

    if not items:
        sys.stderr.write("[ERROR] 没有可用条目。\n")
        return 2

    pb.step("渲染图片")
    outdir = Path(args.outdir or CONFIG.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (args.output or _timestamp_filename("png", "热搜榜"))
    img_path = _render_image_compat(items, out_path, board_title="热搜榜")
    pb.step("完成")
    print(f"[OK] 图片已保存：{img_path}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    pb = _Progress(total=3)
    pb.step("准备条目")
    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        items = data.get("items") or []
    else:
        payload = fetch_hot_search(
            topn=args.topn, tech_only=args.tech_only,
            selenium_fallback=args.selenium_fallback,
            driver_path=args.driver,
        )
        items = payload.get("items") or []
        if payload.get("hint"):
            sys.stderr.write(f"[提示] {payload['hint']}\n")

    if not items:
        sys.stderr.write("[ERROR] 没有可用条目。\n")
        return 2

    has_key = bool((CONFIG.deepseek_api_key or "").strip())
    pb.step("调用 DeepSeek 生成摘要")
    text = summarize_items(items, max_words=args.summary_max)
    used_api = bool(text)
    base_mask = _mask_url(getattr(CONFIG, "deepseek_base_url", ""))
    print(f"[DeepSeek] available={has_key} called=True success={used_api} "
          f"model={getattr(CONFIG, 'deepseek_model', '')} base={base_mask}")

    if not text:
        sys.stderr.write("[WARN] DeepSeek 摘要失败或未配置 DEEPSEEK_API_KEY。\n")
        return 3
    pb.step("输出/保存")
    print(text)
    if args.save:
        p = Path(args.save)
        _save_text(text, p)
        print(f"\n[OK] 摘要已保存：{p}")
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    extra = (1 if args.summary else 0) + (1 if args.with_image else 0) + (1 if args.with_screenshot else 0)
    pb = _Progress(total=4 + extra)
    pb.step("抓取热搜（单榜）")

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        items = data.get("items") or []
        hint = None
    else:
        payload = fetch_hot_search(
            topn=args.topn, tech_only=args.tech_only,
            selenium_fallback=args.selenium_fallback,
            driver_path=args.driver,
        )
        items = payload.get("items") or []
        hint = payload.get("hint")

    if not items:
        sys.stderr.write("[ERROR] 没有可用条目。\n")
        if hint:
            sys.stderr.write(f"[抓取提示] {hint}\n")
        return 2

    summary_text = None
    has_key = bool((CONFIG.deepseek_api_key or "").strip())

    if args.summary:
        pb.step("调用 DeepSeek 生成摘要")
        summary_text = summarize_items(items, max_words=args.summary_max)
        base_mask = _mask_url(getattr(CONFIG, "deepseek_base_url", ""))
        print(f"[DeepSeek] available={has_key} called=True success={bool(summary_text)} "
              f"model={getattr(CONFIG, 'deepseek_model', '')} base={base_mask}")
        if not summary_text:
            sys.stderr.write("[WARN] DeepSeek 摘要失败或未配置 DEEPSEEK_API_KEY，已跳过摘要。\n")

    pb.step("构建 Markdown")
    md = build_markdown(items, title_prefix="微博热搜", summary_text=summary_text, tech_only=args.tech_only)

    image_path = None
    if args.with_image:
        pb.step("渲染榜单图片")
        outdir = Path(args.outdir or CONFIG.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / (args.output or _timestamp_filename("png", "热搜榜"))
        image_path = _render_image_compat(items, out_path, board_title="热搜榜")
        print(f"[OK] 榜单图片已保存：{image_path}")

    screenshot_path = None
    if args.with_screenshot:
        pb.step("浏览器整页截图")
        try:
            screenshot_path = capture_summary_screenshot(
                driver_path=args.driver or str(CONFIG.chromedriver),
                outdir=str(CONFIG.screenshot_dir),
                desktop=args.desktop,
                headless=not args.no_headless,
                timeout=args.timeout,
                inject_cookie=not args.no_cookie,
            )
            print(f"[OK] 整页截图已保存：{screenshot_path}")
        except Exception as e:
            sys.stderr.write(f"[WARN] 截图失败：{e}\n")

    webhook = (args.wecom_key or _resolve_wecom_url() or "").strip()
    do_send = args.wecom or bool(webhook)

    pb.step("发送到企业微信" if do_send else "预览 Markdown")
    if not do_send:
        print("\n=== Markdown 预览 ===\n")
        print(md)
        if hint:
            print(f"\n[抓取提示] {hint}")
        if image_path:
            print(f"\n[提示] 榜单图片：{image_path}")
        if screenshot_path:
            print(f"[提示] 整页截图：{screenshot_path}")
        return 0

    ok_md, md_resps = send_markdown(md, webhook=webhook)
    print(f"[WeCom/markdown] sent={ok_md} resp={md_resps[-1] if md_resps else {} }")
    all_ok = ok_md

    if image_path:
        pb.step("发送榜单图片")
        ok_img, resp_img = send_image_from_path(image_path, webhook=webhook)
        print(f"[WeCom/image] sent={ok_img} resp={resp_img}")
        all_ok = all_ok and ok_img

    if screenshot_path:
        pb.step("发送整页截图")
        ok_shot, resp_shot = send_image_from_path(screenshot_path, webhook=webhook)
        print(f"[WeCom/screenshot] sent={ok_shot} resp={resp_shot}")
        all_ok = all_ok and ok_shot

    return 0 if all_ok else 4


def cmd_screenshot(args: argparse.Namespace) -> int:
    pb = _Progress(total=2)
    pb.step("启动浏览器并访问页面")
    try:
        path = capture_summary_screenshot(
            driver_path=args.driver or str(CONFIG.chromedriver),
            outdir=str(args.outdir or CONFIG.screenshot_dir),
            url=args.url,
            desktop=args.desktop,
            headless=not args.no_headless,
            timeout=args.timeout,
            inject_cookie=not args.no_cookie,
        )
        pb.step("保存截图")
        print(f"[OK] 截图完成：{path}")
        return 0
    except Exception as e:
        sys.stderr.write(f"[ERROR] 截图失败：{e}\n")
        return 2


def cmd_boards(args: argparse.Namespace) -> int:
    # 解析关键词（为空则不筛选；--tech-only 则用 CONFIG.tech_keywords）
    keywords = _load_keywords_for_boards(args)
    if keywords:
        print(f"[Filter] 关键词启用：{keywords}")
    else:
        print("[Filter] 不启用关键词筛选（原生态）")

    # 根据能力预估图片数量（优先合并为一张）
    can_merge = bool(callable(render_two_boards_image) or _PIL_AVAILABLE)
    img_count = 1 if can_merge else 2
    steps = 1 + (1 if args.summary else 0) + 1 + 1 + 1 + img_count  # 抓取/摘要?/构建MD/渲图/发MD/发图xN
    pb = _Progress(total=steps)

    # 1) 抓取两榜
    pb.step("抓取两大榜（热搜/社会）")
    boards, hints = _fetch_two_boards(args.topn, keywords=keywords)
    if not any(boards.values()):
        sys.stderr.write("[ERROR] 两个榜单均未抓到有效数据。\n")
        if hints:
            sys.stderr.write("；".join(hints) + "\n")
        return 2

    # 2) AI 摘要
    summaries = None
    summary_text = None
    has_key = bool((CONFIG.deepseek_api_key or "").strip())

    if args.summary and has_key:
        pb.step("调用 DeepSeek 生成摘要（两榜）")
        summaries, _ = _summarize_boards_with_retry(boards, max_words=args.summary_max, retries=1)
        summary_text = summaries.get("热搜榜") or summaries.get("_overall")
        base_mask = _mask_url(getattr(CONFIG, "deepseek_base_url", ""))
        print(f"[DeepSeek] available={has_key} called=True success={bool(summary_text)} "
              f"model={getattr(CONFIG, 'deepseek_model', '')} base={base_mask}")
    elif args.summary and not has_key:
        pb.step("跳过摘要（未配置 DeepSeek API Key）")
        print("[DeepSeek] available=False called=False success=False")
    else:
        pb.step("跳过摘要（未开启 --summary）")
        print(f"[DeepSeek] available={has_key} called=False success=False")

    # 3) Markdown（AI 摘要 + 两榜 Top3）
    pb.step("构建 Markdown（含两榜 Top3 原文链接）")
    md = _md_summary_only(list(boards.keys()), summary_text, boards)

    print("\n=== Markdown 预览（AI 摘要） ===\n")
    print(md)

    # 4) 渲染图片：优先两榜合并
    outdir = Path(args.outdir or CONFIG.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    merged_paths: List[str] = []

    pb.step("渲染图片：两大榜合并" if can_merge else "渲染图片：逐榜输出")
    if can_merge:
        out_path = outdir / _timestamp_filename("png", "两大榜")
        merged_ok, paths = _render_two_boards_merged(boards, out_path)
        if merged_ok:
            merged_paths = paths
            print(f"[OK] 两大榜 图片已保存：{paths[0]}")
        else:
            merged_paths = paths  # 回退多张
            for p in merged_paths:
                print(f"[OK] 图片已保存：{p}")
    else:
        for name, items in boards.items():
            if not items:
                continue
            p = _render_image_compat(items, outdir / _timestamp_filename("png", name), board_title=name)
            merged_paths.append(p)
            print(f"[OK] {name} 图片已保存：{p}")

    # 5) 发送 Markdown
    webhook = (args.wecom_key or _resolve_wecom_url() or "").strip()
    do_send = args.wecom or bool(webhook)

    pb.step("发送 Markdown" if do_send else "预览完成（未发送）")
    if not do_send:
        if hints:
            print("\n[抓取提示]", "；".join(hints))
        return 0

    ok_md, md_resps = send_markdown(md, webhook=webhook)
    logger.info("WeCom markdown sent_all_ok=%s parts=%d", ok_md, len(md_resps))
    print(f"[WeCom/markdown] sent={ok_md} resp={md_resps[-1] if md_resps else {} }")
    all_ok = ok_md

    # 6) 发送图片（1 或 2 张）
    for p in merged_paths:
        pb.step(f"发送图片：{Path(p).name}")
        ok_img, resp_img = send_image_from_path(p, webhook=webhook)
        print(f"[WeCom/image:{Path(p).name}] sent={ok_img} resp={resp_img}")
        all_ok = all_ok and ok_img

    return 0 if all_ok else 4


def cmd_daily(args: argparse.Namespace) -> int:
    # 默认用两榜流程（与 boards 一致）
    if _bool_env("MULTI_BOARDS", True):
        ns = argparse.Namespace(
            topn=args.topn,
            tech_only=args.tech_only,
            summary=True,
            summary_max=args.summary_max,
            outdir=args.outdir,
            wecom=True,
            wecom_key="",
            keywords=getattr(args, "keywords", ""),
            keywords_file=getattr(args, "keywords_file", ""),
        )
        return cmd_boards(ns)

    # 兼容旧 daily 单榜流程
    extra = (1 if args.summary else 0) + (1 if args.with_image else 0) + (1 if args.with_screenshot else 0)
    pb = _Progress(total=4 + extra)
    pb.step("抓取热搜（单榜）")

    payload = fetch_hot_search(
        topn=args.topn,
        tech_only=args.tech_only,
        selenium_fallback=args.selenium_fallback,
        driver_path=args.driver or str(CONFIG.chromedriver),
    )
    items = payload.get("items") or []
    if not items:
        sys.stderr.write("[ERROR] 未获取到热搜条目；")
        if payload.get("hint"):
            sys.stderr.write(payload["hint"] + "\n")
        else:
            sys.stderr.write("可尝试设置 WEIBO_COOKIE 或启用 --selenium-fallback。\n")
        return 2

    json_path = Path(CONFIG.outdir) / _timestamp_filename("json")
    _save_json(payload, json_path)
    print(f"[OK] 抓取结果已保存：{json_path}")

    summary_text = None
    has_key = bool((CONFIG.deepseek_api_key or "").strip())

    if args.summary:
        pb.step("调用 DeepSeek 生成摘要")
        summary_text = summarize_items(items, max_words=args.summary_max)
        base_mask = _mask_url(getattr(CONFIG, "deepseek_base_url", ""))
        print(f"[DeepSeek] available={has_key} called=True success={bool(summary_text)} "
              f"model={getattr(CONFIG, 'deepseek_model', '')} base={base_mask}")
        if not summary_text:
            sys.stderr.write("[WARN] DeepSeek 摘要失败或未配置 DEEPSEEK_API_KEY，已跳过摘要。\n")

    image_path = None
    if args.with_image:
        pb.step("渲染榜单图片")
        image_path = _render_image_compat(items, Path(CONFIG.outdir) / _timestamp_filename("png", "热搜榜"),
                                          board_title="热搜榜")
        print(f"[OK] 榜单图片已保存：{image_path}")

    screenshot_path = None
    if args.with_screenshot:
        pb.step("浏览器整页截图")
        try:
            screenshot_path = capture_summary_screenshot(
                driver_path=args.driver or str(CONFIG.chromedriver),
                outdir=str(CONFIG.screenshot_dir),
                desktop=args.desktop,
                headless=not args.no_headless,
                timeout=args.timeout,
                inject_cookie=not args.no_cookie,
            )
            print(f"[OK] 整页截图已保存：{screenshot_path}")
        except Exception as e:
            sys.stderr.write(f"[WARN] 截图失败：{e}\n")

    pb.step("构建/发送 Markdown")
    md = build_markdown(items, title_prefix="微博热搜", summary_text=summary_text, tech_only=args.tech_only)

    webhook = (args.wecom_key or _resolve_wecom_url() or "").strip()
    do_send = args.wecom or bool(webhook)
    if not do_send:
        print("\n=== Markdown 预览 ===\n")
        print(md)
        if payload.get("hint"):
            print(f"\n[抓取提示] {payload['hint']}")
        if image_path:
            print(f"\n[提示] 榜单图片：{image_path}")
        if screenshot_path:
            print(f"[提示] 整页截图：{screenshot_path}")
        return 0

    ok_md, md_resps = send_markdown(md, webhook=webhook)
    print(f"[WeCom/markdown] sent={ok_md} resp={md_resps[-1] if md_resps else {} }")
    all_ok = ok_md

    if image_path:
        pb.step("发送图片")
        ok_img, resp_img = send_image_from_path(image_path, webhook=webhook)
        print(f"[WeCom/image] sent={ok_img} resp={resp_img}")
        all_ok = all_ok and ok_img

    if screenshot_path:
        pb.step("发送截图")
        ok_shot, resp_shot = send_image_from_path(screenshot_path, webhook=webhook)
        print(f"[WeCom/screenshot] sent={ok_shot} resp={resp_shot}")
        all_ok = all_ok and ok_shot

    return 0 if all_ok else 4


# ========== CLI wiring ==========
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="微博热搜自动化 · 统一入口 CLI")
    sub = p.add_subparsers(dest="cmd", required=False)

    f = sub.add_parser("fetch", help="抓取微博热搜并输出")
    f.add_argument("--topn", type=int, default=10)
    f.add_argument("--tech-only", action="store_true")
    f.add_argument("--selenium-fallback", action="store_true")
    f.add_argument("--driver", default=str(CONFIG.chromedriver))
    f.add_argument("--format", choices=["plain", "md", "json"], default="plain")
    f.add_argument("--save-json", default="")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("image", help="生成榜单图片")
    i.add_argument("--from-json", default="")
    i.add_argument("--topn", type=int, default=10)
    i.add_argument("--tech-only", action="store_true")
    i.add_argument("--selenium-fallback", action="store_true")
    i.add_argument("--driver", default=str(CONFIG.chromedriver))
    i.add_argument("--outdir", default=str(CONFIG.outdir))
    i.add_argument("--output", default="")
    i.set_defaults(func=cmd_image)

    s = sub.add_parser("summary", help="生成 DeepSeek 摘要（打印调用状态）")
    s.add_argument("--from-json", default="")
    s.add_argument("--topn", type=int, default=10)
    s.add_argument("--tech-only", action="store_true")
    s.add_argument("--selenium-fallback", action="store_true")
    s.add_argument("--driver", default=str(CONFIG.chromedriver))
    s.add_argument("--summary-max", type=int, default=400)
    s.add_argument("--save", default="")
    s.set_defaults(func=cmd_summary)

    se = sub.add_parser("send", help="推送到企业微信（markdown/图片/截图，打印 DeepSeek 调用状态与进度条）")
    se.add_argument("--from-json", default="")
    se.add_argument("--topn", type=int, default=10)
    se.add_argument("--tech-only", action="store_true")
    se.add_argument("--selenium-fallback", action="store_true")
    se.add_argument("--driver", default=str(CONFIG.chromedriver))
    se.add_argument("--summary", action="store_true")
    se.add_argument("--summary-max", type=int, default=400)
    se.add_argument("--with-image", action="store_true")
    se.add_argument("--with-screenshot", action="store_true")
    se.add_argument("--desktop", action="store_true")
    se.add_argument("--no-headless", action="store_true")
    se.add_argument("--timeout", type=int, default=15)
    se.add_argument("--no-cookie", action="store_true")
    se.add_argument("--outdir", default=str(CONFIG.outdir))
    se.add_argument("--output", default="")
    se.add_argument("--wecom", action="store_true")
    se.add_argument("--wecom-key", default="")
    se.set_defaults(func=cmd_send)

    sc = sub.add_parser("screenshot", help="热搜页整页截图（Selenium）")
    sc.add_argument("--driver", default=str(CONFIG.chromedriver))
    sc.add_argument("--outdir", default=str(CONFIG.screenshot_dir))
    sc.add_argument("--url", default="https://s.weibo.com/top/summary?cate=realtimehot")
    sc.add_argument("--desktop", action="store_true")
    sc.add_argument("--no-headless", action="store_true")
    sc.add_argument("--timeout", type=int, default=15)
    sc.add_argument("--no-cookie", action="store_true")
    sc.set_defaults(func=cmd_screenshot)

    b = sub.add_parser("boards", help="两榜：热搜/社会 → 可选关键词过滤 → AI 摘要（限长）+ 两榜Top3超链接 + 合并图 + 推送")
    b.add_argument("--topn", type=int, default=10)
    b.add_argument("--tech-only", action="store_true", help="启用科技关键词（使用 CONFIG.tech_keywords）")
    b.add_argument("--keywords", default="", help="自定义关键词（逗号/空格/中文逗号分隔）")
    b.add_argument("--keywords-file", default="", help="从文件读取关键词（每行或分隔符分隔）")
    b.add_argument("--summary", action="store_true")
    b.add_argument("--summary-max", type=int, default=400)
    b.add_argument("--outdir", default=str(CONFIG.outdir))
    b.add_argument("--wecom", action="store_true")
    b.add_argument("--wecom-key", default="")
    b.set_defaults(func=cmd_boards)

    d = sub.add_parser("daily", help="一条龙（默认两榜）：抓取→可选关键词→摘要→合并图/截图→推送")
    d.add_argument("--topn", type=int, default=10)
    d.add_argument("--tech-only", action="store_true")
    d.add_argument("--keywords", default="", help="自定义关键词")
    d.add_argument("--keywords-file", default="", help="关键词文件")
    d.add_argument("--selenium-fallback", action="store_true")
    d.add_argument("--driver", default=str(CONFIG.chromedriver))
    d.add_argument("--summary", action="store_true")
    d.add_argument("--summary-max", type=int, default=400)
    d.add_argument("--with-image", action="store_true")
    d.add_argument("--with-screenshot", action="store_true")
    d.add_argument("--desktop", action="store_true")
    d.add_argument("--no-headless", action="store_true")
    d.add_argument("--timeout", type=int, default=15)
    d.add_argument("--no-cookie", action="store_true")
    d.add_argument("--outdir", default=str(CONFIG.outdir))
    d.add_argument("--output", default="")
    d.add_argument("--wecom", action="store_true")
    d.add_argument("--wecom-key", default="")
    d.set_defaults(func=cmd_daily)

    return p


def main() -> int:
    try:
        logger.info("CONFIG snapshot: %s", json.dumps(_safe_dump_config(), ensure_ascii=False))
    except Exception:
        pass

    parser = build_parser()

    if len(sys.argv) == 1:
        topn = _int_env("DEFAULT_TOPN", 10)
        tech_only = _bool_env("DEFAULT_TECH_ONLY", True)
        use_summary = _bool_env("AUTO_SUMMARY", True) and bool(CONFIG.deepseek_api_key)
        auto_wechat = _bool_env("AUTO_WECHAT", True) and bool(_resolve_wecom_url())
        multi_boards = _bool_env("MULTI_BOARDS", True)
        env_keywords = _str_env("KEYWORDS", "")

        if multi_boards:
            args = parser.parse_args([
                "boards",
                "--topn", str(topn),
                *(["--tech-only"] if (tech_only and not env_keywords) else []),
                *(["--keywords", env_keywords] if env_keywords else []),
                *(["--summary"] if use_summary else []),
                "--summary-max", "400",
                *(["--wecom"] if auto_wechat else []),
                "--outdir", str(CONFIG.outdir),
            ])
        else:
            args = parser.parse_args([
                "daily",
                "--topn", str(topn),
                *(["--tech-only"] if (tech_only and not env_keywords) else []),
                *(["--keywords", env_keywords] if env_keywords else []),
                *(["--summary"] if use_summary else []),
                "--summary-max", "400",
                *(["--wecom"] if auto_wechat else []),
            ])
    else:
        args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.exception("Uncaught exception")
        raise
