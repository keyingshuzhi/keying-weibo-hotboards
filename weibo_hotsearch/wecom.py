# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午3:10
# @Author: 柯影数智
# @File: wecom.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm


from __future__ import annotations

import time
import base64
import hashlib
import mimetypes
from typing import List, Tuple, Union, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from datetime import datetime

import requests

from .config import CONFIG, WEEKDAY_CN, tzinfo

# ========== 常量 ==========
# 官方 markdown 单消息体上限（字符）
_WECOM_MD_LIMIT = 4096

# 退避参数
_BACKOFF_BASE = 0.6
_BACKOFF_MAX = 4.0

# 常见错误码
_ERR_TOO_LONG = 40058  # content 超长
_ERR_RATE_LIMITS = {93000, 45009}  # 频率/限流等
_ERR_INVALID_MEDIA_ID = 40005  # 媒体文件无效
_ERR_PARAM = {40001, 40002, 40003, 40004, 40007, 40008, 40009}

# 共享会话
_session = requests.Session()


# ========== 工具 ==========
def _effective_timeout(user_timeout: Optional[int]) -> int:
    try:
        return int(user_timeout if user_timeout is not None else getattr(CONFIG, "request_timeout_sec", 12)) or 12
    except Exception:
        return 12


def _effective_retries(user_retries: Optional[int]) -> int:
    try:
        return int(user_retries if user_retries is not None else getattr(CONFIG, "request_retries", 2)) or 2
    except Exception:
        return 2


def _resolve_webhook(webhook: Optional[str]) -> str:
    wb = (webhook or "").strip()
    if wb:
        return wb
    try:
        v = getattr(CONFIG, "wecom_webhook_url", "") or ""
        if callable(v):
            v = v()
        return (v or "").strip()
    except Exception:
        return ""


def _extract_key_from_webhook(webhook: str) -> str:
    """
    从 webhook URL 中提取 key（用于上传 media/file）
    https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx
    """
    try:
        qs = parse_qs(urlparse(webhook).query)
        k = (qs.get("key") or [""])[0]
        return k.strip()
    except Exception:
        return ""


def _post_json(webhook: str, payload: dict, timeout: int) -> dict:
    try:
        r = _session.post(webhook, json=payload, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = {"errcode": -1, "errmsg": f"http {r.status_code}",
                    "_raw": (r.text[:200] if r.text else "")}
        data["_http_status"] = r.status_code
        return data
    except requests.RequestException as e:
        return {"errcode": -1, "errmsg": f"network {type(e).__name__}: {e}"}


def _should_retry(resp: dict) -> bool:
    if not isinstance(resp, dict):
        return True
    code = int(resp.get("errcode", -1))
    if code == 0:
        return False
    if code == _ERR_TOO_LONG:
        return False
    if code in _ERR_RATE_LIMITS:
        return True
    if code == -1:
        return True
    # 对于 5xx 也倾向重试（由 _post_json 返回的 _http_status 可辅助判断）
    status = int(resp.get("_http_status", 0))
    if 500 <= status < 600:
        return True
    return False


def _normalize_markdown(md: str) -> str:
    """
    轻度清洗：去 BOM、统一行尾、压缩多余空行
    """
    s = (md or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()
    # 最多允许连续两行空行
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s


# ========== 分片 ==========
def _chunk_markdown(md: str, limit: int) -> List[str]:
    """
    智能切块：优先按“---”，再按空行，再按行，最后按字符。
    """
    if len(md) <= limit:
        return [md]

    chunks: List[str] = []
    pending = _normalize_markdown(md)

    # 1) 章节
    sections = pending.split("\n---\n")
    buf = ""
    for sec in sections:
        add = sec if not buf else (buf + "\n---\n" + sec)
        if len(add) <= limit:
            buf = add
        else:
            if buf:
                chunks.append(buf)
            # 2) 空段
            if len(sec) > limit:
                paragraphs = sec.split("\n\n")
                sub = ""
                for pg in paragraphs:
                    add2 = pg if not sub else (sub + "\n\n" + pg)
                    if len(add2) <= limit:
                        sub = add2
                    else:
                        if sub:
                            chunks.append(sub)
                        # 3) 行
                        if len(pg) > limit:
                            lines = pg.split("\n")
                            sub2 = ""
                            for ln in lines:
                                add3 = ln if not sub2 else (sub2 + "\n" + ln)
                                if len(add3) <= limit:
                                    sub2 = add3
                                else:
                                    if sub2:
                                        chunks.append(sub2)
                                    # 4) 字符兜底
                                    s = ln
                                    while len(s) > limit:
                                        chunks.append(s[:limit])
                                        s = s[limit:]
                                    sub2 = s or ""
                            if sub2:
                                chunks.append(sub2)
                            sub = ""
                        else:
                            sub = pg
                if sub:
                    chunks.append(sub)
            else:
                chunks.append(sec)
            buf = ""
    if buf:
        chunks.append(buf)

    # 兜底
    out: List[str] = []
    for c in chunks:
        if len(c) <= limit:
            out.append(c)
        else:
            s = c
            while len(s) > limit:
                out.append(s[:limit])
                s = s[limit:]
            if s:
                out.append(s)
    return out


def _chunk_markdown_with_margin(md: str, limit: int, margin: int) -> List[str]:
    eff = max(512, min(limit, _WECOM_MD_LIMIT) - max(0, margin))
    return _chunk_markdown(md, eff)


# ========== 发送：Markdown ==========
def send_markdown(
        content: Union[str, List[str]],
        webhook: str,
        *,
        limit: Optional[int] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        margin: int = 180,
        prefix: str | None = None,
        suffix: str | None = None,
        # 新增：内容形态增强
        title: str | None = None,
        show_header_each_part: bool = False,
        show_footer_each_part: bool = False,
        show_page_indicator: bool = True,
        header: str | None = None,
        footer: str | None = None,
        notify_userids: Optional[List[str]] = None,  # 仅 markdown：以 <@uid> 形式内联
        notify_all: bool = False,  # 仅 markdown：<@all>
) -> Tuple[bool, List[dict]]:
    """
    发送 markdown，自动分片+退避+内容增强。
    兼容旧参数，同时新增标题/页脚/@人/页码等形态。
    """
    wb = _resolve_webhook(webhook)
    if not wb:
        return False, [{"errcode": -1, "errmsg": "no webhook provided"}]

    t = _effective_timeout(timeout)
    r = _effective_retries(retries)
    lim = int(limit or _WECOM_MD_LIMIT)

    # 统一整合为一个字符串（或多段列表先 join）
    if isinstance(content, list):
        body = "\n\n---\n\n".join([_normalize_markdown(x) for x in content if x and x.strip()])
    else:
        body = _normalize_markdown(content)

    # 头/尾（仅对整体内容生效；也可以选择每分片都带）
    header_line = ""
    if title:
        header_line = f"**{title}**"
    if header:
        header_line = f"{header_line}\n{header}" if header_line else header
    footer_line = ""
    if notify_all:
        footer = (footer or "") + ("\n" if footer else "") + "<@all>"
    if notify_userids:
        mentions = " ".join(f"<@{u}>" for u in notify_userids if u)
        if mentions:
            footer = (footer or "") + ("\n" if footer else "") + mentions
    if footer:
        footer_line = footer.strip()

    # 拼接整体消息体（后再分片）
    overall = body
    if header_line:
        overall = f"{header_line}\n\n{overall}"
    if footer_line:
        overall = f"{overall}\n\n{footer_line}"

    # 分片
    src_parts = _chunk_markdown_with_margin(overall, lim, margin)
    total_parts = len(src_parts)

    resps: List[dict] = []
    all_ok = True
    part_idx = 0

    for raw_md in src_parts:
        part_idx += 1
        md_to_send = raw_md

        # 每片的“额外抬头/页脚”
        if show_header_each_part and header_line:
            md_to_send = f"{header_line}\n\n{md_to_send}"
        if show_footer_each_part and footer_line:
            md_to_send = f"{md_to_send}\n\n{footer_line}"

        # 页码标识
        if show_page_indicator and total_parts > 1:
            indicator = f"\n\n—— （{part_idx}/{total_parts}）——"
            # 保证不超限（保守裁掉末尾若超）
            if len(md_to_send) + len(indicator) <= max(512, lim - margin):
                md_to_send = md_to_send + indicator

        # 兼容 prefix/suffix
        if prefix:
            md_to_send = f"{prefix}{md_to_send}"
        if suffix:
            md_to_send = f"{md_to_send}{suffix}"

        # 发送+退避
        attempt = 0
        backoff = _BACKOFF_BASE
        cur_limit = lim - max(0, margin)

        while True:
            attempt += 1
            payload = {"msgtype": "markdown", "markdown": {"content": md_to_send}}
            resp = _post_json(wb, payload, timeout=t)
            resp["_part"] = part_idx
            resp["_ok"] = (resp.get("errcode") == 0)
            resps.append(resp)

            if resp["_ok"]:
                break

            code = int(resp.get("errcode", -1))
            if code == _ERR_TOO_LONG and len(md_to_send) > 512:
                # 收紧 15% 重切
                cur_limit = max(512, int(cur_limit * 0.85))
                smaller_chunks = _chunk_markdown(md_to_send, cur_limit)
                if len(smaller_chunks) > 1:
                    insert_at = part_idx
                    for extra in reversed(smaller_chunks[1:]):
                        src_parts.insert(insert_at, extra)
                    md_to_send = smaller_chunks[0]
                    continue

            if attempt <= r and _should_retry(resp):
                time.sleep(backoff)
                backoff = min(_BACKOFF_MAX, backoff * 2)
                continue

            all_ok = False
            break

    return all_ok, resps


# ========== 发送：图片 ==========
def send_image_from_path(
        path: Union[str, Path],
        webhook: str,
        *,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
) -> Tuple[bool, dict]:
    """
    企微图片接口需要 base64+md5；带退避。
    """
    wb = _resolve_webhook(webhook)
    if not wb:
        return False, {"errcode": -1, "errmsg": "no webhook provided"}

    p = Path(path)
    try:
        data = p.read_bytes()
    except Exception as e:
        return False, {"errcode": -1, "errmsg": f"read file error: {e}"}

    b64 = base64.b64encode(data).decode("utf-8")
    md5 = hashlib.md5(data).hexdigest()

    t = _effective_timeout(timeout)
    r = _effective_retries(retries)

    payload = {"msgtype": "image", "image": {"base64": b64, "md5": md5}}

    attempt = 0
    backoff = _BACKOFF_BASE
    last = {}
    while True:
        attempt += 1
        resp = _post_json(wb, payload, timeout=t)
        resp["_ok"] = (resp.get("errcode") == 0)
        last = resp
        if resp["_ok"]:
            return True, resp
        if attempt <= r and _should_retry(resp):
            time.sleep(backoff)
            backoff = min(_BACKOFF_MAX, backoff * 2)
            continue
        return False, last


# ========== 发送：text（可 @ 人）==========
def send_text(
        content: str,
        webhook: str,
        *,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        at_mobiles: Optional[List[str]] = None,
        at_userids: Optional[List[str]] = None,
        at_all: bool = False,
) -> Tuple[bool, dict]:
    """
    纯文本推送；支持 @手机号 / @userid / @all。
    注意：text 的 @ 通过 payload 字段控制；markdown 请用 <@uid> 内联。
    """
    wb = _resolve_webhook(webhook)
    if not wb:
        return False, {"errcode": -1, "errmsg": "no webhook provided"}

    t = _effective_timeout(timeout)
    r = _effective_retries(retries)

    payload: Dict[str, Any] = {
        "msgtype": "text",
        "text": {
            "content": content or "",
            "mentioned_mobile_list": at_mobiles or [],
            "mentioned_list": (["@all"] if at_all else (at_userids or [])),
        },
    }

    attempt = 0
    backoff = _BACKOFF_BASE
    while True:
        attempt += 1
        resp = _post_json(wb, payload, timeout=t)
        ok = (resp.get("errcode") == 0)
        if ok:
            return True, resp
        if attempt <= r and _should_retry(resp):
            time.sleep(backoff)
            backoff = min(_BACKOFF_MAX, backoff * 2)
            continue
        return False, resp


# ========== 发送：news（图文卡片）==========
def send_news(
        articles: List[Dict[str, str]],
        webhook: str,
        *,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
) -> Tuple[bool, dict]:
    """
    发送图文消息（最多 8 条）。
    article: {"title": "...", "description": "...", "url": "https://...", "picurl": "https://..."}
    """
    wb = _resolve_webhook(webhook)
    if not wb:
        return False, {"errcode": -1, "errmsg": "no webhook provided"}

    arts = []
    for a in (articles or [])[:8]:
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not title or not url:
            continue
        arts.append({
            "title": title[:128],
            "description": (a.get("description") or "").strip()[:512],
            "url": url[:1024],
            "picurl": (a.get("picurl") or "").strip()[:1024],
        })
    if not arts:
        return False, {"errcode": -1, "errmsg": "no valid article"}

    t = _effective_timeout(timeout)
    r = _effective_retries(retries)

    payload = {"msgtype": "news", "news": {"articles": arts}}

    attempt = 0
    backoff = _BACKOFF_BASE
    while True:
        attempt += 1
        resp = _post_json(wb, payload, timeout=t)
        ok = (resp.get("errcode") == 0)
        if ok:
            return True, resp
        if attempt <= r and _should_retry(resp):
            time.sleep(backoff)
            backoff = min(_BACKOFF_MAX, backoff * 2)
            continue
        return False, resp


# ========== 发送：file（通过 webhook 上传获取 media_id）==========
def _upload_file_to_webhook(file_path: Union[str, Path], webhook: str, timeout: int) -> Tuple[bool, dict]:
    """
    通过 webhook 的 upload_media 接口上传文件以获取 media_id：
    POST https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=KEY&type=file
    """
    key = _extract_key_from_webhook(webhook)
    if not key:
        return False, {"errcode": -1, "errmsg": "cannot extract webhook key for upload"}

    url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type=file"
    p = Path(file_path)
    if not p.exists():
        return False, {"errcode": -1, "errmsg": "file not found"}

    mime, _ = mimetypes.guess_type(str(p))
    files = {"file": (p.name, p.read_bytes(), mime or "application/octet-stream")}
    try:
        r = _session.post(url, files=files, timeout=timeout)
        data = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {"raw": r.text}
        if r.status_code == 200 and data.get("errcode") == 0 and data.get("media_id"):
            return True, data
        return False, data
    except requests.RequestException as e:
        return False, {"errcode": -1, "errmsg": f"upload error: {e}"}


def send_file_from_path(
        path: Union[str, Path],
        webhook: str,
        *,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
) -> Tuple[bool, dict]:
    """
    发送文件（如生成的 JSON/Markdown/PDF 等）到群。
    """
    wb = _resolve_webhook(webhook)
    if not wb:
        return False, {"errcode": -1, "errmsg": "no webhook provided"}

    t = _effective_timeout(timeout)
    r = _effective_retries(retries)

    attempt = 0
    backoff = _BACKOFF_BASE
    up_ok = False
    up_data: dict = {}
    while True:
        attempt += 1
        up_ok, up_data = _upload_file_to_webhook(path, wb, timeout=t)
        if up_ok:
            break
        if attempt <= r:
            time.sleep(backoff)
            backoff = min(_BACKOFF_MAX, backoff * 2)
            continue
        return False, up_data

    media_id = up_data.get("media_id")
    payload = {"msgtype": "file", "file": {"media_id": media_id}}

    attempt = 0
    backoff = _BACKOFF_BASE
    last = {}
    while True:
        attempt += 1
        resp = _post_json(wb, payload, timeout=t)
        resp["_ok"] = (resp.get("errcode") == 0)
        last = resp
        if resp["_ok"]:
            return True, resp
        if attempt <= r and _should_retry(resp):
            time.sleep(backoff)
            backoff = min(_BACKOFF_MAX, backoff * 2)
            continue
        return False, last


# ========== 内容构建助手（可选用）==========
def format_broadcast_markdown(
        *,
        title: str,
        timestamp_line: str,
        overview: Optional[str] = None,
        bullets: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    统一风格的“群播报” Markdown，适配企微渲染。
    - bullets: 每条已不带前缀，函数中会加 "• "
    - tags: 自动前置 # 并空格分隔
    - links: (title, url)
    """
    lines: List[str] = []
    lines.append(f"**{title}**")
    lines.append(timestamp_line)
    lines.append("")

    if overview:
        lines.append("> **AI 摘要**")
        lines.append(overview.strip())
        lines.append("")

    if bullets:
        lines.append("**要点速览**")
        for b in bullets:
            s = (b or "").strip()
            if s:
                lines.append(f"• {s}")
        lines.append("")

    if tags:
        tags_norm = [f"#{t.strip().lstrip('#')}" for t in tags if t and t.strip()]
        if tags_norm:
            lines.append("**话题**  " + " ".join(tags_norm))
            lines.append("")

    if links:
        lines.append("**原文链接**  ")
        for i, (title, url) in enumerate(links, 1):
            if title and url:
                lines.append(f"{i}. [{title}]({url})")

    md = "\n".join(lines).rstrip()
    return _normalize_markdown(md)


# ========== 两榜内容构建与一键推送 ==========
_BADGE_TEXT = {"新": "〔新〕", "爆": "〔爆〕", "沸": "〔沸〕"}


def _fmt_board_lines(
        name: str,
        items: List[Dict[str, Any]],
        *,
        topn: int = 10,
        with_links: bool = True,
        show_badge: bool = True,
) -> List[str]:
    """渲染单个榜区为行列表（带序号/可选链接/徽章文案）"""
    lines: List[str] = []
    if not items:
        return [f"**微博·{name}**  \n（暂无数据）"]
    lines.append(f"**微博·{name}**")
    for i, it in enumerate(items[: max(1, topn)], 1):
        title = str(it.get("title") or "").strip()
        url = str(it.get("url") or "").strip()
        label = str(it.get("label") or "").strip()
        badge = _BADGE_TEXT.get(label, "")
        if with_links and title and url:
            line = f"{i}. [{title}]({url})"
        else:
            line = f"{i}. {title}"
        if show_badge and badge:
            line = f"{line} {badge}"
        lines.append(line)
    return lines


def format_two_boards_markdown(
        hot_items: List[Dict[str, Any]] | None,
        social_items: List[Dict[str, Any]] | None,
        *,
        topn: int = 10,
        summary_md: str | None = None,
        tags: List[str] | None = None,
) -> str:
    """
    组合【热搜榜 + 社会榜】为一段 Markdown（仅正文，不含题头/页脚）。
    - summary_md：可传 AI 摘要（例如 summary_deepseek 的输出），会置顶显示
    - tags：#话题 列表（自动加 #）
    """
    blocks: List[str] = []

    if summary_md:
        blocks.append("> **AI 摘要**\n" + _normalize_markdown(summary_md))

    # 两榜分区
    hot_lines = _fmt_board_lines("热搜榜", hot_items or [], topn=topn, with_links=True, show_badge=True)
    social_lines = _fmt_board_lines("社会榜", social_items or [], topn=topn, with_links=True, show_badge=True)
    blocks.append("\n".join(hot_lines))
    blocks.append("\n".join(social_lines))

    # 话题
    if tags:
        tags_norm = " ".join(f"#{t.strip().lstrip('#')}" for t in tags if t and t.strip())
        if tags_norm:
            blocks.append(f"**话题**  {tags_norm}")

    # 用 “---” 做分隔，便于 send_markdown 智能分页
    return "\n\n---\n\n".join(b for b in blocks if b and b.strip())


def _now_timestamp_line() -> str:
    """生成形如：YYYY年MM月DD日 HH:MM 星期X 的时间行（按 .env TZ）"""
    now = datetime.now(tzinfo())
    weekday = WEEKDAY_CN[(now.isoweekday() - 1) % 7]
    return now.strftime("%Y年%m月%d日 %H:%M ") + weekday


def send_two_boards_digest(
        hot_items: List[Dict[str, Any]] | None,
        social_items: List[Dict[str, Any]] | None,
        webhook: str,
        *,
        topn: int = 10,
        summary_md: str | None = None,
        tags: List[str] | None = None,
        # 样式 & 行为透传到 send_markdown
        title: str | None = None,
        timestamp_line: str | None = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        show_header_each_part: bool = False,
        show_footer_each_part: bool = False,
        show_page_indicator: bool = True,
        notify_userids: Optional[List[str]] = None,
        notify_all: bool = False,
) -> Tuple[bool, List[dict]]:
    """
    组合两榜 +（可选）AI 摘要，一键推送到企微。
    默认标题：`微博两大榜｜{YYYY年MM月DD日 HH:MM 星期X}`
    默认时间行：使用 .env 的 TZ 生成。
    """
    body = format_two_boards_markdown(hot_items, social_items, topn=topn, summary_md=summary_md, tags=tags)

    # 题头与时间行
    title = title or f"微博两大榜｜{_now_timestamp_line()}"
    timestamp_line = timestamp_line or _now_timestamp_line()

    return send_markdown(
        content=body,
        webhook=webhook,
        timeout=timeout,
        retries=retries,
        # 样式增强
        title=title,
        header=timestamp_line,
        show_header_each_part=show_header_each_part,
        show_footer_each_part=show_footer_each_part,
        show_page_indicator=show_page_indicator,
        notify_userids=notify_userids,
        notify_all=notify_all,
    )


__all__ = [
    "send_markdown",
    "send_image_from_path",
    "send_text",
    "send_news",
    "send_file_from_path",
    "format_broadcast_markdown",
    # 新增
    "format_two_boards_markdown",
    "send_two_boards_digest",
]
