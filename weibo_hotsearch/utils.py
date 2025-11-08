# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午2:10
# @Author: 柯影数智
# @File: utils.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm


"""
weibo_hotsearch.utils — 常用工具集合
依赖：requests, Pillow(可选), urllib3(可选), selenium(仅截图模块用到时)
配置：统一从 weibo_hotsearch.config.CONFIG 读取
"""

from __future__ import annotations

import os
import re
import io
import json
import time
import html
import base64
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from .config import CONFIG, WEEKDAY_CN, now_local

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )


# =========================
# HTTP 会话 / 重试
# =========================

def make_session(timeout: float = 8.0, retries: int = 2) -> requests.Session:
    """
    创建带重试策略的 Session；urllib3 不可用时回退为普通 Session。
    """
    sess = requests.Session()
    sess.headers.update(CONFIG.weibo_headers())
    try:
        from requests.adapters import HTTPAdapter  # type: ignore
        try:
            from urllib3.util.retry import Retry  # type: ignore
            retry = Retry(
                total=retries,
                connect=retries,
                read=retries,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "POST", "HEAD"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
        except Exception:
            adapter = HTTPAdapter()
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
    except Exception:
        # 最简回退：不挂 Adapter
        pass
    sess.request = _wrap_timeout(sess.request, timeout)  # type: ignore
    return sess


def _wrap_timeout(request_func, default_timeout: float):
    """
    包装 requests.Session.request，统一超时默认值。
    """

    def wrapper(method, url, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = default_timeout
        return request_func(method, url, **kwargs)

    return wrapper


# =========================
# 微博抓取与解析
# =========================

def fetch_weibo_json(sess: requests.Session | None = None) -> Optional[Dict[str, Any]]:
    """
    访问官方 JSON 接口：WEIBO_API
    成功返回 dict（含 data），失败返回 None
    """
    sess = sess or make_session(timeout=CONFIG.request_timeout_sec, retries=CONFIG.request_retries)
    url = CONFIG.weibo_api
    try:
        r = sess.get(url, headers={"Accept": "application/json, text/plain, */*"})
        if r.status_code == 200:
            data = r.json().get("data", {}) or {}
            if data.get("realtime"):
                return data
        # 403/418/451 等大概率 Cookie 过期或风控
        logger.debug("fetch_weibo_json status=%s body_head=%s", r.status_code, r.text[:200])
    except Exception as e:
        logger.warning("fetch_weibo_json error: %s", e)
    return None


def fetch_weibo_summary_html(sess: requests.Session | None = None) -> Optional[str]:
    """
    抓取 summary 页面 HTML（作为 JSON 失败的兜底）
    """
    sess = sess or make_session(timeout=CONFIG.request_timeout_sec, retries=CONFIG.request_retries)
    url = CONFIG.weibo_summary
    try:
        r = sess.get(
            url,
            headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        )
        if r.status_code == 200 and ("实时热点" in r.text or "<table" in r.text):
            return r.text
        logger.debug("fetch_weibo_summary_html status=%s", r.status_code)
    except Exception as e:
        logger.warning("fetch_weibo_summary_html error: %s", e)
    return None


def parse_weibo_html(html_text: str) -> List[Dict[str, Any]]:
    """
    解析 summary HTML，提取列表项
    返回 item 列表：{title, hot, label, url, category}
    """
    rows = re.findall(r"<tr.*?>.*?</tr>", html_text, flags=re.S | re.I)
    items: List[Dict[str, Any]] = []
    for row in rows:
        m_title = re.search(r'<td[^>]*class="td-02"[^>]*>.*?<a[^>]*>(.*?)</a>', row, flags=re.S | re.I)
        if not m_title:
            continue
        raw = re.sub(r"<.*?>", "", m_title.group(1), flags=re.S)
        title = _normalize_title(raw)
        if not title:
            continue

        label = ""
        if re.search(r'icon[\s-]*new', row, flags=re.I):
            label = "新"
        elif re.search(r'icon[\s-]*hot', row, flags=re.I):
            label = "爆"
        elif re.search(r'icon[\s-]*fei', row, flags=re.I) or "沸" in row:
            label = "沸"

        url = search_url_for_title(title)
        items.append({"title": title, "hot": 0, "label": label, "url": url, "category": ""})
    return items


def normalize_from_json(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    将 JSON data 规整为 (hotgov, items)；items：{title, hot, label, url, category}
    """
    hotgov = (data or {}).get("hotgov") or {}
    realtime = (data or {}).get("realtime") or []
    items: List[Dict[str, Any]] = []
    for x in realtime:
        title = _normalize_title(x.get("note") or x.get("word") or x.get("title") or "")
        if not title:
            continue
        hot = _to_int(x.get("num") or x.get("hot") or 0)
        label = str(x.get("label_name") or "").strip()
        label = label if label in ("新", "爆", "沸") else ""
        url = x.get("scheme") or search_url_for_title(title)
        items.append({
            "title": title,
            "hot": hot,
            "label": label,
            "url": url,
            "category": x.get("category") or ""
        })
    items = dedupe_and_sort(items)
    return hotgov, items


def search_url_for_title(title: str) -> str:
    return f"https://s.weibo.com/weibo?q={quote('#' + title + '#')}"


def _normalize_title(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def is_tech_item(item: Dict[str, Any], keywords: Optional[List[str]] = None) -> bool:
    """
    判断是否科技类：优先看 category；否则看标题是否命中关键词
    """
    keywords = keywords or CONFIG.tech_keywords
    cat = (item.get("category") or "").lower()
    if any(k in cat for k in ["科技", "数码", "互联网", "it"]):
        return True
    title_up = (item.get("title") or "").upper()
    return any(k.upper() in title_up for k in keywords)


def filter_tech(items: List[Dict[str, Any]], keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    return [it for it in items if is_tech_item(it, keywords)]


def dedupe_and_sort(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去重（按标题）并按热度降序、标题升序排序
    """
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in items:
        t = it.get("title")
        if t and t not in seen:
            seen.add(t)
            uniq.append(it)
    uniq.sort(key=lambda d: (-int(d.get("hot", 0)), d.get("title", "")))
    return uniq


# =========================
# 时间与格式化
# =========================

def format_time_line() -> str:
    """
    返回图像/文案中使用的时间行，如：2025年 11月 07日 13:00  星期五
    """
    t = now_local()
    wd = WEEKDAY_CN[(t.isoweekday() - 1) % 7]
    return t.strftime("20%y年 %m月 %d日 %H:%M ") + f"{wd}"


def format_filename_png() -> str:
    """
    返回文件名，如：2025年11月07日13:00.png
    """
    t = now_local()
    return t.strftime("20%y年%m月%d日%H:%M.png")


# =========================
# 图片/绘制工具（Pillow 依赖）
# =========================

def ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


def load_font(path: Optional[str], size: int):
    """
    加载字体；缺失时回退到 PIL 默认字体。
    """
    try:
        from PIL import ImageFont  # type: ignore
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()
    except Exception:
        # 未安装 Pillow 时的兜底（返回 None，上层应避免绘制）
        return None


def wrap_text(draw, text: str, font, max_width: int, max_lines: int = 2, ellipsis: str = "…") -> List[str]:
    """
    基于像素宽度的自动换行（中文逐字处理）
    """
    if not text:
        return [""]
    try:
        # Pillow 存在
        words = list(text)
        lines: List[str] = []
        cur = ""
        for idx, ch in enumerate(words):
            test = cur + ch
            w = draw.textlength(test, font=font)
            if w <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = ch
                if len(lines) >= max_lines - 1:
                    tail = ""
                    for c in words[idx:]:
                        if draw.textlength(tail + c + ellipsis, font=font) <= max_width:
                            tail += c
                        else:
                            break
                    lines.append(tail + (ellipsis if tail else ellipsis))
                    return lines
        if cur:
            lines.append(cur)
        return lines[:max_lines]
    except Exception:
        # 没有 Pillow 时的简单回退：按字符宽度近似
        width_chars = max(1, max_width // 16)
        wrapped = []
        s = text
        while s and len(wrapped) < max_lines:
            wrapped.append(s[:width_chars])
            s = s[width_chars:]
        if s:
            wrapped[-1] = wrapped[-1][:-1] + ellipsis
        return wrapped


def draw_badge(draw, xy: Tuple[int, int, int, int], text: str, font, fill, text_fill):
    """
    画圆角徽标（用于 “新/爆/沸”）
    """
    try:
        draw.rounded_rectangle(xy, radius=(xy[3] - xy[1]) // 2, fill=fill)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        tx = xy[0] + (xy[2] - xy[0] - w) // 2
        ty = xy[1] + (xy[3] - xy[1] - h) // 2 - 1
        draw.text((tx, ty), text, font=font, fill=text_fill)
    except Exception:
        # Pillow 不可用则忽略徽标
        pass


# =========================
# 企业微信发送
# =========================

def send_wecom_markdown(content: str) -> Tuple[bool, Dict[str, Any]]:
    """
    发送 markdown 到企业微信群机器人；需要在 .env 设置 WECOM_WEBHOOK_KEY（URL 或 key）
    """
    url = CONFIG.wecom_webhook_url()
    if not url:
        return False, {"errcode": -1, "errmsg": "wecom webhook not configured"}
    payload = {"msgtype": "markdown", "markdown": {"content": content}}
    try:
        r = requests.post(url, json=payload, timeout=10)
        data = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {"raw": r.text}
        ok = (r.status_code == 200) and (data.get("errcode") == 0)
        return ok, data
    except Exception as e:
        return False, {"errcode": -1, "errmsg": str(e)}


def send_wecom_image(image_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    发送图片到企业微信群机器人（走 image base64+md5）
    """
    url = CONFIG.wecom_webhook_url()
    if not url:
        return False, {"errcode": -1, "errmsg": "wecom webhook not configured"}
    try:
        with open(image_path, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode()
        md5 = hashlib.md5(content).hexdigest()
        payload = {"msgtype": "image", "image": {"base64": b64, "md5": md5}}
        r = requests.post(url, json=payload, timeout=10)
        data = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {"raw": r.text}
        ok = (r.status_code == 200) and (data.get("errcode") == 0)
        return ok, data
    except Exception as e:
        return False, {"errcode": -1, "errmsg": str(e)}


def build_markdown(items: List[Dict[str, Any]], title_prefix: str = "微博热搜 TopN",
                   summary_text: Optional[str] = None, tech_only: bool = False) -> str:
    """
    构造企微 markdown 文本
    """
    ts = now_local().strftime("20%y年%m月%d日 %H:%M ")
    wd = WEEKDAY_CN[(now_local().isoweekday() - 1) % 7]
    title = "微博科技热搜" if tech_only else "微博热搜"
    head = f"**{title}｜{ts}{wd}**"
    lines = [head, ""]
    if summary_text:
        lines.append(summary_text.strip())
        lines.append("\n——")
    lines.append("**原文链接**  ")
    for i, it in enumerate(items, 1):
        tag = f" **{it.get('label')}**" if it.get("label") else ""
        lines.append(f"{i}. [{it.get('title')}]({it.get('url')}){tag}")
    return "\n".join(lines)


# =========================
# DeepSeek 摘要
# =========================

def deepseek_summary(items: List[Dict[str, Any]], max_words: int = 400) -> Optional[str]:
    """
    使用 DeepSeek（OpenAI 兼容 API）生成中文摘要。
    依赖 .env: DEEPSEEK_API_KEY，base_url 使用 CONFIG.deepseek_base_url
    """
    api_key = CONFIG.deepseek_api_key
    if not api_key:
        return None

    # 优先用 openai SDK；不可用则退回 requests
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key, base_url=CONFIG.deepseek_base_url)
        payload = {
            "model": CONFIG.deepseek_model,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": "你是资深科技编辑，请将热搜词条整理为中文摘要。"},
                {"role": "user", "content": (
                        "请基于如下JSON(含title,url,hot,label)生成：\n"
                        "1) 10条要点：每条7-20字；\n"
                        "2) 一段50-80字趋势点评；\n"
                        "3) 3-6个 #话题标签；\n"
                        "4) 最后列出原文链接清单；\n"
                        f"限制总体不超过{max_words}字。\n\nJSON:\n" + json.dumps(items, ensure_ascii=False, indent=2)
                )}
            ],
        }
        resp = client.chat.completions.create(**payload)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # requests 回退
        try:
            url = f"{CONFIG.deepseek_base_url.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": CONFIG.deepseek_model,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": "你是资深科技编辑，请将热搜词条整理为中文摘要。"},
                    {"role": "user", "content": (
                            "请基于如下JSON(含title,url,hot,label)生成：\n"
                            "1) 10条要点：每条7-20字；\n"
                            "2) 一段50-80字趋势点评；\n"
                            "3) 3-6个 #话题标签；\n"
                            "4) 最后列出原文链接清单；\n"
                            f"限制总体不超过{max_words}字。\n\nJSON:\n" + json.dumps(items, ensure_ascii=False, indent=2)
                    )}
                ],
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip() or None
        except Exception as e:
            logger.warning("deepseek_summary error: %s", e)
    return None


# =========================
# 组合抓取（多层兜底）
# =========================

def fetch_any(selenium_fallback: bool = False, driver_path: Optional[str] = None) -> Dict[str, Any]:
    """
    三层兜底抓取：
      1) JSON 接口
      2) summary HTML + requests 解析
      3) 可选 Selenium 渲染解析（慢，谨慎使用）
    返回规范化结构：{"hotgov": {...}, "realtime": [items...]}
    """
    sess = make_session(timeout=CONFIG.request_timeout_sec, retries=CONFIG.request_retries)

    # 1) JSON
    data = fetch_weibo_json(sess=sess)
    if data and data.get("realtime"):
        hotgov, items = normalize_from_json(data)
        return {"hotgov": hotgov, "realtime": items}

    # 2) HTML
    html_text = fetch_weibo_summary_html(sess=sess)
    if html_text:
        items = parse_weibo_html(html_text)
        items = dedupe_and_sort(items)
        return {"hotgov": {}, "realtime": items}

    # 3) Selenium（可选）
    if selenium_fallback and driver_path:
        try:
            from selenium import webdriver  # type: ignore
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            opts = ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--lang=zh-CN,zh;q=0.9,en;q=0.8")
            opts.add_argument(f"--user-agent={CONFIG.head_ua}")
            opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            opts.add_experimental_option("useAutomationExtension", False)

            service = ChromeService(executable_path=driver_path)
            driver = webdriver.Chrome(service=service, options=opts)
            try:
                driver.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"}
                )
                driver.set_window_size(1366, 2200)
                driver.get(CONFIG.weibo_summary)

                WebDriverWait(driver, 25).until(
                    EC.any_of(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table tbody tr")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#pl_top_realtimehot"))
                    )
                )
                html_src = driver.page_source
            finally:
                driver.quit()
            items = parse_weibo_html(html_src)
            items = dedupe_and_sort(items)
            return {"hotgov": {}, "realtime": items}
        except Exception as e:
            logger.error("selenium fallback failed: %s", e)

    return {}


__all__ = [
    # HTTP
    "make_session",
    # 微博
    "fetch_weibo_json", "fetch_weibo_summary_html", "parse_weibo_html",
    "normalize_from_json", "filter_tech", "is_tech_item", "dedupe_and_sort",
    "fetch_any", "search_url_for_title",
    # 时间
    "format_time_line", "format_filename_png",
    # 图片
    "ensure_dir", "load_font", "wrap_text", "draw_badge",
    # DeepSeek
    "deepseek_summary",
    # 企微
    "send_wecom_markdown", "send_wecom_image", "build_markdown",
]
