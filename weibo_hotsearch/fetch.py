# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午1:23
# @Author: 柯影数智
# @File: fetch.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

"""
抓取核心：
- fetch_hot_search：单榜（兼容原逻辑）
- fetch_boards：三大榜（热搜/生活/社会）统一抓取，HTML 直解析 + 可选 Selenium 兜底
"""

from __future__ import annotations

import json
import argparse
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import requests

try:
    # 包方式运行优先
    if __package__:
        from .config import CONFIG, WEEKDAY_CN, now_local
        from .utils import (
            make_session, fetch_weibo_json, fetch_weibo_summary_html,
            normalize_from_json, filter_tech, dedupe_and_sort,
            format_time_line, build_markdown,
        )
    else:
        raise ImportError
except Exception:
    # 直跑兜底
    import os, sys

    PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PKG_ROOT not in sys.path:
        sys.path.insert(0, PKG_ROOT)
    from weibo_hotsearch.config import CONFIG, WEEKDAY_CN, now_local
    from weibo_hotsearch.utils import (
        make_session, fetch_weibo_json, fetch_weibo_summary_html,
        normalize_from_json, filter_tech, dedupe_and_sort,
        format_time_line, build_markdown,
    )

REQUEST_TIMEOUT = float(getattr(CONFIG, "request_timeout_sec", 8.0))
REQUEST_RETRIES = int(getattr(CONFIG, "request_retries", 2))


# ------------------------
# 内部：更细粒度 JSON 抓取（带状态码）
# ------------------------
def _fetch_json_with_status(sess: requests.Session) -> Tuple[Optional[Dict[str, Any]], int, str]:
    url = CONFIG.weibo_api
    try:
        r = sess.get(url, headers={
            "Accept": "application/json, text/plain, */*",
            "User-Agent": CONFIG.head_ua,
        })
        if r.status_code == 200:
            try:
                data = r.json().get("data", {}) or {}
            except Exception:
                data = {}
            return (data if data.get("realtime") else None, r.status_code, r.text[:200])
        else:
            return (None, r.status_code, r.text[:200] if r.text else "")
    except Exception as e:
        return (None, -1, str(e))


def _make_hint_from_status(status: int, text_head: str) -> str:
    if status == 200:
        return ""
    if status in (401, 403, 418, 451):
        return f"微博接口被拒绝访问（可能 Cookie 过期/未登录/风控）。（HTTP {status}）请在 .env 设置/更新 WEIBO_COOKIE，然后重试；必要时使用 --selenium-fallback 兜底。"
    if status == 429:
        return "请求过于频繁（HTTP 429）。建议降低频率或稍后再试。"
    if status in (500, 502, 503, 504):
        return f"微博服务端暂时异常（HTTP {status}）。建议稍后重试。"
    if status == -1:
        return f"请求异常：{text_head}"
    return f"微博接口返回 HTTP {status}。可尝试设置 WEIBO_COOKIE 或使用 --selenium-fallback。"


# ------------------------
# 单榜：兼容原接口
# ------------------------
def fetch_hot_search(
        topn: int = 10,
        tech_only: bool = False,
        selenium_fallback: bool = False,
        driver_path: Optional[str] = None,
) -> Dict[str, Any]:
    sess = make_session(timeout=REQUEST_TIMEOUT, retries=REQUEST_RETRIES)
    hint = ""
    source = "none"
    pinned = None
    items: List[Dict[str, Any]] = []

    # 1) JSON
    data, status, head = _fetch_json_with_status(sess)
    if data and data.get("realtime"):
        pinned, items = normalize_from_json(data)
        source = "json"
    else:
        hint = _make_hint_from_status(status, head)
        # 2) HTML
        html_text = fetch_weibo_summary_html(sess)
        if html_text:
            items = _parse_weibo_html_simple(html_text)
            source = "html"
        else:
            # 3) Selenium 兜底
            if selenium_fallback and driver_path:
                try:
                    html_src = _selenium_get_html(driver_path)
                    items = _parse_weibo_html_simple(html_src)
                    source = "selenium"
                except Exception as e:
                    hint = (hint + "；" if hint else "") + f"Selenium 兜底失败：{e}"

    if tech_only:
        items = filter_tech(items)
    items = items[:topn]

    return {
        "time": now_local().isoformat(),
        "topn": topn,
        "tech_only": tech_only,
        "source": source,
        "hint": hint,
        "pinned": (pinned.get("word").strip("#") if isinstance(pinned, dict) and pinned.get("word") else None),
        "items": items,
    }


def _parse_weibo_html_simple(html: str) -> List[Dict[str, Any]]:
    """
    兼容原 parse：只取主榜（热搜榜）的一张表。
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        # 极简正则兜底：抓取链接文字（不如 bs4 稳健）
        pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>([^<]{2,60})</a>')
        items = []
        for m in pattern.finditer(html):
            url, title = m.group(1), m.group(2)
            if "s.weibo.com/weibo?q=" in url:
                items.append({"title": title.strip(), "url": url, "label": "", "category": "热搜榜"})
                if len(items) >= 50:
                    break
        return items

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table tbody")
    items: List[Dict[str, Any]] = []
    if not table:
        return items

    for tr in table.select("tr"):
        a = tr.select_one(".td-02 a")
        if not a or not a.get("href"):
            continue
        title = (a.get_text(strip=True) or "").replace("\xa0", " ").strip()
        url = a.get("href")
        label = ""
        # 标签图标文字
        hot_tag = tr.select_one(".td-02 .icon-hot, .td-02 .icon-fried, .td-02 .icon-new")
        if hot_tag and hot_tag.get_text(strip=True):
            label = hot_tag.get_text(strip=True)[:1]
        # 热度
        hot_text = tr.select_one(".td-03")
        try:
            hot = int(re.sub(r"\D", "", hot_text.get_text()) or "0") if hot_text else 0
        except Exception:
            hot = 0
        items.append({
            "title": title,
            "url": url,
            "label": label,
            "hot": hot,
            "category": "热搜榜",
        })
    return items


def _selenium_get_html(driver_path: str) -> str:
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
        driver.set_window_size(1366, 2400)
        driver.get(CONFIG.weibo_summary)
        WebDriverWait(driver, 25).until(
            EC.any_of(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table tbody tr")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "#pl_top_realtimehot"))
            )
        )
        return driver.page_source
    finally:
        driver.quit()


# ------------------------
# 多榜：热搜/生活/社会
# ------------------------
def fetch_boards(
        topn: int = 10,
        selenium_fallback: bool = False,
        driver_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从 summary 页面同时解析三张表（或可视区域的三块）：
    返回 { "热搜榜": [...], "生活榜": [...], "社会榜": [...] }
    """
    sess = make_session(timeout=REQUEST_TIMEOUT, retries=REQUEST_RETRIES)
    html = fetch_weibo_summary_html(sess)
    source = "html"
    if not html and selenium_fallback and driver_path:
        try:
            html = _selenium_get_html(driver_path)
            source = "selenium"
        except Exception:
            html = ""

    boards = _parse_boards_from_html(html) if html else OrderedDict([("热搜榜", []), ("生活榜", []), ("社会榜", [])])

    # TopN 裁剪
    for k in list(boards.keys()):
        boards[k] = boards[k][:topn]
    return boards


def _parse_boards_from_html(html: str) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """
    解析 summary 页面中出现的多张表：尽力匹配“热搜榜 / 生活榜 / 社会榜”。
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        # 简易兜底：只解析主榜
        return OrderedDict([("热搜榜", _parse_weibo_html_simple(html)), ("生活榜", []), ("社会榜", [])])

    soup = BeautifulSoup(html, "html.parser")
    # 取所有 table，尝试从相邻标题文本推断板块名
    tables = soup.select("table")
    out: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict([("热搜榜", []), ("生活榜", []), ("社会榜", [])])

    def _guess_board_name(tbl) -> str:
        # 在 table 之前找最近的标题元素
        header = tbl.find_previous(lambda tag: tag.name in ("h2", "h3", "h4", "strong"))
        text = (header.get_text(separator="", strip=True) if header else "") or ""
        # 常见命名匹配
        if "生活" in text: return "生活榜"
        if "社会" in text: return "社会榜"
        if "热搜" in text or "实时" in text or "榜" in text: return "热搜榜"
        # 再尝试在父容器内找
        parent = tbl.parent
        for _ in range(3):
            if not parent: break
            t = parent.get_text(separator=" ", strip=True) if hasattr(parent, "get_text") else ""
            if "生活" in t: return "生活榜"
            if "社会" in t: return "社会榜"
            parent = parent.parent
        return "热搜榜"

    for tbl in tables:
        tbody = tbl.select_one("tbody")
        if not tbody:  # 页面里可能有其他表
            continue
        board = _guess_board_name(tbl)
        for tr in tbody.select("tr"):
            a = tr.select_one(".td-02 a")
            if not a or not a.get("href"):
                continue
            title = (a.get_text(strip=True) or "").replace("\xa0", " ").strip()
            url = a.get("href")
            # 标签
            label = ""
            hot_tag = tr.select_one(".td-02 .icon-hot, .td-02 .icon-fried, .td-02 .icon-new")
            if hot_tag and hot_tag.get_text(strip=True):
                label = hot_tag.get_text(strip=True)[:1]
            # 热度
            hot_text = tr.select_one(".td-03")
            try:
                hot = int(re.sub(r"\D", "", hot_text.get_text()) or "0") if hot_text else 0
            except Exception:
                hot = 0
            out[board].append({
                "title": title,
                "url": url,
                "label": label,
                "hot": hot,
                "category": board
            })
    # 基本去重（同标题保留热度高者）
    for k, v in out.items():
        seen = {}
        for it in v:
            t = it["title"]
            if t not in seen or it.get("hot", 0) > seen[t].get("hot", 0):
                seen[t] = it
        out[k] = list(seen.values())
    return out


# ------------------------
# CLI（单文件调试用）
# ------------------------
def _render_plain(payload: Dict[str, Any]) -> str:
    ts = now_local().strftime("微博热搜榜 20%y年%m月%d日 %H:%M ")
    wd = WEEKDAY_CN[(now_local().isoweekday() - 1) % 7]
    lines = [f"{ts}{wd}（来源：{payload.get('source', '-')}）"]
    if payload.get("pinned"):
        lines.append(f"置顶: {payload['pinned']}")
    for i, it in enumerate(payload.get("items", []), 1):
        tag = f" {it.get('label')}" if it.get("label") else ""
        lines.append(f"{i}. {it.get('title')}{tag} 链接：{it.get('url')}")
    if payload.get("hint"):
        lines.append(f"\n提示：{payload['hint']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="抓取微博热搜（单榜/多榜）")
    ap.add_argument("--mode", choices=["single", "boards"], default="single")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--selenium-fallback", action="store_true")
    ap.add_argument("--driver", default=getattr(CONFIG, "chromedriver", "resource/chromedriver-mac-arm64/chromedriver"))
    ap.add_argument("--format", choices=["plain", "json"], default="plain")
    args = ap.parse_args()

    if args.mode == "boards":
        boards = fetch_boards(topn=args.topn, selenium_fallback=args.selenium_fallback, driver_path=args.driver)
        if args.format == "json":
            print(json.dumps(boards, ensure_ascii=False, indent=2))
        else:
            for k, v in boards.items():
                print(f"\n[{k}]")
                for i, it in enumerate(v, 1):
                    print(f"{i}. {it.get('title')} -> {it.get('url')}")
        return

    payload = fetch_hot_search(topn=args.topn, tech_only=False,
                               selenium_fallback=args.selenium_fallback, driver_path=args.driver)
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(_render_plain(payload))


if __name__ == "__main__":
    main()
