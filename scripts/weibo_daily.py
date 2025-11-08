# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午2:12
# @Author: 柯影数智
# @File: weibo_daily.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

"""
scripts/weibo_daily.py — 微博热搜 · 抓取→摘要(DeepSeek)→图片卡片→发企业微信

整合流程
- 抓取：优先 JSON，失败回退 HTML；可选 Selenium 兜底（需 chromedriver）
- 筛选：支持仅“科技类”与 TopN
- 摘要：可选 DeepSeek（读取 .env 的 DEEPSEEK_API_KEY）
- 图片：渲染榜单图片卡片（读取 .env 的 BG/FONT/NUMFONT/OUTDIR）
- 推送：企业微信机器人（读取 .env 的 WECOM_WEBHOOK_KEY）
- 附带：可同时发送截图版（基于 Selenium 的全页截图）

示例
-----
# 科技类 Top10，生成摘要与榜单图片，并推送到企业微信群
python scripts/weibo_daily.py --tech-only --topn 10 \
  --summary --with-image --wecom \
  --selenium-fallback --driver resource/chromedriver-mac-arm64/chromedriver

# 仅在控制台输出 Markdown（不推送）
python scripts/weibo_daily.py --tech-only --topn 10 --summary

# 同时附加整页截图（在发完 Markdown 后再发一条图片消息）
python scripts/weibo_daily.py --with-screenshot --selenium-fallback --wecom
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from weibo_hotsearch.config import CONFIG, tzinfo, WEEKDAY_CN
from weibo_hotsearch.fetch import fetch_hot_search
from weibo_hotsearch.utils import build_markdown
from weibo_hotsearch.render_image import render_image
from weibo_hotsearch.summary_deepseek import summarize_items
from weibo_hotsearch.screenshot import capture_summary_screenshot


def _timestamp_filename(ext: str = "png") -> str:
    now = datetime.now(tzinfo())
    return now.strftime(f"20%y年%m月%d日%H:%M.{ext}")


def _save_payload_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="微博热搜：抓取→摘要→图片→企微 推送脚本")
    ap.add_argument("--topn", type=int, default=10, help="前 N 条（默认 10）")
    ap.add_argument("--tech-only", action="store_true", help="仅科技类")
    ap.add_argument("--selenium-fallback", action="store_true", help="失败时启用 Selenium 兜底")
    ap.add_argument("--driver", default=str(CONFIG.chromedriver), help="chromedriver 路径（配合 --selenium-fallback）")

    ap.add_argument("--summary", action="store_true", help="启用 DeepSeek 摘要（需 DEEPSEEK_API_KEY）")
    ap.add_argument("--summary-max", type=int, default=400, help="摘要最大字数（默认 400）")

    ap.add_argument("--with-image", action="store_true", help="生成并附带榜单图片（image 消息）")
    ap.add_argument("--with-screenshot", action="store_true", help="附带整页截图（Selenium，全页截图）")

    ap.add_argument("--wecom", action="store_true", help="发送到企业微信群（读取 .env 的 WECOM_WEBHOOK_KEY）")
    ap.add_argument("--wecom-key", default="", help="覆盖默认 Webhook（可填 key 或完整 URL）")

    ap.add_argument("--save-json", default="", help="把抓取结果另存为 JSON（路径）")
    ap.add_argument("--outdir", default=str(CONFIG.outdir), help="图片输出目录（默认读取 .env）")
    args = ap.parse_args()

    # 1) 抓取
    payload = fetch_hot_search(
        topn=args.topn,
        tech_only=args.tech_only,
        selenium_fallback=args.selenium_fallback,
        driver_path=args.driver,
    )

    items = payload.get("items", []) or []
    if args.save_json:
        _save_payload_json(payload, Path(args.save_json))

    if not items:
        sys.stderr.write("[ERROR] 未获取到热搜条目；")
        if payload.get("hint"):
            sys.stderr.write(payload["hint"] + "\n")
        else:
            sys.stderr.write("可尝试设置 WEIBO_COOKIE 或启用 --selenium-fallback。\n")
        return 2

    # 2) 摘要（可选）
    summary_text = None
    if args.summary:
        summary_text = summarize_items(items, max_words=args.summary_max)
        if not summary_text:
            sys.stderr.write("[WARN] DeepSeek 摘要失败或未配置 DEEPSEEK_API_KEY，已跳过摘要。\n")

    # 3) 生成 Markdown
    md = build_markdown(
        items,
        title_prefix="微博热搜",
        summary_text=summary_text,
        tech_only=args.tech_only,
    )

    # 4) 生成榜单图片（可选）
    image_path = None
    if args.with_image:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = _timestamp_filename("png")
        out_path = outdir / fname
        image_path = render_image(
            items,
            bg_path=str(CONFIG.bg) if CONFIG.bg and CONFIG.bg.exists() else None,
            out_path=out_path,
            font_path=str(CONFIG.font) if CONFIG.font and CONFIG.font.exists() else None,
            num_font_path=str(CONFIG.numfont) if CONFIG.numfont and CONFIG.numfont.exists() else None,
        )
        print(f"[OK] 榜单图片已保存：{image_path}")

    # 5) 可选整页截图（Selenium）
    screenshot_path = None
    if args.with_screenshot:
        try:
            screenshot_path = capture_summary_screenshot(
                driver_path=args.driver,
                outdir=str(CONFIG.screenshot_dir),
                desktop=False,
                headless=True,
                timeout=15,
                inject_cookie=True,
            )
            print(f"[OK] 整页截图已保存：{screenshot_path}")
        except Exception as e:
            sys.stderr.write(f"[WARN] 截图失败：{e}\n")

    # 6) 推送或打印
    if args.wecom:
        from weibo_hotsearch.wecom import send_markdown, send_image_from_path

        key = args.wecom_key or None
        ok_md, md_resps = send_markdown(md, webhook=key)
        print(f"[WeCom/markdown] sent={ok_md} resp={md_resps[-1] if md_resps else {} }")
        all_ok = ok_md

        if image_path:
            ok_img, resp_img = send_image_from_path(image_path, webhook=key)
            print(f"[WeCom/image] sent={ok_img} resp={resp_img}")
            all_ok = all_ok and ok_img

        if screenshot_path:
            ok_shot, resp_shot = send_image_from_path(screenshot_path, webhook=key)
            print(f"[WeCom/screenshot] sent={ok_shot} resp={resp_shot}")
            all_ok = all_ok and ok_shot

        return 0 if all_ok else 4

    # 默认打印 Markdown 到 stdout（便于 crontab 日志查看）
    print(md)
    if image_path:
        print(f"\n[提示] 榜单图片：{image_path}")
    if screenshot_path:
        print(f"[提示] 整页截图：{screenshot_path}")
    if payload.get("hint"):
        print(f"\n[抓取提示] {payload['hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
