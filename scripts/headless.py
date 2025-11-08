# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午2:12
# @Author: 柯影数智
# @File: headless.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm

"""
scripts/headless.py — 微博热搜页面“无头”截图脚本（基于 weibo_hotsearch.screenshot）

功能
- 读取 .env（经 weibo_hotsearch.config.CONFIG）中的 CHROMEDRIVER / WEIBO_COOKIE / TZ
- 支持移动端/桌面端渲染；可关闭无头模式便于调试
- 自动等待热搜表格渲染，触发懒加载，进行全页截图（CDP），失败回退普通截图
- 可选版本自检：检测本机 Chrome 与 chromedriver 主版本是否一致并给出提示
- 截图完成后可选自动用系统默认查看器打开（macOS: open）

用法
-----
# 最常用：移动端风格 + 无头
python scripts/headless.py

# 桌面模式 + 调试（非无头）
python scripts/headless.py --desktop --no-headless

# 指定 chromedriver、输出目录、超时
python scripts/headless.py --driver resource/chromedriver-mac-arm64/chromedriver \
                           --outdir screenshot --timeout 20

# 仅做版本自检
python scripts/headless.py --check

注意
- 如遇到接口 403/需登录等问题，确保 .env 中配置了 WEIBO_COOKIE（浏览器复制整行）。
- 如遇 “This version of ChromeDriver only supports Chrome version …”：
  1) 将 chromedriver 更新到与 Chrome 主版本一致；
  2) 或安装匹配版本的 Chrome；再在 .env 中修正 CHROMEDRIVER 路径。
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime

from weibo_hotsearch.config import CONFIG, tzinfo, WEEKDAY_CN
from weibo_hotsearch.screenshot import capture_summary_screenshot, WEIBO_SUMMARY_URL


# =========================
#  版本检测（可选）
# =========================
def _parse_major(ver_str: str) -> int | None:
    """
    从版本字符串中解析主版本号：
    - "Google Chrome 142.0.7444.60"  -> 142
    - "ChromeDriver 142.0.0.0 (…)"   -> 142
    """
    if not ver_str:
        return None
    m = re.search(r"\b(\d+)\.\d+\.\d+\.\d+\b", ver_str)
    return int(m.group(1)) if m else None


def _chrome_version_str() -> str | None:
    """尝试获取本机 Chrome/Chromium 版本字符串。"""
    candidates = []
    if platform.system() == "Darwin":  # macOS
        candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        candidates.append("/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta")
    elif platform.system() == "Linux":
        candidates.extend(["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"])
    else:  # Windows 或其他平台
        # Windows 下可手动补充，也可依赖用户自检
        candidates.extend(["chrome", "chrome.exe", "chrome.bat"])

    for exe in candidates:
        try:
            out = subprocess.check_output([exe, "--version"], stderr=subprocess.STDOUT, text=True)
            if out:
                return out.strip()
        except Exception:
            continue
    return None


def _chromedriver_version_str(driver_path: str | Path) -> str | None:
    """获取 chromedriver 版本字符串。"""
    p = str(driver_path)
    try:
        out = subprocess.check_output([p, "--version"], stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def check_versions(driver_path: str | Path) -> tuple[int | None, int | None, list[str]]:
    """
    返回 (chrome_major, driver_major, warnings)
    """
    warns: list[str] = []
    ch_str = _chrome_version_str() or ""
    dr_str = _chromedriver_version_str(driver_path) or ""

    ch_major = _parse_major(ch_str)
    dr_major = _parse_major(dr_str)

    if not ch_str:
        warns.append("未能自动检测到本机 Chrome 版本（可忽略，如你确认版本已匹配）。")
    if not dr_str:
        warns.append(f"未能检测到 chromedriver 版本（路径：{driver_path}），请确认文件存在且可执行。")

    if ch_major and dr_major and ch_major != dr_major:
        warns.append(
            f"Chrome 与 chromedriver 主版本不一致：Chrome={ch_major} / chromedriver={dr_major}。\n"
            "建议安装匹配版本的 chromedriver，或更换 Chrome 到对应主版本。"
        )

    return ch_major, dr_major, warns


# =========================
#  CLI
# =========================
def main() -> int:
    ap = argparse.ArgumentParser(description="微博热搜页面无头截图（移动/桌面 + 全页截图）")
    ap.add_argument("--driver", default=str(CONFIG.chromedriver), help="chromedriver 路径（默认读取 .env）")
    ap.add_argument("--outdir", default=str(CONFIG.screenshot_dir), help="截图输出目录（默认读取 .env）")
    ap.add_argument("--url", default=WEIBO_SUMMARY_URL, help="目标 URL（默认微博热搜 summary）")
    ap.add_argument("--desktop", action="store_true", help="使用桌面模式（默认移动端 UA）")
    ap.add_argument("--no-headless", action="store_true", help="关闭无头模式（便于调试）")
    ap.add_argument("--timeout", type=int, default=15, help="等待热搜表格渲染的超时时间（秒）")
    ap.add_argument("--no-cookie", action="store_true", help="不注入 .env 的 WEIBO_COOKIE")
    ap.add_argument("--open", dest="auto_open", action="store_true", help="截图完成后用系统默认查看器打开")
    ap.add_argument("--check", action="store_true", help="仅进行 Chrome/chromedriver 版本自检并退出")
    args = ap.parse_args()

    driver_path = Path(args.driver)
    outdir = Path(args.outdir)

    # 版本自检
    ch_major, dr_major, warns = check_versions(driver_path)
    if args.check:
        print("版本自检：")
        print(f"  Chrome   : {ch_major if ch_major is not None else '未知'}")
        print(f"  Driver   : {dr_major if dr_major is not None else '未知'}")
        if warns:
            print("\n提示：")
            for w in warns:
                print(" - " + w)
        return 0

    if warns:
        for w in warns:
            sys.stderr.write("[WARN] " + w + "\n")

    # 执行截图
    try:
        path = capture_summary_screenshot(
            driver_path=str(driver_path),
            outdir=str(outdir),
            url=args.url,
            desktop=args.desktop,
            headless=not args.no_headless,
            timeout=args.timeout,
            inject_cookie=not args.no_cookie,
        )
    except Exception as e:
        sys.stderr.write(f"[ERROR] 截图失败：{e}\n")
        return 2

    # 输出结果与时间信息
    now = datetime.now(tzinfo())
    wd = WEEKDAY_CN[(now.isoweekday() - 1) % 7]
    print(f"截图完成：{path}")
    print(f"时间：{now.strftime('20%y年%m月%d日 %H:%M')} {wd}")

    # 可选自动打开
    if args.auto_open:
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", path], check=False)
            elif platform.system() == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            else:  # Linux
                subprocess.run(["xdg-open", path], check=False)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
