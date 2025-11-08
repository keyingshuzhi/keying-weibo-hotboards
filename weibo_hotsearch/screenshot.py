# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午2:11
# @Author: 柯影数智
# @File: screenshot.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm


"""
weibo_hotsearch.screenshot — 微博热搜页面截图（支持移动/桌面 + 全页截图）

特性
- 显式使用 chromedriver（默认读取 .env -> CONFIG.chromedriver）
- 移动端/桌面端两种渲染模式，可关闭无头便于调试
- 等待热搜表格渲染完成，再做懒加载滚动
- 通过 CDP 获取内容尺寸并进行「全页截图」，失败时回退普通截图
- 若 .env 配置了 WEIBO_COOKIE，自动注入到浏览器（weibo.com 与 s.weibo.com）

用法（命令行）
--------------
python -m weibo_hotsearch.screenshot \
  --desktop                 # 使用桌面模式（默认移动端 UA）
  --no-headless             # 关闭无头便于调试
  --timeout 20              # 等待热搜表格的超时时间（秒）
  --driver resource/chromedriver-mac-arm64/chromedriver
  --url "https://s.weibo.com/top/summary?cate=realtimehot"
  --outdir screenshot

也可在代码中调用 capture_summary_screenshot(...)
"""

from __future__ import annotations

import base64
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .config import CONFIG, tzinfo, WEEKDAY_CN

WEIBO_SUMMARY_URL = "https://s.weibo.com/top/summary?cate=realtimehot"


# =========================
#  内部：Selenium 构造与工具
# =========================
def _build_options(*, desktop: bool, headless: bool, user_agent: Optional[str] = None) -> Options:
    opts = Options()
    if headless:
        # 新 headless 更稳定
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--lang=zh-CN,zh;q=0.9,en;q=0.8")

    if desktop:
        # 桌面模式
        opts.add_argument("--window-size=1366,2200")
        ua = user_agent or CONFIG.head_ua
        opts.add_argument(f"--user-agent={ua}")
    else:
        # 移动端模拟
        mobile_emulation = {
            "deviceMetrics": {"width": 375, "height": 750, "pixelRatio": 2.0},
            "userAgent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 "
                "Mobile/15E148 Safari/604.1 Edg/116.0.0.0"
            ),
        }
        opts.add_experimental_option("mobileEmulation", mobile_emulation)
    return opts


def _apply_stealth(driver: webdriver.Chrome) -> None:
    """尽量减少 webdriver 特征。"""
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN','zh','en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            """
        },
    )


def _inject_cookie_if_any(driver: webdriver.Chrome) -> None:
    """
    如果 .env 配置了 WEIBO_COOKIE，则把 Cookie 注入到 weibo.com 与 s.weibo.com。
    注意：Selenium 需要先打开同域页面才能 set_cookie。
    """
    ck = (CONFIG.weibo_cookie or "").strip()
    if not ck:
        return

    def _parse(cookie_str: str) -> dict:
        # 粗解析；键值内不应包含分号
        out = {}
        parts = [p for p in cookie_str.split(";") if p.strip()]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    kv = _parse(ck)
    if not kv:
        return

    # 先到 weibo.com 再 set_cookie
    for domain in ["https://weibo.com", "https://s.weibo.com"]:
        try:
            driver.get(domain)
            # 注入关键 Cookie
            for k, v in kv.items():
                try:
                    driver.add_cookie({"name": k, "value": v, "path": "/", "domain": ".weibo.com"})
                    driver.add_cookie({"name": k, "value": v, "path": "/", "domain": ".s.weibo.com"})
                except Exception:
                    pass
        except Exception:
            pass


def _wait_hot_table(driver: webdriver.Chrome, timeout: int) -> None:
    """
    等待热搜表格/列表渲染完成。
    常见容器：#pl_top_realtimehot 或包含热搜行的 table/tbody。
    """
    wait = WebDriverWait(driver, timeout)
    wait.until(
        EC.any_of(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#pl_top_realtimehot")),
            EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")),
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.data tbody tr")),
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='weibo']")),
        )
    )


def _fullpage_screenshot(driver: webdriver.Chrome, out_path: Path) -> bool:
    """
    通过 CDP 获取布局尺寸并全页截图；失败时回退为普通截图。
    """
    try:
        metrics = driver.execute_cdp_cmd("Page.getLayoutMetrics", {})
        width = int(metrics["contentSize"]["width"])
        height = int(metrics["contentSize"]["height"])

        # 控制最大尺寸，避免超限
        width = max(375, min(width, 4000))
        height = max(750, min(height, 20000))
        driver.set_window_size(width, height)

        result = driver.execute_cdp_cmd(
            "Page.captureScreenshot",
            {"format": "png", "fromSurface": True, "captureBeyondViewport": True},
        )
        data = base64.b64decode(result["data"])
        out_path.write_bytes(data)
        return True
    except Exception:
        # 回退普通截图
        return driver.get_screenshot_as_file(str(out_path))


# =========================
#  对外：核心截图函数
# =========================
def capture_summary_screenshot(
        *,
        driver_path: Optional[str] = None,
        outdir: Optional[str | Path] = None,
        url: Optional[str] = None,
        desktop: bool = False,
        headless: bool = True,
        timeout: int = 15,
        inject_cookie: bool = True,
) -> str:
    """
    打开微博热搜页并全页截图，返回保存的文件路径（str）。

    参数
    ----
    driver_path : chromedriver 路径（默认取 CONFIG.chromedriver）
    outdir      : 输出目录（默认取 CONFIG.screenshot_dir）
    url         : 目标 URL（默认热搜 summary）
    desktop     : True=桌面模式；False=移动端模式（默认）
    headless    : 是否无头（默认 True）
    timeout     : 等待热搜表格的超时时间（秒）
    inject_cookie: 若 .env 有 Cookie 是否注入（默认 True）
    """
    driver_path = driver_path or CONFIG.chromedriver
    out_dir = Path(outdir or CONFIG.screenshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tzinfo())
    filename = now.strftime("20%y年%m月%d日%H:%M.png")
    out_path = out_dir / filename

    opts = _build_options(desktop=desktop, headless=headless, user_agent=CONFIG.head_ua)
    service = Service(executable_path=str(driver_path))

    try:
        driver = webdriver.Chrome(service=service, options=opts)
    except SessionNotCreatedException as e:
        # 常见：Chrome 与 chromedriver 版本不匹配
        msg = (
            f"[ERROR] 无法创建浏览器会话：{e}\n"
            "原因可能是 Chrome 与 chromedriver 主版本不一致。\n"
            "请确保：\n"
            "  1) 本机 Chrome 版本与 chromedriver 主版本一致（chrome://version / chromedriver --version）\n"
            "  2) 或者更新 .env 中 CHROMEDRIVER 指向匹配版本的驱动文件。"
        )
        raise RuntimeError(msg) from e
    except WebDriverException as e:
        raise RuntimeError(f"[ERROR] 启动 Chrome 失败：{e}") from e

    try:
        _apply_stealth(driver)

        if inject_cookie and (CONFIG.weibo_cookie or "").strip():
            _inject_cookie_if_any(driver)

        target = url or WEIBO_SUMMARY_URL
        driver.get(target)

        # 等待关键元素加载
        _wait_hot_table(driver, timeout=timeout)

        # 触发懒加载
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.4);")
        driver.execute_script("window.scrollTo(0, 0);")

        # 全页截图
        ok = _fullpage_screenshot(driver, out_path)
        if not ok:
            raise RuntimeError("截图写盘失败")

        return str(out_path.resolve())

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# =========================
#  CLI
# =========================
def _cli() -> None:
    ap = argparse.ArgumentParser(description="微博热搜页面截图（移动/桌面 + 全页截图）")
    ap.add_argument("--driver", default=CONFIG.chromedriver, help="chromedriver 路径")
    ap.add_argument("--outdir", default=CONFIG.screenshot_dir, help="截图输出目录")
    ap.add_argument("--url", default=WEIBO_SUMMARY_URL, help="目标 URL（默认热搜 summary）")
    ap.add_argument("--desktop", action="store_true", help="使用桌面模式（默认移动端）")
    ap.add_argument("--no-headless", action="store_true", help="关闭无头（调试时使用）")
    ap.add_argument("--timeout", type=int, default=15, help="等待热搜表格超时时间（秒）")
    ap.add_argument("--no-cookie", action="store_true", help="不注入 .env 的 WEIBO_COOKIE")
    args = ap.parse_args()

    path = capture_summary_screenshot(
        driver_path=args.driver,
        outdir=args.outdir,
        url=args.url,
        desktop=args.desktop,
        headless=not args.no_headless,
        timeout=args.timeout,
        inject_cookie=not args.no_cookie,
    )
    now = datetime.now(tzinfo())
    wd = WEEKDAY_CN[(now.isoweekday() - 1) % 7]
    print(f"截图完成：{path} ｜ 时间：{now.strftime('20%y年%m月%d日 %H:%M')} {wd}")


if __name__ == "__main__":
    _cli()
