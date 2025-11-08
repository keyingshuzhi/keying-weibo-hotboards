# -*- coding = utf-8 -*-
# @Time: 2025/11/7 下午3:30
# @Author: 柯影数智
# @File: render_image.py
# @Email: 1090461393@qq.com
# @SoftWare: PyCharm


from __future__ import annotations
from typing import List, Dict, Tuple, OrderedDict as OD, Optional
from pathlib import Path
import datetime as _dt

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter  # type: ignore
except Exception as e:  # 显式抛错，提示安装 pillow
    raise RuntimeError("Pillow is required for rendering. Please `pip install pillow`.") from e

# ========== 兼容导入 CONFIG / tz / WEEKDAY ==========
try:
    from .config import CONFIG, tzinfo, WEEKDAY_CN  # type: ignore
except Exception:
    def tzinfo():
        return _dt.timezone(_dt.timedelta(hours=8))


    WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


    class _C:
        bg = ""
        font = ""
        numfont = ""


    CONFIG = _C()  # type: ignore

# ------------------ 设计令牌 ------------------
PAL = {
    "bg": (247, 249, 252),  # 画布底
    "brand_grad": [(255, 121, 78), (255, 83, 57)],
    "card": (255, 255, 255),  # 卡片底
    "line": (237, 240, 243),
    "title": (30, 33, 38),
    "muted": (120, 130, 140),
    "rank_1": (229, 57, 53),
    "rank_2": (255, 143, 0),
    "rank_3": (251, 192, 45),
    "rank_other": (98, 106, 115),
}

TOK = {
    "W": 1080, "H_MIN": 1280,  # 最小高度；实际会根据内容自适应
    "SAFE": 48,
    "RADIUS": 24,
    "GAP": 24,
    "HEADER_H": 168,
    "CARD_HEAD_H": 72,
    "ROW_H": 72,
    "MAX_ROWS": 10,
}


# ------------------ 字体加载 ------------------
def _load_font(path: str, size: int, fallback: str = "Arial.ttf") -> ImageFont.FreeTypeFont:
    p = (path or "").strip()
    if p and Path(p).exists():
        return ImageFont.truetype(p, size)
    try:
        return ImageFont.truetype(fallback, size)
    except Exception:
        return ImageFont.load_default()


def _fonts():
    return {
        "h1": _load_font(getattr(CONFIG, "font", ""), 44),
        "h2": _load_font(getattr(CONFIG, "font", ""), 24),
        "board_head": _load_font(getattr(CONFIG, "font", ""), 26),
        "t_top": _load_font(getattr(CONFIG, "font", ""), 30),
        "t_row": _load_font(getattr(CONFIG, "font", ""), 28),
        "num": _load_font(getattr(CONFIG, "numfont", ""), 26),
        "badge": _load_font(getattr(CONFIG, "numfont", ""), 24),
    }


# ------------------ 绘图小工具 ------------------
def _round_rect(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], r: int, fill, outline=None, width=1):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill, outline=outline, width=width)


def _text(draw: ImageDraw.ImageDraw, xy, text, font, fill, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _shadow_card(base: Image.Image, rect: Tuple[int, int, int, int], radius=24, spread=6):
    x1, y1, x2, y2 = rect
    w = x2 - x1;
    h = y2 - y1
    shadow = Image.new("RGBA", (w + spread * 2, h + spread * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    _round_rect(d, (spread, spread, spread + w, spread + h), radius, (0, 0, 0, 64))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    base.alpha_composite(shadow, (x1 - spread, y1 - spread))


def _header_gradient(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), PAL["brand_grad"][0])
    p0, p1 = PAL["brand_grad"]
    d = ImageDraw.Draw(img)
    for x in range(w):
        t = x / max(1, w - 1)
        r = int(p0[0] + (p1[0] - p0[0]) * t)
        g = int(p0[1] + (p1[1] - p0[1]) * t)
        b = int(p0[2] + (p1[2] - p0[2]) * t)
        d.line([(x, 0), (x, h)], fill=(r, g, b))
    return img


def _text_ellipsis(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 2) -> str:
    if not text:
        return ""
    if max_lines <= 1:
        s = text
        while s and draw.textlength(s, font=font) > max_w:
            s = s[:-2] + "…"
        return s
    # 多行逐字符
    words = list(text)
    lines, cur = [], ""
    for ch in words:
        if draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur);
            cur = ch
            if len(lines) == max_lines - 1:
                break
    if cur:
        lines.append(cur)
    if draw.textlength(lines[-1], font=font) > max_w:
        s = lines[-1]
        while s and draw.textlength(s, font=font) > max_w:
            s = s[:-2] + "…"
        lines[-1] = s
    return "\n".join(lines[:max_lines])


# ------------------ 行渲染 ------------------
def _rank_color(rank: int):
    if rank == 1: return PAL["rank_1"]
    if rank == 2: return PAL["rank_2"]
    if rank == 3: return PAL["rank_3"]
    return PAL["rank_other"]


def _draw_row(draw, x, y, w, text, rank, f_top, f_row, f_num, max_lines=2):
    # 徽章
    badge_r = 24
    bx = x;
    by = y + TOK["ROW_H"] // 2
    draw.ellipse([bx, by - badge_r, bx + badge_r * 2, by + badge_r * 2], fill=_rank_color(rank))
    _text(draw, (bx + badge_r, by), str(rank), f_num, (255, 255, 255), anchor="mm")

    # 文本
    tx = bx + badge_r * 2 + 16
    maxw = max(40, w - (tx - x) - 8)
    font = f_top if rank <= 3 else f_row
    s = _text_ellipsis(draw, text, font, maxw, max_lines=max_lines)
    _text(draw, (tx, y + TOK["ROW_H"] // 2), s, font, PAL["title"], anchor="lm")


# ------------------ 卡片 ------------------
def _card_height(n_items: int) -> int:
    return TOK["CARD_HEAD_H"] + TOK["ROW_H"] * min(TOK["MAX_ROWS"], n_items) + 16


def _draw_board_card(canvas: Image.Image, xy: Tuple[int, int], card_w: int, title: str, items: List[Dict]):
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    h = _card_height(len(items))
    rect = (x, y, x + card_w, y + h)

    # 阴影 + 卡体
    _shadow_card(canvas, rect, TOK["RADIUS"])
    _round_rect(draw, rect, TOK["RADIUS"], PAL["card"])

    f = _fonts()
    # 卡头
    head_rect = (x, y, x + card_w, y + TOK["CARD_HEAD_H"])
    _round_rect(draw, head_rect, TOK["RADIUS"], PAL["card"], outline=PAL["line"], width=1)
    _text(draw, (x + 20, y + TOK["CARD_HEAD_H"] // 2), title, f["board_head"], PAL["muted"], anchor="lm")

    # 分割线
    draw.line([(x, y + TOK["CARD_HEAD_H"]), (x + card_w, y + TOK["CARD_HEAD_H"])], fill=PAL["line"], width=1)

    # 行
    cy = y + TOK["CARD_HEAD_H"]
    for i, it in enumerate(items[:TOK["MAX_ROWS"]], 1):
        _draw_row(draw, x + 20, cy + (i - 1) * TOK["ROW_H"], card_w - 40,
                  it.get("title", ""), i, f["t_top"], f["t_row"], f["badge"])
        if i < min(TOK["MAX_ROWS"], len(items)):
            yline = cy + i * TOK["ROW_H"]
            draw.line([(x + 20, yline), (x + card_w - 20, yline)], fill=PAL["line"], width=1)
    return rect[3]  # 返回卡片底部 y


# ------------------ 单榜渲染（供 main 的 _render_image_compat 使用） ------------------
def render_image(items: List[Dict], out_path: Path | str, board_title: Optional[str] = None) -> str:
    W = TOK["W"]
    card_h = _card_height(len(items))
    final_h = max(TOK["H_MIN"], TOK["HEADER_H"] + TOK["GAP"] + card_h + TOK["SAFE"])

    img = Image.new("RGBA", (W, final_h), PAL["bg"] + (255,))
    draw = ImageDraw.Draw(img)

    # 顶部品牌条
    header = _header_gradient(W, TOK["HEADER_H"])
    img.paste(header, (0, 0))

    f = _fonts()
    now = _dt.datetime.now(tzinfo())
    title = "微博热搜"
    sub = now.strftime("20%y年%m月%d日 %H:%M") + f"  {WEEKDAY_CN[(now.isoweekday() - 1) % 7]}｜{board_title or '榜单'}"
    _text(draw, (TOK["SAFE"], 48), title, f["h1"], (255, 255, 255), anchor="la")
    _text(draw, (TOK["SAFE"], 48 + 56), sub, f["h2"], (255, 255, 255), anchor="la")

    # 卡片
    x0 = TOK["SAFE"];
    y0 = TOK["HEADER_H"] + TOK["GAP"]
    card_w = W - TOK["SAFE"] * 2
    _draw_board_card(img, (x0, y0), card_w, f"微博 · {board_title or '榜单'}", items)

    # 保存
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(outp, quality=92, subsampling=2)
    return str(outp)


# ------------------ 两榜合并渲染（main 优先调用） ------------------
def render_two_boards_image(
        boards: "OD[str, List[Dict]]",
        out_path: Path | str,
        *,
        size: Tuple[int, int] | None = None,
) -> str:
    W = (size[0] if size else TOK["W"])
    # 预估画布高度
    names = list(boards.keys())[:2]  # 仅热搜/社会
    h1 = _card_height(len(boards.get(names[0], []))) if len(names) >= 1 else 0
    h2 = _card_height(len(boards.get(names[1], []))) if len(names) >= 2 else 0

    content_w = W - TOK["SAFE"] * 2
    gap = TOK["GAP"]
    two_cols = content_w >= (440 * 2 + gap)
    body_h = max(h1, h2) if two_cols else (h1 + gap + h2)
    final_h = max(TOK["H_MIN"], TOK["HEADER_H"] + TOK["GAP"] + body_h + TOK["SAFE"])

    img = Image.new("RGBA", (W, final_h), PAL["bg"] + (255,))
    draw = ImageDraw.Draw(img)

    # 顶部品牌条
    header = _header_gradient(W, TOK["HEADER_H"])
    img.paste(header, (0, 0))

    f = _fonts()
    now = _dt.datetime.now(tzinfo())
    _text(draw, (TOK["SAFE"], 48), "微博热搜", f["h1"], (255, 255, 255), anchor="la")
    sub = now.strftime("20%y年%m月%d日 %H:%M") + f"  {WEEKDAY_CN[(now.isoweekday() - 1) % 7]}｜热搜榜 · 社会榜"
    _text(draw, (TOK["SAFE"], 48 + 56), sub, f["h2"], (255, 255, 255), anchor="la")

    # 内容区
    x0 = TOK["SAFE"];
    y0 = TOK["HEADER_H"] + TOK["GAP"]
    col_w = (content_w - (gap if two_cols else 0)) // (2 if two_cols else 1)

    cur_y = y0
    if two_cols:
        # 左右双栏
        for col, name in enumerate(names):
            items = boards.get(name) or []
            _draw_board_card(img, (x0 + col * (col_w + gap), y0), col_w, f"微博 · {name}", items)
    else:
        # 上下排布
        for i, name in enumerate(names):
            items = boards.get(name) or []
            bottom = _draw_board_card(img, (x0, cur_y), col_w, f"微博 · {name}", items)
            cur_y = bottom + gap

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(outp, quality=92, subsampling=2)
    return str(outp)


# ------------------ 三榜占位（保持兼容，当前未使用） ------------------
def render_three_boards_image(*args, **kwargs) -> str:
    raise NotImplementedError("render_three_boards_image is not implemented in this renderer.")
