"""Generate the QFactor-Penny README demo loop GIF.

The animation is intentionally small and deterministic: it is a visual abstract
of the benchmark's failure mode, not a rendered copy of the full explainer video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "qfactor-penny-demo-loop.gif"
LOGO = ROOT / "assets" / "qfactor-penny-logo.png"

W, H = 1280, 720
FPS = 10
DURATION_SECONDS = 7.2
FRAMES = int(FPS * DURATION_SECONDS)

BG = (8, 18, 16)
PANEL = (13, 28, 25)
PANEL_2 = (18, 38, 35)
GRID = (35, 67, 59)
TEXT = (235, 240, 230)
MUTED = (137, 158, 148)
MINT = (126, 199, 166)
AMBER = (224, 168, 86)
RUST = (215, 119, 104)
BLUE_GREY = (109, 139, 151)

TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
INITIAL_SCORES = [0.45, 0.68, 0.33, 0.57, 0.77, 0.39, 0.88, 0.51, 0.61, 0.84, 0.47]
COLLAPSED = 0.52
FEATURES = ["month_cos", "weekday_cos", "month_sin"]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logo = _load_logo()
    frames = [_frame(i, logo) for i in range(FRAMES)]
    quantized = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
    quantized[0].save(
        OUTPUT,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


def _frame(index: int, logo: Image.Image) -> Image.Image:
    t = index / max(FRAMES - 1, 1) * DURATION_SECONDS
    image = Image.new("RGBA", (W, H), (*BG, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    fonts = _fonts()
    _draw_background(draw)
    _draw_header(image, draw, logo, fonts, t)
    _draw_ticker_line(draw, fonts, t)
    _draw_qnn_machine(draw, fonts, t)
    _draw_failure_matrix(draw, fonts, t)
    _draw_score_bars(draw, fonts, t)
    _draw_lesson(draw, fonts, t)
    _draw_fade(image, t)
    return image


def _draw_background(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, W, H), fill=BG)
    for x in range(0, W, 80):
        draw.line((x, 0, x, H), fill=(*GRID, 38), width=1)
    for y in range(0, H, 80):
        draw.line((0, y, W, y), fill=(*GRID, 32), width=1)
    draw.line((78, 92, W - 78, 92), fill=(*GRID, 90), width=1)
    draw.line((78, H - 82, W - 78, H - 82), fill=(*GRID, 60), width=1)
    draw.rectangle((0, 0, W, H), outline=(*GRID, 80), width=2)


def _draw_header(image: Image.Image, draw: ImageDraw.ImageDraw, logo: Image.Image, fonts: dict[str, ImageFont.ImageFont], t: float) -> None:
    intro = max(0.65, _appear(t, 0.0, 0.9)) * _disappear(t, 6.6, 0.45)
    image.alpha_composite(_opacity(logo, intro), (82, 39))
    _text(draw, (180, 46), "QFactor-Penny", fonts["title"], TEXT, intro)
    _text(draw, (182, 91), "leakage-aware QML benchmark", fonts["small"], MUTED, intro)
    _text(draw, (1055, 48), "SPY reference only", fonts["mono_small"], BLUE_GREY, _appear(t, 1.15, 0.45) * _disappear(t, 5.4, 0.4))


def _draw_ticker_line(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], t: float) -> None:
    alpha = _appear(t, 0.85, 0.65) * _disappear(t, 5.85, 0.55)
    _text(draw, (88, 150), "11 sector ETFs line up for a five-day ranking task", fonts["label"], TEXT, alpha)
    x0, y = 90, 204
    gap = 99
    start_y = 84
    for i, ticker in enumerate(TICKERS):
        local = _appear(t, 0.95 + i * 0.045, 0.45)
        yy = _lerp(start_y, y, _ease(local))
        _chip(draw, (x0 + i * gap, yy), 74, 36, ticker, fonts["mono"], MINT, alpha * local)
    _text(draw, (90, 269), "top 3 = winners        middle 5 return at inference        bottom 3 = losers", fonts["mono_small"], MUTED, alpha)
    _rounded(draw, (1006, 132, 1195, 179), 12, PANEL_2, BLUE_GREY, alpha)
    _text(draw, (1032, 146), "SPY", fonts["mono"], TEXT, alpha)
    _text(draw, (1085, 150), "benchmark", fonts["mono_small"], BLUE_GREY, alpha)


def _draw_qnn_machine(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], t: float) -> None:
    alpha = _appear(t, 1.8, 0.65) * _disappear(t, 5.95, 0.45)
    x1, y1, x2, y2 = 106, 335, 520, 520
    _rounded(draw, (x1, y1, x2, y2), 18, PANEL, MINT, alpha * 0.9)
    _text(draw, (132, 360), "small QNN scorer", fonts["label"], TEXT, alpha)
    _text(draw, (132, 407), "4 features", fonts["mono_small"], MUTED, alpha)
    _text(draw, (276, 407), "4 qubits", fonts["mono_small"], MUTED, alpha)
    _text(draw, (404, 407), "score", fonts["mono_small"], MUTED, alpha)
    for x in [238, 366]:
        draw.line((x, 416, x + 68, 416), fill=(*MINT, int(185 * alpha)), width=2)
        draw.polygon([(x + 68, 416), (x + 58, 410), (x + 58, 422)], fill=(*MINT, int(185 * alpha)))
    for i in range(4):
        cx, cy = 296 + i * 22, 465 + (i % 2) * 13
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=(*MINT, int(210 * alpha)), width=2)
    draw.arc((286, 440, 374, 501), 18, 342, fill=(*AMBER, int(180 * alpha)), width=2)


def _draw_failure_matrix(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], t: float) -> None:
    alpha = _appear(t, 3.15, 0.6) * _disappear(t, 6.45, 0.45)
    if alpha <= 0:
        return
    _text(draw, (578, 145), "calendar-heavy features slide in", fonts["label"], TEXT, alpha)
    for i, feature in enumerate(FEATURES):
        x = 585 + i * 158
        yy = _lerp(104, 188, _ease(_appear(t, 3.15 + i * 0.12, 0.48)))
        _chip(draw, (x, yy), 136, 34, feature, fonts["mono_small"], AMBER, alpha)
    left, top = 586, 252
    cell_w, cell_h = 103, 24
    _rounded(draw, (left - 16, top - 37, left + cell_w * 3 + 18, top + cell_h * 11 + 16), 14, PANEL, AMBER, alpha * 0.8)
    _text(draw, (left, top - 27), "same date, same value", fonts["mono_small"], MUTED, alpha)
    for col in range(3):
        x = left + col * cell_w
        draw.rectangle((x, top, x + cell_w - 8, top + cell_h * 11), fill=(*AMBER, int(35 * alpha)))
        for row in range(11):
            y = top + row * cell_h
            shade = 48 + col * 20
            draw.rectangle((x, y, x + cell_w - 8, y + cell_h - 4), fill=(shade + 70, shade + 64, 42, int(115 * alpha)))
    for row, ticker in enumerate(TICKERS):
        _text(draw, (left - 64, top + row * cell_h + 2), ticker, fonts["mono_tiny"], MUTED, alpha)
    _text(draw, (585, 556), "Calendar signal ≠ cross-sectional signal", fonts["label"], RUST, alpha)


def _draw_score_bars(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], t: float) -> None:
    alpha = _appear(t, 2.15, 0.6)
    if alpha <= 0:
        return
    collapse = _ease(_appear(t, 4.0, 1.0))
    x0, y0 = 842, 486
    bar_w, gap = 22, 14
    _text(draw, (842, 337), "QNN sector scores", fonts["label"], TEXT, alpha)
    draw.line((x0 - 10, y0 + 5, x0 + 11 * (bar_w + gap), y0 + 5), fill=(*GRID, int(190 * alpha)), width=1)
    for i, ticker in enumerate(TICKERS):
        score = _lerp(INITIAL_SCORES[i], COLLAPSED + (0.012 if i % 4 == 0 else 0), collapse)
        h = int(150 * score)
        x = x0 + i * (bar_w + gap)
        color = RUST if collapse > 0.72 else MINT
        draw.rounded_rectangle((x, y0 - h, x + bar_w, y0), radius=5, fill=(*color, int(215 * alpha)))
        _text(draw, (x - 3, y0 + 16), ticker, fonts["mono_tiny"], MUTED, alpha)
    if collapse > 0.3:
        label_alpha = alpha * min(1.0, (collapse - 0.3) / 0.5)
        _rounded(draw, (865, 538, 1178, 591), 14, (47, 24, 22), RUST, label_alpha)
        _text(draw, (892, 556), "constant-score collapse", fonts["mono"], RUST, label_alpha)


def _draw_lesson(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.ImageFont], t: float) -> None:
    alpha = _appear(t, 5.35, 0.7) * _disappear(t, 6.85, 0.35)
    _rounded(draw, (300, 598, 980, 662), 18, (18, 24, 23), MINT, alpha * 0.6)
    _center_text(draw, (W // 2, 613), "Removing collapse ≠ finding signal", fonts["lesson"], TEXT, alpha)
    _center_text(draw, (W // 2, 660), "benchmark reveals fragility before noise is mistaken for advantage", fonts["small"], MUTED, alpha)


def _draw_fade(image: Image.Image, t: float) -> None:
    fade_out = 1.0 - _appear(t, 6.92, 0.28)
    opacity = fade_out
    if opacity >= 0.995:
        return
    overlay = Image.new("RGBA", (W, H), (*BG, 255))
    faded = Image.blend(image, overlay, 1 - opacity)
    image.paste(faded)


def _chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    width: int,
    height: int,
    label: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if alpha <= 0:
        return
    x, y = xy
    fill = (15, 30, 27)
    _rounded(draw, (x, y, x + width, y + height), 9, fill, color, alpha)
    bbox = draw.textbbox((0, 0), label, font=font)
    tx = x + (width - (bbox[2] - bbox[0])) / 2
    ty = y + (height - (bbox[3] - bbox[1])) / 2 - 2
    _text(draw, (tx, ty), label, font, TEXT, alpha)


def _rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    alpha: float,
) -> None:
    if alpha <= 0:
        return
    draw.rounded_rectangle(box, radius=radius, fill=(*fill, int(210 * alpha)), outline=(*outline, int(150 * alpha)), width=1)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if alpha <= 0:
        return
    draw.text(xy, text, font=font, fill=(*color, int(255 * max(0, min(1, alpha)))))


def _center_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = center[0] - (bbox[2] - bbox[0]) / 2
    y = center[1] - (bbox[3] - bbox[1]) / 2
    _text(draw, (x, y), text, font, color, alpha)


def _fonts() -> dict[str, ImageFont.ImageFont]:
    return {
        "title": _font(40, bold=True),
        "lesson": _font(35, bold=True),
        "label": _font(25, bold=True),
        "small": _font(20),
        "mono": _mono(22),
        "mono_small": _mono(18),
        "mono_tiny": _mono(13),
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/Library/Fonts/Inter_FXH-Bold.ttf" if bold else "/Library/Fonts/Inter_FXH-Regular.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    return _first_font(candidates, size)


def _mono(size: int) -> ImageFont.ImageFont:
    return _first_font(
        [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Supplemental/PTMono.ttc",
            "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
        ],
        size,
    )


def _first_font(paths: Iterable[str], size: int) -> ImageFont.ImageFont:
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_logo() -> Image.Image:
    logo = Image.open(LOGO).convert("RGBA")
    logo = logo.crop((72, 145, 468, 615))
    pixels = logo.load()
    for y in range(logo.height):
        for x in range(logo.width):
            r, g, b, a = pixels[x, y]
            if r > 218 and g > 218 and b > 205:
                pixels[x, y] = (r, g, b, 0)
    width = 72
    height = int(logo.height * width / logo.width)
    return logo.resize((width, height), Image.Resampling.LANCZOS)


def _opacity(image: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 1:
        return image
    out = image.copy()
    channel = out.getchannel("A").point(lambda value: int(value * max(0, min(1, alpha))))
    out.putalpha(channel)
    return out


def _appear(t: float, start: float, duration: float) -> float:
    return max(0.0, min(1.0, (t - start) / duration))


def _disappear(t: float, start: float, duration: float) -> float:
    return 1.0 - _appear(t, start, duration)


def _ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _lerp(a: float, b: float, x: float) -> float:
    return a + (b - a) * x


if __name__ == "__main__":
    main()
