"""
synth_plate_generator.py

Week 9, V1.1 dataset strategy, item 1: generates synthetic Alberta-style
license plate crop images with realistic degradations, for a balanced
6-character / 7-character training set covering multiple plausible
letter/digit layouts (not one fixed pattern).

Plate patterns are drawn from what's actually been observed in this
project's real local plates (13 examples, held out from training):
    7-char: LLLDDDD (e.g. CMK7507, CTY8283, CSV4780, BPM9818, CYZ7703, BCP7506)
    6-char: LLLDDD (BPD845, NEL248), LLLLDD (DJPT60), DDDLLL (726NNS, 242MMQ)
A couple of additional plausible variants are included per pattern length
so the model isn't taught a single rigid layout, per the plan.

Degradations (each applied with randomized parameters, independently
toggled per image so a "substantial clean subset" is retained too):
    blur, perspective warp, exposure/brightness shift, glare (specular
    highlight), Gaussian noise, JPEG re-compression artifacts, downscale/
    upscale (simulates low resolution).

Character set matches plate_config.yaml exactly: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
(pad char '_' excluded — that's a training-time padding token, never part
of generated plate text).
"""

import io
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# (pattern using 'L' for letter, 'D' for digit) -> approximate real-world weight
PATTERNS_7 = ["LLLDDDD", "LLLDDDD", "LLLDDDD", "LDDDDDD", "LLDDDDD"]  # LLLDDDD weighted heaviest (matches all 6 observed 7-char examples)
PATTERNS_6 = ["LLLDDD", "LLLLDD", "DDDLLL", "LLDDDD", "DDLLLL"]


def random_plate_text(length: int) -> str:
    pattern = random.choice(PATTERNS_7 if length == 7 else PATTERNS_6)
    return "".join(random.choice(LETTERS) if ch == "L" else random.choice(DIGITS) for ch in pattern)


def _find_fonts(max_fonts: int = 8) -> list:
    """Searches common system font directories at runtime (not hardcoded
    to any one machine) for bold/condensed sans-serif TTFs — a reasonable
    visual approximation of plate lettering. Falls back to PIL's built-in
    font if nothing is found."""
    search_dirs = ["/usr/share/fonts", "/usr/local/share/fonts", "/mnt/skills"]
    candidates = []
    preferred_keywords = ["bold", "condensed", "mono", "black", "heavy"]
    for d in search_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ttf in p.rglob("*.ttf"):
            name_lower = ttf.name.lower()
            if any(k in name_lower for k in preferred_keywords):
                candidates.append(ttf)
    random.shuffle(candidates)
    return candidates[:max_fonts] if candidates else []


_FONT_PATHS = _find_fonts()


def _get_font(size: int):
    if _FONT_PATHS:
        path = random.choice(_FONT_PATHS)
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _random_background_color():
    # mostly white/near-white plate backgrounds, occasional light tint
    if random.random() < 0.85:
        base = random.randint(230, 255)
        return (base, base, base)
    return (random.randint(200, 240), random.randint(210, 245), random.randint(220, 255))


def _random_text_color():
    # dark text — mostly black/near-black, occasional dark blue
    if random.random() < 0.8:
        v = random.randint(0, 40)
        return (v, v, v)
    return (random.randint(0, 30), random.randint(0, 30), random.randint(60, 110))


def render_clean_plate(text: str, width: int = 320, height: int = 100) -> Image.Image:
    """Renders one plate image with random font/background/text color/
    spacing, no degradations applied yet. Font size is auto-fit to the
    plate width for the given text length, so longer strings don't
    overflow the plate boundary."""
    bg = _random_background_color()
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    border_color = tuple(max(0, c - 60) for c in bg)
    border_width = random.randint(2, 5)
    draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=border_width)

    margin = border_width + 8
    max_text_width = width - 2 * margin
    letter_spacing = random.randint(2, 8)
    font_path = random.choice(_FONT_PATHS) if _FONT_PATHS else None

    # shrink font size until the rendered text fits max_text_width for
    # THIS text's length (character widths vary by font, so this is
    # computed per-image, not assumed from a fixed size)
    font_size = int(height * 0.72)
    font, char_widths, total_text_width = None, [], 0
    while font_size > 10:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
        char_widths = [draw.textbbox((0, 0), ch, font=font)[2] - draw.textbbox((0, 0), ch, font=font)[0] for ch in text]
        total_text_width = sum(char_widths) + letter_spacing * (len(text) - 1)
        if total_text_width <= max_text_width:
            break
        font_size -= 2

    text_color = _random_text_color()
    x = margin + max(0, (max_text_width - total_text_width) // 2)
    y_jitter = random.randint(-2, 2)
    for ch, cw in zip(text, char_widths):
        bbox = draw.textbbox((0, 0), ch, font=font)
        y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1] + y_jitter
        draw.text((x, y), ch, font=font, fill=text_color)
        x += cw + letter_spacing

    return img


def _apply_perspective(img: Image.Image, strength: float = 0.06) -> Image.Image:
    w, h = img.size
    dx, dy = w * strength, h * strength
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.uniform(-dx, dx), random.uniform(-dy, dy)),
        (w + random.uniform(-dx, dx), random.uniform(-dy, dy)),
        (w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)),
        (random.uniform(-dx, dx), h + random.uniform(-dy, dy)),
    ]
    # PIL's QUAD transform maps a quadrilateral in the SOURCE to the output
    # rectangle, so we invert the roles: distort by sampling from `dst`.
    coeffs = _find_quad_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(220, 220, 220))


def _find_quad_coeffs(src_pts, dst_pts):
    """Standard 8-coefficient perspective transform solve (PIL recipe)."""
    matrix = []
    for (x, y), (X, Y) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(dst_pts, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B) if A.shape[0] == A.shape[1] else np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()


def _apply_glare(img: Image.Image) -> Image.Image:
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = random.randint(0, w), random.randint(0, h)
    r = random.randint(int(w * 0.15), int(w * 0.4))
    alpha = random.randint(60, 140)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=r * 0.4))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _apply_noise(img: Image.Image, sigma: float = 12.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_low_resolution(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _apply_exposure(img: Image.Image, factor: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_random_degradations(img: Image.Image, clean_probability: float = 0.2) -> Image.Image:
    """With clean_probability chance, returns the image unmodified (the
    'substantial clean subset' the plan asks for). Otherwise applies a
    random subset of degradations, each independently toggled."""
    if random.random() < clean_probability:
        return img

    if random.random() < 0.5:
        img = _apply_perspective(img, strength=random.uniform(0.02, 0.08))
    if random.random() < 0.4:
        img = _apply_exposure(img, factor=random.uniform(0.6, 1.5))
    if random.random() < 0.25:
        img = _apply_glare(img)
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.2)))
    if random.random() < 0.35:
        img = _apply_noise(img, sigma=random.uniform(5, 18))
    if random.random() < 0.3:
        img = _apply_low_resolution(img, scale=random.uniform(0.3, 0.7))
    if random.random() < 0.3:
        img = _apply_jpeg_compression(img, quality=random.randint(25, 70))

    return img


def generate_plate_image(text: str, width: int = 320, height: int = 100, clean_probability: float = 0.2) -> Image.Image:
    img = render_clean_plate(text, width=width, height=height)
    img = apply_random_degradations(img, clean_probability=clean_probability)
    return img
