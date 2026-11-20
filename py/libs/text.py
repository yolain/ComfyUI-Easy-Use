"""Arabic text on Pillow images, for the Table Editor and the XY plot labels.

Pillow only joins Arabic letters and lays them out right to left when it was built
with Raqm and the system has FriBiDi, which the Windows wheels (ComfyUI portable)
don't, and the bundled wenquan.ttf and OpenSans-Medium.ttf have no Arabic letters.
khatt shapes Arabic itself and draws the letters a font lacks with Noto Naskh
Arabic. Text without Arabic letters goes to Pillow exactly as before.
"""

import unicodedata

from PIL import Image, ImageDraw

try:
    from khatt.pillow import text as _khatt_text
    from khatt.pillow import textbbox as _khatt_textbbox
except ImportError:  # without khatt, Pillow draws everything as before
    _khatt_text = None
    _khatt_textbbox = None

_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def has_arabic(text):
    """Whether text has Arabic letters (only when khatt is there to draw them)."""
    return _khatt_text is not None and any(unicodedata.bidirectional(char) == "AL" for char in text)


def reads_right_to_left(text):
    """Whether text is Arabic that reads right to left: its first letter is Arabic."""
    if not has_arabic(text):
        return False
    for char in text:
        kind = unicodedata.bidirectional(char)
        if kind == "AL":
            return True
        if kind in ("L", "R"):
            return False
    return False


def draw_text(draw, xy, text, **kwargs):
    """``draw.text(xy, text, **kwargs)``, with Arabic drawn by khatt."""
    if has_arabic(text):
        _khatt_text(draw, xy, text, **kwargs)
    else:
        draw.text(xy, text, **kwargs)


def textbbox(draw, xy, text, font):
    """``draw.textbbox(xy, text, font=font)``, measuring Arabic as draw_text draws it."""
    if has_arabic(text):
        return _khatt_textbbox(draw, xy, text, font=font)
    return draw.textbbox(xy, text, font=font)


def getbbox(font, text):
    """``font.getbbox(text)``, measuring Arabic as draw_text draws it."""
    if has_arabic(text):
        return _khatt_textbbox(_MEASURE, (0, 0), text, font=font)
    return font.getbbox(text)
