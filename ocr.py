# ocr.py - runs on the ESP32 (MicroPython with ulab)
#
# Usage:
#   import ocr
#   m = ocr.load()                          # loads ocr.bin
#   gray, w, h = ocr.load_pgm("test.pgm")   # or a grayscale camera frame
#   print(ocr.read_text(m, gray, w, h))     # a line of text
#   print(ocr.read_char(m, gray, w, h))     # an image of one character

from ulab import numpy as np
import struct

# Must match CHARS and S in train_chars74k.py
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
S = 20


# ---------------------------------------------------------------- model
def load(path="ocr.bin"):
    layers = []
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            i, o = struct.unpack("<II", f.read(8))
            W = np.frombuffer(f.read(i * o * 4), dtype=np.float).reshape((i, o))
            b = np.frombuffer(f.read(o * 4), dtype=np.float)
            layers.append((W, b))
    n_out = layers[-1][1].shape[0] if hasattr(layers[-1][1], "shape") else len(layers[-1][1])
    if n_out != len(CHARS):
        print("WARNING: model has", n_out, "outputs but CHARS has", len(CHARS))
    return layers


def classify(layers, x):
    """Returns (character, confidence 0..1)."""
    for k, (W, b) in enumerate(layers):
        x = np.dot(x, W) + b
        if k < len(layers) - 1:
            x = np.maximum(x, 0)
    i = int(np.argmax(x))
    e = np.exp(x - np.max(x))
    return CHARS[i], float(e[i] / np.sum(e))


# ---------------------------------------------------------------- image prep
def otsu(gray):
    """Automatic black/white threshold (same as the PC training code)."""
    hist = [0] * 256
    for p in gray:
        hist[p] += 1
    total = len(gray)
    sum_all = 0
    for i in range(256):
        sum_all += i * hist[i]
    wb = 0
    sb = 0
    best = -1.0
    thr = 0
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        mb = sb / wb
        mf = (sum_all - sb) / wf
        v = wb * wf * (mb - mf) ** 2
        if v > best:
            best = v
            thr = t
    return thr


def text_mask(gray, w, h):
    """bytearray with 1 where the text is. Handles dark-on-light and light-on-dark."""
    t = otsu(gray)
    bright = 0
    n = 0
    for x in range(w):
        bright += (gray[x] > t) + (gray[(h - 1) * w + x] > t)
    for y in range(h):
        bright += (gray[y * w] > t) + (gray[y * w + w - 1] > t)
    n = 2 * w + 2 * h
    dark_text = bright * 2 > n          # background is bright -> text is dark
    mask = bytearray(w * h)
    for i in range(w * h):
        mask[i] = (gray[i] <= t) if dark_text else (gray[i] > t)
    return mask


def to_input(mask, w, x0, x1, y0, y1):
    """Crop box -> S x S model input (same maths as the PC)."""
    out = np.zeros(S * S)
    for r in range(S):
        sy = y0 + r * (y1 - y0) // S
        row = sy * w
        for c in range(S):
            sx = x0 + c * (x1 - x0) // S
            if mask[row + sx]:
                out[r * S + c] = 1.0
    return out


def bbox(mask, w, x0, x1, y0, y1):
    """Tight box around the text pixels inside the given region, or None."""
    top = bot = left = right = -1
    for y in range(y0, y1):
        row = y * w
        for x in range(x0, x1):
            if mask[row + x]:
                if top < 0:
                    top = y
                bot = y
                if left < 0 or x < left:
                    left = x
                if x > right:
                    right = x
    if top < 0:
        return None
    return left, right + 1, top, bot + 1


def segment(mask, w, h, min_width=2):
    """Split a line of text into characters at empty columns."""
    spans = []
    start = None
    for x in range(w + 1):
        on = False
        if x < w:
            for y in range(h):
                if mask[y * w + x]:
                    on = True
                    break
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= min_width:
                spans.append((start, x))
            start = None
    return spans


# ---------------------------------------------------------------- main API
def read_char(layers, gray, w, h):
    """Image containing ONE character -> (char, confidence)."""
    mask = text_mask(gray, w, h)
    b = bbox(mask, w, 0, w, 0, h)
    if b is None:
        return "", 0.0
    return classify(layers, to_input(mask, w, *b))


def read_text(layers, gray, w, h, min_conf=0.0):
    """Image containing ONE LINE of text -> string."""
    mask = text_mask(gray, w, h)
    text = ""
    for x0, x1 in segment(mask, w, h):
        b = bbox(mask, w, x0, x1, 0, h)
        if b is None:
            continue
        ch, conf = classify(layers, to_input(mask, w, *b))
        text += ch if conf >= min_conf else "?"
    return text


def load_pgm(path):
    """Load a binary PGM (P5) grayscale image -> (bytearray, w, h)."""
    with open(path, "rb") as f:
        data = f.read()
    fields = []
    i = 0
    while len(fields) < 4:
        while data[i] in b" \t\r\n":
            i += 1
        if data[i] == ord("#"):
            while data[i] not in b"\r\n":
                i += 1
            continue
        j = i
        while data[j] not in b" \t\r\n":
            j += 1
        fields.append(data[i:j])
        i = j
    if fields[0] != b"P5":
        raise ValueError("not a binary PGM (P5)")
    w, h = int(fields[1]), int(fields[2])
    i += 1
    return bytearray(data[i:i + w * h]), w, h