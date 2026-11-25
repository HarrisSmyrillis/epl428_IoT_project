# ocr.py - runs on the ESP32 (MicroPython with ulab)
#
# Reads text from image files on the board: PNG, BMP or PGM/PPM (not JPEG -
# the ESP32 can't decode it; save JPEGs as PNG on the PC first).
# Images are turned into grayscale and shrunk on the board, then read by
# the model trained with training.py (ocr.bin).
#
# Run it (from the PC, with the board connected):
#     mpremote cp ocr.py ocr.bin :          # once
#     mpremote cp photo.png :               # every image you want to read
#     mpremote run ocr.py                   # asks which image to read
#
# Or use it from your own code:
#   import ocr
#   m = ocr.load()                            # loads ocr.bin
#   ocr.predict_file(m, "photo.png")          # load + read + print, returns the text
#   gray, w, h = ocr.load_image("photo.png")  # or a grayscale camera frame
#   print(ocr.read_text(m, gray, w, h))       # a line of text
#   print(ocr.read_char(m, gray, w, h))       # an image of one character

from ulab import numpy as np
import gc
import os
import struct
import time

# Must match CHARS and S in training.py
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
S = 20


# ---------------------------------------------------------------- model
def load(path="ocr.bin"):
    # ocr.bin holds 32-bit floats. Firmware built with 64-bit floats would
    # read them wrongly, so check instead of giving silent nonsense.
    z = np.zeros(1, dtype=np.float)
    if hasattr(z, "itemsize") and z.itemsize != 4:
        raise ValueError("this firmware uses 64-bit floats; ocr.bin needs 32-bit (standard ESP32 builds)")
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
    x = x.flatten()                     # in case dot() returned shape (1, n) instead of (n,)
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


def predict_file(layers, path, min_conf=0.5):
    """Load an image file, read it and print the result. Returns the text.
    Characters the model is less sure about than min_conf are shown as '?'."""
    gc.collect()
    try:
        t0 = _ms()
        gray, w, h = load_image(path)
        text = read_text(layers, gray, w, h, min_conf)
        ms = _ms_since(t0)
    except MemoryError:
        print(path + ": out of memory - use a smaller image (or lower MAX_SIDE)")
        return ""
    except (OSError, ValueError) as e:
        print(path + ": " + str(e))
        return ""
    line = "{}: '{}'".format(path, text)
    if len(text) == 1:                          # single character: show how sure it is
        line += "  ({:.0f}% sure)".format(read_char(layers, gray, w, h)[1] * 100)
    print(line + "  [{}x{}, {} ms]".format(w, h, ms))
    return text


def main():
    """Ask for image files on the board and read them, until Enter is pressed."""
    layers = load()
    print("Model loaded:", len(layers), "layers, characters:", CHARS)
    if hasattr(gc, "mem_free"):
        print("Free memory:", gc.mem_free(), "bytes")
    while True:
        images = [f for f in sorted(os.listdir()) if _ext(f) in IMAGE_EXT]
        print("\nImages on the board:", ", ".join(images) if images else "(none - copy some with mpremote cp)")
        name = input("Image to read (Enter = quit): ").strip()
        if not name:
            break
        predict_file(layers, name)


# ---------------------------------------------------------------- image files
# Every format is turned into grayscale (0 = black .. 255 = white) and shrunk
# while it is read, so even big images need little memory.
MAX_SIDE = 200      # longest side after shrinking. ocr.py loops over every pixel,
                    # so smaller = faster. Raise it if characters come out too small.
IMAGE_EXT = ("png", "bmp", "pgm", "ppm", "jpg", "jpeg")


def load_image(path, max_side=MAX_SIDE):
    """Image file (PNG, BMP, PGM, PPM) -> (gray bytearray, w, h)."""
    with open(path, "rb") as f:
        head = f.read(8)
        f.seek(0)
        if head[:2] == b"BM":
            return _load_bmp(f, max_side)
        if head[:2] in (b"P5", b"P6"):
            return _load_pnm(f, max_side)
        if head == b"\x89PNG\r\n\x1a\n":
            return _load_png(f, max_side)
        if head[:2] == b"\xff\xd8":
            raise ValueError("JPEG can't be decoded on the ESP32 - save it as PNG on the PC first")
    raise ValueError("unsupported image format - use PNG, BMP, PGM or PPM")


def _ext(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.time() * 1000)


def _ms_since(t0):
    # ticks_ms() wraps around, so MicroPython needs ticks_diff() for differences
    return time.ticks_diff(_ms(), t0) if hasattr(time, "ticks_diff") else _ms() - t0


def _gray(r, g, b):
    return (r * 77 + g * 150 + b * 29) >> 8


def _out_size(w, h, max_side):
    m = max(w, h)
    if m <= max_side:
        return w, h
    return max(1, w * max_side // m), max(1, h * max_side // m)


def _shrink(w, h, max_side, get_row, pixel):
    """Build the output image. get_row(y) returns source row y (rows are asked
    for in increasing order), pixel(row, x) returns its gray value."""
    ow, oh = _out_size(w, h, max_side)
    out = bytearray(ow * oh)
    xs = [c * w // ow for c in range(ow)]
    for r in range(oh):
        row = get_row(r * h // oh)
        base = r * ow
        for c in range(ow):
            out[base + c] = pixel(row, xs[c])
    return out, ow, oh


def _load_bmp(f, max_side):
    hdr = f.read(54)
    offset = struct.unpack("<I", hdr[10:14])[0]
    dib = struct.unpack("<I", hdr[14:18])[0]
    w, h = struct.unpack("<ii", hdr[18:26])
    bpp = struct.unpack("<H", hdr[28:30])[0]
    comp = struct.unpack("<I", hdr[30:34])[0]
    if comp not in (0, 3) or bpp not in (1, 4, 8, 24, 32):
        raise ValueError("unsupported BMP (use 8, 24 or 32 bit, uncompressed)")
    top_down = h < 0
    h = abs(h)
    stride = (w * bpp + 31) // 32 * 4
    pal = None
    if bpp <= 8:
        n = struct.unpack("<I", hdr[46:50])[0] or (1 << bpp)
        f.seek(14 + dib)
        p = f.read(4 * n)
        pal = bytes([_gray(p[4 * i + 2], p[4 * i + 1], p[4 * i]) for i in range(n)])

    def get_row(y):
        f.seek(offset + (y if top_down else h - 1 - y) * stride)
        return f.read(stride)

    if bpp == 24:
        pixel = lambda row, x: _gray(row[3 * x + 2], row[3 * x + 1], row[3 * x])
    elif bpp == 32:
        pixel = lambda row, x: _gray(row[4 * x + 2], row[4 * x + 1], row[4 * x])
    elif bpp == 8:
        pixel = lambda row, x: pal[row[x]]
    elif bpp == 4:
        pixel = lambda row, x: pal[(row[x >> 1] >> (4 - 4 * (x & 1))) & 15]
    else:
        pixel = lambda row, x: pal[(row[x >> 3] >> (7 - (x & 7))) & 1]
    return _shrink(w, h, max_side, get_row, pixel)


def _load_pnm(f, max_side):
    head = f.read(512)
    space = (32, 9, 13, 10)                         # ' ', tab, CR, LF as byte values
    fields = []
    i = 0
    while len(fields) < 4:
        while head[i] in space:
            i += 1
        if head[i] == 35:                           # '#' comment until end of line
            while head[i] not in (13, 10):
                i += 1
            continue
        j = i
        while head[j] not in space:
            j += 1
        fields.append(head[i:j].decode())
        i = j
    w, h, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    if maxval > 255:
        raise ValueError("16-bit PGM/PPM not supported")
    start = i + 1
    ch = 1 if fields[0] == "P5" else 3

    def get_row(y):
        f.seek(start + y * w * ch)
        return f.read(w * ch)

    if ch == 1:
        pixel = lambda row, x: row[x]
    else:
        pixel = lambda row, x: _gray(row[3 * x], row[3 * x + 1], row[3 * x + 2])
    return _shrink(w, h, max_side, get_row, pixel)


def _inflate(data):
    """zlib-compressed bytes -> stream of the uncompressed bytes."""
    import io
    try:
        import deflate                              # MicroPython 1.21 and newer
        return deflate.DeflateIO(io.BytesIO(data), deflate.ZLIB)
    except ImportError:
        pass
    try:
        import zlib
    except ImportError:
        import uzlib as zlib                        # older MicroPython names it uzlib
    if hasattr(zlib, "DecompIO"):                   # older MicroPython
        return zlib.DecompIO(io.BytesIO(data), 15)
    return io.BytesIO(zlib.decompress(data))        # PC (for testing)


def _read_exact(s, n):
    b = s.read(n)
    while len(b) < n:
        more = s.read(n - len(b))
        if not more:
            raise ValueError("PNG data ended early")
        b += more
    return b


def _load_png(f, max_side):
    f.read(8)
    idat = []
    pal = None
    while True:
        n = struct.unpack(">I", f.read(4))[0]
        typ = f.read(4)
        data = f.read(n)
        f.read(4)                                   # CRC
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", data)
        elif typ == b"PLTE":
            pal = bytes([_gray(data[3 * i], data[3 * i + 1], data[3 * i + 2]) for i in range(n // 3)])
        elif typ == b"IDAT":
            idat.append(data)
        elif typ == b"IEND":
            break
    if interlace:
        raise ValueError("interlaced PNG not supported - save it again without interlacing")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    if depth < 8 and ctype not in (0, 3):
        raise ValueError("unsupported PNG bit depth")
    stride = (w * channels * depth + 7) // 8
    bpp = max(1, channels * depth // 8)             # bytes per pixel, for the filters
    stream = _inflate(b"".join(idat))
    del idat
    gc.collect()
    state = {"y": -1, "row": bytearray(stride)}

    def get_row(y):
        # PNG rows depend on the row above, so every row up to y is decoded.
        while state["y"] < y:
            prev = state["row"]
            ft = _read_exact(stream, 1)[0]
            line = bytearray(_read_exact(stream, stride))
            if ft == 1:                             # Sub
                for i in range(bpp, stride):
                    line[i] = (line[i] + line[i - bpp]) & 255
            elif ft == 2:                           # Up
                for i in range(stride):
                    line[i] = (line[i] + prev[i]) & 255
            elif ft == 3:                           # Average
                for i in range(stride):
                    a = line[i - bpp] if i >= bpp else 0
                    line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
            elif ft == 4:                           # Paeth
                for i in range(stride):
                    a = line[i - bpp] if i >= bpp else 0
                    b = prev[i]
                    c = prev[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    line[i] = (line[i] + pr) & 255
            state["row"] = line
            state["y"] += 1
        return state["row"]

    if depth < 8:                                   # 1/2/4-bit gray or palette
        per_byte = 8 // depth
        mask = (1 << depth) - 1

        def pixel(row, x):
            v = (row[x // per_byte] >> (8 - depth * (x % per_byte + 1))) & mask
            return pal[v] if ctype == 3 else v * 255 // mask
    else:
        k = depth // 8                              # 16-bit: use the high byte

        def pixel(row, x):
            i = x * channels
            if ctype == 3:
                return pal[row[x]]
            if ctype in (0, 4):
                v = row[i * k]
            else:
                v = _gray(row[i * k], row[(i + 1) * k], row[(i + 2) * k])
            if ctype in (4, 6):                     # transparency: put it on white
                a = row[(i + channels - 1) * k]
                v = (v * a + 255 * (255 - a)) // 255
            return v
    return _shrink(w, h, max_side, get_row, pixel)


if __name__ == "__main__":
    main()