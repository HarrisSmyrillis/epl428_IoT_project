"""
train.py - Train a tiny character-OCR model on your PC for the ESP32 (MicroPython + ulab).

Install once:
    pip install tensorflow pillow numpy

Run:
    python train.py

Output:
    ocr.bin          -> copy to the ESP32 (same folder as ocr.py)
    samples.png      -> preview of what the model is trained on
"""

import glob
import os
import random
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------
# SETTINGS - change these
# ----------------------------------------------------------------------------
CHARS = "0123456789"        # characters to recognise (must match CHARS in ocr.py)
S = 20                      # input size 20x20 (must match S in ocr.py)
SAMPLES_PER_CHAR = 3000     # synthetic images per character
EPOCHS = 15
HIDDEN = 64                 # neurons in the hidden layer (more = bigger + slower)
OUT_FILE = "ocr.bin"

# Optional: put your own .ttf/.otf fonts in ./fonts (best: fonts that look like
# the text you will actually read).
# Optional: put REAL cropped character images in ./data/<char>/*.png,
# e.g. data/7/img001.png. These are used in addition to synthetic data.
FONT_DIR = "fonts"
REAL_DIR = "data"

# Fonts tried automatically if ./fonts is empty (Windows / Mac / Linux)
SYSTEM_FONTS = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


# ----------------------------------------------------------------------------
# PREPROCESSING - identical to preprocess() in ocr.py on the ESP32
# Input: 2D uint8 array, BRIGHT text on DARK background.
# ----------------------------------------------------------------------------
def preprocess(gray, thr=128):
    on = gray > thr
    ys, xs = np.nonzero(on)
    if len(xs) == 0:
        return np.zeros(S * S, np.float32)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    out = np.zeros((S, S), np.float32)
    for r in range(S):
        sy = y0 + r * (y1 - y0) // S
        for c in range(S):
            sx = x0 + c * (x1 - x0) // S
            out[r, c] = 1.0 if on[sy, sx] else 0.0
    return out.flatten()


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def find_fonts():
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_DIR, "*.otf"))
    if not fonts:
        fonts = [f for f in SYSTEM_FONTS if os.path.exists(f)]
    if not fonts:
        sys.exit("No fonts found. Create a 'fonts' folder next to train.py and put some .ttf files in it.")
    print("Using fonts:", *[os.path.basename(f) for f in fonts])
    return fonts


def synth(ch, fonts):
    """Render one character with random font, size, position, rotation, thickness, blur, noise."""
    img = Image.new("L", (64, 64), 0)
    font = ImageFont.truetype(random.choice(fonts), random.randint(26, 44))
    ImageDraw.Draw(img).text((random.randint(6, 18), random.randint(0, 8)), ch, fill=255, font=font)
    img = img.rotate(random.uniform(-10, 10), resample=Image.BILINEAR)
    # random stretch (different aspect ratios)
    w, h = img.size
    img = img.resize((int(w * random.uniform(0.8, 1.2)), h), Image.BILINEAR)
    if random.random() < 0.25:
        img = img.filter(ImageFilter.MaxFilter(3))      # thicker strokes
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    a = np.array(img, np.float32)
    a = a * random.uniform(0.85, 1.0) + random.uniform(0, 20)  # contrast / brightness
    a += np.random.normal(0, random.uniform(2, 12), a.shape)   # noise
    x = preprocess(np.clip(a, 0, 255).astype(np.uint8))
    if x.sum() < 10:                    # character got lost -> try again
        return synth(ch, fonts)
    return x


def load_real():
    """Load real images from data/<char>/*. Auto-inverts dark-text-on-light images."""
    X, y = [], []
    for i, ch in enumerate(CHARS):
        for path in glob.glob(os.path.join(REAL_DIR, ch, "*")):
            try:
                a = np.array(Image.open(path).convert("L"))
            except Exception:
                continue
            if a.mean() > 127:          # mostly light background -> invert
                a = 255 - a
            x = preprocess(a)
            for _ in range(20):         # repeat real samples so they count more
                X.append(x); y.append(i)
    if X:
        print(f"Loaded {len(X) // 20} real images from ./{REAL_DIR}")
    return X, y


def save_preview(X, y, path="samples.png", n=12):
    rows = []
    for i in range(len(CHARS)):
        idx = np.where(y == i)[0][:n]
        tiles = [X[j].reshape(S, S) for j in idx]
        tiles += [np.zeros((S, S))] * (n - len(tiles))
        rows.append(np.hstack([np.pad(t, 1) for t in tiles]))
    grid = (np.vstack(rows) * 255).astype(np.uint8)
    Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST).save(path)
    print("Saved preview:", path)


# ----------------------------------------------------------------------------
# EXPORT + CHECK
# ----------------------------------------------------------------------------
def export(model, path):
    """Format: [num_layers] then per layer [in, out, W float32 (in*out), b float32 (out)]."""
    dense = [l for l in model.layers if l.get_weights()]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(dense)))
        for l in dense:
            W, b = l.get_weights()
            f.write(struct.pack("<II", *W.shape))
            f.write(W.astype("<f4").tobytes())
            f.write(b.astype("<f4").tobytes())
    print(f"Saved {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def check_export(path, X, y):
    """Run ocr.bin with plain numpy, the same way the ESP32 does, to verify the file."""
    layers = []
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            i, o = struct.unpack("<II", f.read(8))
            W = np.frombuffer(f.read(i * o * 4), "<f4").reshape(i, o)
            b = np.frombuffer(f.read(o * 4), "<f4")
            layers.append((W, b))
    h = X
    for k, (W, b) in enumerate(layers):
        h = h @ W + b
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    acc = (h.argmax(1) == y).mean()
    print(f"Accuracy using ocr.bin (ESP32-style inference): {acc * 100:.1f}%")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    import tensorflow as tf

    fonts = find_fonts()
    print(f"Generating {SAMPLES_PER_CHAR} images for each of {len(CHARS)} characters...")
    X, y = [], []
    for i, ch in enumerate(CHARS):
        for _ in range(SAMPLES_PER_CHAR):
            X.append(synth(ch, fonts)); y.append(i)
        print(f"  '{ch}' done")
    Xr, yr = load_real()
    X += Xr; y += yr
    X, y = np.array(X, np.float32), np.array(y, np.int32)

    p = np.random.permutation(len(X))
    X, y = X[p], y[p]
    split = int(len(X) * 0.9)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    save_preview(Xtr, ytr)

    model = tf.keras.Sequential([
        tf.keras.layers.Input((S * S,)),
        tf.keras.layers.Dense(HIDDEN, activation="relu"),
        tf.keras.layers.Dropout(0.2),       # only used during training; not exported
        tf.keras.layers.Dense(len(CHARS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=64, validation_data=(Xte, yte))

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.1f}%")

    export(model, OUT_FILE)
    check_export(OUT_FILE, Xte, yte)
    print(f"\nDone. Copy {OUT_FILE} to the ESP32. CHARS on the ESP32 must be: {CHARS!r}")


if __name__ == "__main__":
    main()