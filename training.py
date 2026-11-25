"""
train_chars74k.py - Train the ESP32 OCR model on the Chars74K dataset.

Put this file in the same folder as the files you downloaded:
    EnglishFnt.tgz   (characters rendered from computer fonts)
    EnglishHnd.tgz   (handwritten characters)
    EnglishImg.tgz   (characters cropped from real photos)
    (ListsTXT.tgz is not needed)

Install once:
    pip install tensorflow pillow numpy

Run:
    python train_chars74k.py

The .tgz files are unpacked automatically the first time (into ./English).
Output:
    ocr.bin       -> copy to the ESP32 next to ocr.py
    samples.png   -> preview of what the model sees (20x20 images)
"""

import glob
import os
import random
import struct
import sys
import tarfile
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
# Characters to learn. Any subset of 0-9, A-Z, a-z.
# Tip: fewer characters = higher accuracy. Lowercase letters like o/O/0, l/1/I,
# s/S, c/C look almost the same, so leave lowercase out unless you need it.
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

S = 20               # model input is S x S (must match S in ocr.py)
HIDDEN = 256         # neurons in 1st hidden layer (64 = smaller/faster, 256 = more accurate/slower)
HIDDEN2 = 256        # neurons in 2nd hidden layer (0 = only one hidden layer)
EPOCHS = 25
OUT_FILE = "ocr.bin"

# Training data. Comment out a line to leave that source out.
#   max/char: use at most this many images per character (None = all)
#   augment:  extra random variations made of every training image
#   test:     fraction kept aside (never trained on) to measure accuracy at the end
# Keep "Img" first: the loading order decides the random test split.
SOURCES = {  # name:  archive,           folder,                    max/char, augment, test
    "Img": ("EnglishImg.tgz", "English/Img/GoodImg/Bmp",  None,     8,       0.15),  # real photos <- most important
    "Fnt": ("EnglishFnt.tgz", "English/Fnt",              400,      0,       0.0),   # fonts (~1000/char; limited so photos aren't drowned out)
    "Hnd": ("EnglishHnd.tgz", "English/Hnd/Img",          None,     3,       0.15),  # handwriting
}

# Chars74K folder SampleNNN -> character
ALL_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ----------------------------------------------------------------------------
# PREPROCESSING - must be IDENTICAL to ocr.py on the ESP32
# ----------------------------------------------------------------------------
def otsu(gray):
    """Pick the best black/white threshold automatically (same code as ocr.py)."""
    hist = np.bincount(gray.ravel(), minlength=256).tolist()
    total = gray.size
    sum_all = sum(i * h for i, h in enumerate(hist))
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


def text_mask(gray):
    """True where the text is. Works for dark-on-light AND light-on-dark."""
    t = otsu(gray)
    bright = gray > t
    border = np.concatenate([bright[0], bright[-1], bright[:, 0], bright[:, -1]])
    if border.mean() > 0.5:       # background is bright -> text is the dark part
        return ~bright
    return bright


def to_input(mask):
    """Crop to the character and sample it down to S x S (same maths as ocr.py)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    rows = y0 + (np.arange(S) * (y1 - y0)) // S
    cols = x0 + (np.arange(S) * (x1 - x0)) // S
    return mask[np.ix_(rows, cols)].astype(np.float32).flatten()


def prepare(img):
    """PIL image -> model input (or None)."""
    img = img.convert("L")
    # Big images (handwriting is 900x1200) are shrunk smoothly first so thin
    # strokes aren't lost. ESP32 camera crops are small, so this isn't needed there.
    m = max(img.size)
    if m > 64:
        k = 64 / m
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    return to_input(text_mask(np.array(img)))


def augment(img):
    """Random small changes so the model sees more variety."""
    img = img.convert("L")
    w, h = img.size
    # random padding with the edge colour, then rotate
    pad = int(max(w, h) * random.uniform(0.0, 0.15))
    bg = int(np.median(np.array(img)[0]))
    canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), bg)
    canvas.paste(img, (pad, pad))
    img = canvas.rotate(random.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=bg)
    # random stretch
    img = img.resize((max(4, int(img.width * random.uniform(0.8, 1.2))), img.height), Image.BILINEAR)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    return img


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------
def extract(tgz):
    if not os.path.exists(tgz):
        print(f"  {tgz} not found - skipping")
        return False
    print(f"  Unpacking {tgz} (only the first time, can take a few minutes)...")
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(".", filter="data")
        except TypeError:             # older Python without the filter option
            tf.extractall(".")
    return True


def load_source(name):
    """Load one Chars74K source (settings from SOURCES) -> (X_train, y_train, X_test, y_test).

    Test images are never augmented, so nothing about them leaks into training.
    """
    tgz, folder, max_per_char, n_aug, test_frac = SOURCES[name]
    Xtr, ytr, Xte, yte = [], [], [], []
    if not os.path.isdir(folder) and not extract(tgz):
        return Xtr, ytr, Xte, yte
    for ch in CHARS:
        sample = f"Sample{ALL_CHARS.index(ch) + 1:03d}"
        files = sorted(glob.glob(os.path.join(folder, sample, "*.png")))
        if max_per_char or test_frac:
            random.shuffle(files)
        files = files[:max_per_char]                     # None keeps all
        n_test = max(1, int(len(files) * test_frac)) if files and test_frac else 0
        label = CHARS.index(ch)
        for k, f in enumerate(files):
            try:
                img = Image.open(f)
                img.load()
            except Exception:
                continue
            x = prepare(img)
            if x is None:
                continue
            if k < n_test:
                Xte.append(x); yte.append(label)
                continue
            Xtr.append(x); ytr.append(label)
            for _ in range(n_aug):
                xa = prepare(augment(img))
                if xa is not None:
                    Xtr.append(xa); ytr.append(label)
    print(f"  {name}: {len(Xtr)} training images (with variations), {len(Xte)} kept for testing")
    return Xtr, ytr, Xte, yte


def save_preview(X, y, path="samples.png", n=12):
    rows = []
    for i in range(len(CHARS)):
        idx = np.where(y == i)[0][:n]
        tiles = [X[j].reshape(S, S) for j in idx] + [np.zeros((S, S))] * (n - len(idx))
        rows.append(np.hstack([np.pad(t, 1) for t in tiles]))
    grid = (np.vstack(rows) * 255).astype(np.uint8)
    Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(path)
    print("Saved preview:", path)


# ----------------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------------
def export(model, path):
    """Format: [num_layers] then per layer [in, out, W float32, b float32]."""
    dense = [l for l in model.layers if l.get_weights()]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(dense)))
        for l in dense:
            W, b = l.get_weights()
            f.write(struct.pack("<II", *W.shape))
            f.write(W.astype("<f4").tobytes())
            f.write(b.astype("<f4").tobytes())
    print(f"Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def run_bin(path, X):
    """Run ocr.bin with numpy exactly like the ESP32 does."""
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
    return h.argmax(1)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    bad = [c for c in CHARS if c not in ALL_CHARS]
    if bad:
        sys.exit(f"CHARS contains characters not in Chars74K: {bad}")
    random.seed(0)
    np.random.seed(0)

    print("Loading data...")
    X, y, tests = [], [], {}           # tests: source name -> (X_test, y_test)
    for name in SOURCES:
        a, b, c, d = load_source(name)
        X += a; y += b
        if c:
            tests[name] = (np.array(c, np.float32), np.array(d))
    if not X:
        sys.exit("No images loaded. Put the .tgz files in the same folder as this script.")

    X, y = np.array(X, np.float32), np.array(y, np.int32)
    p = np.random.permutation(len(X))
    X, y = X[p], y[p]
    save_preview(X, y)

    import tensorflow as tf
    layers = [
        tf.keras.layers.Input((S * S,)),
        tf.keras.layers.Dense(HIDDEN, activation="relu"),
        tf.keras.layers.Dropout(0.3),        # training only; not exported
    ]
    if HIDDEN2:
        layers += [
            tf.keras.layers.Dense(HIDDEN2, activation="relu"),
            tf.keras.layers.Dropout(0.3),
        ]
    layers.append(tf.keras.layers.Dense(len(CHARS), activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=EPOCHS, batch_size=128, validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

    export(model, OUT_FILE)

    # Test each source separately on images it has never seen, so the
    # real-photo score (what matters for the camera) isn't mixed with the others.
    if tests:
        print("\nAccuracy on test images it has never seen:")
    for name, (Xte, yte) in tests.items():
        pred = run_bin(OUT_FILE, Xte)
        print(f"  {name}: {(pred == yte).mean() * 100:.1f}%  ({len(Xte)} images)")
        mistakes = Counter((CHARS[t], CHARS[p]) for t, p in zip(yte, pred) if t != p)
        if mistakes:
            print("    most common mistakes (real -> guessed):",
                  ", ".join(f"{a}->{b} x{n}" for (a, b), n in mistakes.most_common(8)))

    print(f"\nDone. Copy {OUT_FILE} to the ESP32.")
    print(f"In ocr.py set:  CHARS = {CHARS!r}")


if __name__ == "__main__":
    main()