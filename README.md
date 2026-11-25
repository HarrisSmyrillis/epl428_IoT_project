# ESP32 OCR: Character Recognition on a Microcontroller

A small optical character recognition (OCR) system that runs entirely on an
ESP32. A neural network is trained on a PC, exported to a compact binary file,
and run on the board in MicroPython to read characters (**0–9 and A–Z**) from
images. The prediction is printed and shown on the board's OLED display.

Developed for EPL428 (IoT) and tested on a **LilyGO T-Beam Supreme** (ESP32-S3,
1.3" SH1106 OLED).

```
PC:     Chars74K images ──► training.py ──► ocr.bin  (trained network)
                                               │  upload to the board
ESP32:  image file ──► ocr.py ──► grayscale ──► black/white ──► split into
        characters ──► neural network ──► "HELLO42"  ──► Shell + OLED
```

## Repository contents

| File | Runs on | Purpose |
|---|---|---|
| `training.py` | PC | Trains the network on the Chars74K dataset and writes `ocr.bin` |
| `ocr.py` | ESP32 (MicroPython + ulab) | Loads `ocr.bin`, reads image files, shows results on the OLED |
| `ocr.bin` | ESP32 | The trained network (weights) |
| `samples.png` | — | Preview of the 20×20 images the network is trained on |
| `test_images/` | ESP32 | Example images to try on the board |

## Results

Accuracy is measured on images held back from training (never seen by the
model):

| Test set | Accuracy |
|---|---|
| Real-world photos (Chars74K `Img`) | ~88% |
| Handwriting (Chars74K `Hnd`) | ~93% |

Most remaining errors are characters that look the same, such as `0`/`O`,
`1`/`I` and `5`/`S`. The test split is random on every run, so results vary by
about ±1–2% between runs.

---

## 1. Training the model (PC)

### Requirements

- Python 3.10+
- `pip install tensorflow pillow numpy`

### Dataset

The model is trained on the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).
Download these three files into the repository folder:

- `EnglishImg.tgz` – characters cropped from real photos (most important)
- `EnglishFnt.tgz` – characters rendered from computer fonts
- `EnglishHnd.tgz` – handwritten characters

They are unpacked automatically into `English/` on the first run. The dataset
is **not** stored in git (see `.gitignore`), because the archives are larger
than GitHub's 100 MB file limit.

### Run

```
python training.py
```

This produces:

- `ocr.bin`: the trained network, to copy to the board
- `samples.png`: a preview of what the network sees
- the accuracy on unseen photos and handwriting, with the most common mistakes

### How training works

1. **Load and split.** 15% of the photos and of the handwriting are randomly
   held back for testing. Fonts are limited to 400 per character so they don't
   drown out the photos.
2. **Augment.** Each training photo gets 8 random variations and each
   handwritten image 3 (rotation, stretch, padding, blur).
3. **Preprocess** every image exactly as `ocr.py` does on the board: grayscale,
   Otsu threshold to black and white (works for dark-on-light and
   light-on-dark text), crop to the character, sample down to 20×20.
4. **Train** a fully connected network: 400 inputs → 256 → 256 → 36 outputs,
   ReLU, dropout 0.3, Adam optimizer, cross-entropy loss, early stopping.
5. **Export** the weights to `ocr.bin`, then reload that file and test it with
   plain numpy, exactly like the board will run it.

### Settings (top of `training.py`)

| Setting | Default | Meaning |
|---|---|---|
| `CHARS` | `0-9A-Z` | Characters to learn (must match `ocr.py`) |
| `S` | `20` | Input image size S×S (must match `ocr.py`) |
| `HIDDEN`, `HIDDEN2` | `256`, `256` | Neurons per hidden layer (`HIDDEN2 = 0` → one layer) |
| `EPOCHS` | `50` | Maximum epochs (early stopping usually ends sooner) |
| `SEED` | `None` | `None` = new random split each run; a number repeats a run exactly |
| `SOURCES` | table | Per source: images per character, augmentations, test fraction |

**Model size vs. memory:** `ocr.bin` is held in RAM on the board. The default
(256/256) is ~700 KB and needs a board with PSRAM. For an ESP32 without PSRAM
(~100 KB free), use `HIDDEN = 64` and `HIDDEN2 = 0` (~110 KB).

---

## 2. Running on the ESP32

### Requirements

- An ESP32 or ESP32-S3 board. PSRAM is needed for the default model size.
- **MicroPython firmware with [ulab](https://github.com/v923z/micropython-ulab)**
  built in. The standard MicroPython firmware does not include ulab. For the
  T-Beam Supreme, use an ESP32-S3 build with PSRAM (SPIRAM) support.
- [Thonny](https://thonny.org/) (or `mpremote`) to copy files and run the
  program.

### Setup with Thonny

1. Connect the board. Under **Tools → Options → Interpreter**, select
   **MicroPython (ESP32)** and the board's COM port.
2. Check the firmware in the Shell: `import ulab` should give no error.
3. Open **View → Files**. In the PC panel, select `ocr.py`, `ocr.bin` and some
   images (e.g. from `test_images/`), then right-click → **Upload to /**.
4. Open `ocr.py` and press **Run ▶**.

The program lists the images on the board and asks which one to read:

```
OLED found on SDA=17, SCL=18
Model loaded: 3 layers, characters: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
Images on the board: char_K.bmp, esp32ok.bmp, hello42.png
Image to read (Enter = quit): hello42.png
hello42.png: 'HELLO42'  [200x47, 1250 ms]
```

Press Enter on an empty line to quit.

### Images

- **Formats:** PNG, BMP, PGM and PPM (colour, grayscale, palette, 16-bit and
  transparent images are all supported). **JPEG cannot be decoded on the board**:
  save it as PNG first (e.g. in Paint).
- The board converts each image to grayscale and shrinks it to at most 200 px
  (`MAX_SIDE` in `ocr.py`) while reading it, so large images need little memory.
  Large PNGs are slow to decode, though; BMP is fastest.
- **For good results:** crop to **one character or one line of text**, use a
  plain background, and make sure the characters don't touch each other.
- Characters the model is less than 50% sure about are shown as `?`.

### OLED display

The result is shown centred on the board's OLED: the file name at the top, the
predicted text in large letters, and the time taken (plus confidence for a
single character) at the bottom. The display is found automatically:

| Board | Display | Pins (SDA, SCL) |
|---|---|---|
| LilyGO T-Beam Supreme (ESP32-S3) | 1.3" SH1106 | 17, 18 (power switched on via the AXP2101 if needed) |
| LilyGO/TTGO LoRa32 V1.3+ / T-Beam V1.x | 0.96" SSD1306 | 21, 22 |
| LilyGO/TTGO LoRa32 V1.0 | 0.96" SSD1306 | 4, 15 (reset 16) |

For other wiring, set `OLED_PINS = (SDA, SCL, reset_pin_or_None, column_offset)`
in `ocr.py` (column offset 2 for SH1106, 0 for SSD1306). Without a display, the
program still works and prints results in the Shell only.

### Using it from your own code

```python
import ocr

m = ocr.load()                              # load ocr.bin
ocr.predict_file(m, "photo.png")            # read, print and display; returns the text

gray, w, h = ocr.load_image("photo.png")    # or any grayscale buffer, e.g. a camera frame
ocr.read_text(m, gray, w, h)                # one line of text -> "HELLO42"
ocr.read_char(m, gray, w, h)                # one character -> ("K", 0.99)

ocr.show("Title", "HI", "info line")        # draw on the OLED
ocr.clear_display()
```

---

## Keeping the PC and the board in sync

The network only works if the board prepares images exactly like the training
did. When changing anything, keep these identical in both files:

- `CHARS` and `S`
- the preprocessing (`otsu`, `text_mask`, `to_input`)

After retraining, upload the new `ocr.bin` to the board.

## Limitations

- Reads one character or one line of text per image; no spaces, multiple
  lines, or touching characters.
- Uppercase letters and digits only. Lookalike pairs (`0`/`O`, `1`/`I`) remain
  the main source of errors.
- Speed: roughly one to a few seconds per image on the board, because the
  image processing runs pixel by pixel in MicroPython.
