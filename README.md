# ESP32 OCR: Character Recognition on a Microcontroller

A small optical character recognition (OCR) system that runs entirely on an
ESP32. A neural network is trained on a PC, exported to a compact binary file,
and run on the board in MicroPython to read characters (**0–9 and A–Z**) from
images. The images are sent from a laptop over **Bluetooth Low Energy**; the
prediction is shown on the board's OLED display and sent back to the laptop.

Developed for EPL428 (IoT) and tested on a **LilyGO T-Beam Supreme** (ESP32-S3,
1.3" SH1106 OLED).

```
PC:     Chars74K images ──► training.py ──► ocr.bin  (trained network)
                                               │  upload to the board (once)
Laptop: image file ──► send_images.py ──► Bluetooth ──┐
ESP32:  ocr.py ──► grayscale ──► black/white ──► split into characters
        ──► neural network ──► "HELLO42" ──► OLED + back to the laptop
```

## Repository contents

| File | Runs on | Purpose |
|---|---|---|
| `training.py` | PC | Trains the network on the Chars74K dataset and writes `ocr.bin` |
| `ocr.py` | ESP32 (MicroPython + ulab) | Loads `ocr.bin`, receives images over Bluetooth, reads them, shows results on the OLED |
| `send_images.py` | Laptop | Sends images to the board over Bluetooth and prints the results |
| `send_images_gui.py` | Laptop | The same in a window: pick images, send them, see the results (no terminal needed) |
| `ocr.bin` | ESP32 | The trained network (weights) |
| `samples.png` | — | Preview of the 20×20 images the network is trained on |
| `test_images/` | Laptop | Example images to send to the board |

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
- A laptop with Bluetooth (4.0 or newer) and Python 3, with
  `pip install bleak pillow` (`pillow` is optional; it converts JPEGs).
- **MicroPython firmware with [ulab](https://github.com/v923z/micropython-ulab)**
  built in. The standard MicroPython firmware does not include ulab. For the
  T-Beam Supreme, use an ESP32-S3 build with PSRAM (SPIRAM) support.
- [Thonny](https://thonny.org/) (or `mpremote`) to copy files and run the
  program.

### Setup with Thonny

1. Connect the board. Under **Tools → Options → Interpreter**, select
   **MicroPython (ESP32)** and the board's COM port.
2. Check the firmware in the Shell: `import ulab` should give no error.
3. Open **View → Files**. In the PC panel, select `ocr.py` and `ocr.bin`, then
   right-click → **Upload to /**. (Images are not uploaded: they come over
   Bluetooth.)
4. Open `ocr.py` and press **Run ▶**. The board advertises over Bluetooth as
   `ESP32-OCR` and the OLED shows "waiting for laptop".

### Sending images from the laptop

Turn Bluetooth on (no pairing in the Windows settings is needed).

**With the window (no terminal):** double-click `send_images_gui.py` (or run
`python send_images_gui.py`).

1. Press **Connect**. The board is found by its Bluetooth name (`ESP32-OCR`);
   enter an address instead to skip the scan.
2. The images in `test_images/` are listed already. **Add images...** and
   **Add folder...** add more; click an image to preview it.
3. Press **Send selected** or **Send all** (or double-click an image). The list
   shows each image's progress, then what the board read, how sure it was and
   how long it took; the selected image's result is shown in large letters next
   to its preview. **Stop** stops after the current image.

**From a terminal**, run on the laptop:

```
python send_images.py test_images/char_K.bmp test_images/photo_A.png
python send_images.py test_images/      # every image in a folder
python send_images.py                   # asks for one image at a time
```

Example output (transfer and reading times depend on the image):

```
Looking for 'ESP32-OCR'...
Connected to ESP32-OCR (packet size 244 bytes)
test_images/char_K.bmp
  sent 29974 bytes in 2.1 s (13.9 KB/s), board is reading it...
  => 'K'  (100% sure)  [90x110, 980 ms on the board]
```

The board receives each image into RAM (nothing is written to its flash),
reads it, shows the result on the OLED and sends it back to the laptop. Stop
the board with Thonny's **Stop** button (or Ctrl+C).

- **JPEGs and very large images** (over 150 KB) are converted to PNG and
  shrunk to at most 400 px on the laptop before sending, if `pillow` is
  installed. The grayscale conversion still happens on the board.
- If the board isn't found, check that `ocr.py` is running and that no other
  program (e.g. a phone app) is connected to it. `--address` connects to a
  known Bluetooth address directly, and `--name` uses another board name
  (`BT_NAME` in `ocr.py`).
- The board uses the standard Nordic UART Service, so BLE UART apps such as
  nRF Connect can also connect to it (e.g. send `PING` and get `PONG`).

**Without Bluetooth:** set `USE_BLUETOOTH = False` in `ocr.py`, upload images
to the board with Thonny, and the program asks which file on the board to
read (press Enter on an empty line to quit).

#### Bluetooth protocol

| Direction | Message |
|---|---|
| laptop → board | `IMG <size> <name>
`; the board answers `OK` or `ERROR <reason>` |
| laptop → board | the image file's bytes, in blocks of 4096 (`BT_BLOCK`) |
| board → laptop | `ACK <bytes so far>` after each block, `DONE <size>` after the last one |
| board → laptop | `RESULT {"text": ..., "conf": ..., "ms": ..., "w": ..., "h": ..., "error": ...}` |

The laptop waits for each `ACK` before sending the next block, so the board's
receive buffer can never overflow.

### Images

- **Formats:** PNG, BMP, PGM and PPM (colour, grayscale, palette, 16-bit and
  transparent images are all supported). **JPEG cannot be decoded on the board**:
  `send_images.py` converts it to PNG automatically (with `pillow`).
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
