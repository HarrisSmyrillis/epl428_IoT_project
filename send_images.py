# send_images.py - runs on the laptop
#
# Sends images to the ESP32 over Bluetooth Low Energy and prints what the
# board read. The board must be running ocr.py (with USE_BLUETOOTH = True).
#
#   pip install bleak pillow          (pillow is optional, see below)
#   python send_images.py photo.png test_images/char_K.bmp
#   python send_images.py test_images/           # every image in a folder
#   python send_images.py                        # asks for images one by one
#
# The board can't decode JPEG, so JPEGs (and any other format Pillow can open)
# are converted to PNG here before sending. Very large images are also made
# smaller first, so they fit in the board's memory and transfer quickly.
# No pairing in the Windows settings is needed: just turn Bluetooth on.

import argparse
import asyncio
import io
import json
import os
import sys
import time

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    sys.exit("bleak is missing - install it with:  pip install bleak")

# Must match ocr.py
BT_NAME = "ESP32-OCR"
BT_BLOCK = 4096
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"      # we write here
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"      # board notifies here

BOARD_FORMATS = ("png", "bmp", "pgm", "ppm")         # what ocr.py can decode
MAX_BYTES = 150_000          # bigger files are shrunk before sending
SHRINK_SIDE = 400            # longest side after shrinking (the board goes down to 200 itself)
TIMEOUT = 20                 # seconds to wait for an answer from the board


def prepare(path):
    """Read an image file and return (bytes to send, note about any conversion)."""
    with open(path, "rb") as f:
        data = f.read()
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in BOARD_FORMATS and len(data) <= MAX_BYTES:
        return data, ""
    try:
        from PIL import Image
    except ImportError:
        if ext not in BOARD_FORMATS:
            raise ValueError("the board can't read .{} - install pillow (pip install pillow) "
                             "to convert it automatically, or save it as PNG".format(ext))
        return data, "(large file - install pillow to shrink it before sending)"
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode not in ("L", "RGB", "RGBA", "P"):
        img = img.convert("RGB")
    note = "converted to PNG" if ext not in BOARD_FORMATS else ""
    if max(img.size) > SHRINK_SIDE:
        old = img.size
        img.thumbnail((SHRINK_SIDE, SHRINK_SIDE))
        note = (note + ", " if note else "") + "shrunk {}x{} -> {}x{}".format(*old, *img.size)
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue(), "(" + note + ")" if note else ""


def image_files(args):
    for a in args:
        if os.path.isdir(a):
            for name in sorted(os.listdir(a)):
                p = os.path.join(a, name)
                if os.path.isfile(p) and name.rsplit(".", 1)[-1].lower() in BOARD_FORMATS + ("jpg", "jpeg"):
                    yield p
        else:
            yield a


class Board:
    def __init__(self):
        self.client = None
        self.lines = asyncio.Queue()
        self.partial = b""

    async def connect(self, name, address=None):
        if address:
            target = address
        else:
            print("Looking for '{}'...".format(name))
            target = await BleakScanner.find_device_by_name(name, timeout=15)
            if target is None:
                raise RuntimeError("'{}' not found - is ocr.py running on the board, "
                                   "and Bluetooth on here?".format(name))
        self.client = BleakClient(target, disconnected_callback=self._lost)
        await self.client.connect()
        await self.client.start_notify(NUS_TX, self._notify)
        self.chunk = max(20, min(512, self.client.mtu_size - 3))
        print("Connected to {} (packet size {} bytes)".format(name, self.chunk))

    def _lost(self, _client):
        self.lines.put_nowait("ERROR connection lost")

    def _notify(self, _char, data):
        self.partial += bytes(data)
        while b"\n" in self.partial:
            line, self.partial = self.partial.split(b"\n", 1)
            self.lines.put_nowait(line.decode(errors="replace"))

    async def _expect(self, *words, timeout=TIMEOUT):
        """Wait for a message that starts with one of the words (or ERROR)."""
        while True:
            try:
                line = await asyncio.wait_for(self.lines.get(), timeout)
            except asyncio.TimeoutError:
                raise RuntimeError("no answer from the board (waited {} s)".format(timeout))
            word, _, rest = line.partition(" ")
            if word == "ERROR":
                raise RuntimeError(rest)
            if word in words:
                return word, rest

    async def _write(self, data):
        for k in range(0, len(data), self.chunk):
            await self.client.write_gatt_char(NUS_RX, data[k:k + self.chunk], response=False)

    async def send(self, name, data):
        """Send one image, return the board's result dict."""
        while not self.lines.empty():           # forget leftovers from an earlier image
            self.lines.get_nowait()
        name = name.replace("\n", " ").replace("\r", " ")
        await self._write("IMG {} {}\n".format(len(data), name).encode())
        await self._expect("OK")
        t0 = time.time()
        sent = 0
        while sent < len(data):
            block = data[sent:sent + BT_BLOCK]
            await self._write(block)
            sent += len(block)
            await self._expect("DONE" if sent == len(data) else "ACK")
            print("\r  sending {}%".format(sent * 100 // len(data)), end="", flush=True)
        secs = time.time() - t0
        print("\r  sent {} bytes in {:.1f} s ({:.1f} KB/s), board is reading it...".format(
            len(data), secs, len(data) / 1024 / max(secs, 1e-3)))
        _, rest = await self._expect("RESULT", timeout=120)
        return json.loads(rest)

    async def close(self):
        if self.client is not None and self.client.is_connected:
            await self.client.disconnect()


async def send_one(board, path):
    print(path)
    try:
        data, note = prepare(path)
    except (OSError, ValueError) as e:
        print("  can't send it:", e)
        return
    if note:
        print("  " + note)
    res = await board.send(os.path.basename(path), data)
    if res.get("error"):
        print("  board error:", res["error"])
        return
    text = res["text"] or "(nothing found)"
    extra = "  ({}% sure)".format(res["conf"]) if res.get("conf") is not None else ""
    print("  => '{}'{}  [{}x{}, {} ms on the board]".format(text, extra, res["w"], res["h"], res["ms"]))


async def main():
    ap = argparse.ArgumentParser(description="Send images to the ESP32 OCR over Bluetooth")
    ap.add_argument("images", nargs="*", help="image files or folders (none = ask)")
    ap.add_argument("--name", default=BT_NAME, help="Bluetooth name of the board")
    ap.add_argument("--address", help="Bluetooth address of the board (skips the scan)")
    args = ap.parse_args()

    board = Board()
    await board.connect(args.name, args.address)
    try:
        if args.images:
            for path in image_files(args.images):
                await send_one(board, path)
        else:
            loop = asyncio.get_running_loop()
            while True:
                path = (await loop.run_in_executor(None, input, "\nImage to send (Enter = quit): ")).strip().strip('"')
                if not path:
                    break
                for p in image_files([path]):
                    await send_one(board, p)
    except RuntimeError as e:
        print("\nError:", e)
    finally:
        await board.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        sys.exit(str(e))
