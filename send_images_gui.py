# send_images_gui.py - runs on the laptop
#
# A window for sending images to the ESP32 over Bluetooth Low Energy: the same
# thing send_images.py does, without the terminal. The board must be running
# ocr.py (with USE_BLUETOOTH = True).
#
#   pip install bleak pillow          (pillow is optional: JPEGs and previews)
#   python send_images_gui.py         (or double-click it, or: pythonw send_images_gui.py)
#
# 1. Press Connect (the board is found by its Bluetooth name).
# 2. Add images or a folder. Click an image to preview it.
# 3. Press "Send selected" / "Send all" (or double-click an image).
#    The board's answer appears in the list and next to the preview.
#
# The Bluetooth work runs in its own thread with an asyncio loop, so the window
# never freezes; results are passed back to the window through a queue.

import asyncio
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk

try:
    import send_images as si
except SystemExit as e:                       # bleak is missing
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Send images to ESP32", str(e))
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

HERE = os.path.dirname(os.path.abspath(__file__))
FILE_TYPES = [("Images", "*.png *.bmp *.pgm *.ppm *.jpg *.jpeg"
               + (" *.gif *.webp *.tif *.tiff" if Image else "")),
              ("All files", "*.*")]


def describe(e):
    """A readable message for an exception from bleak or the board."""
    msg = str(e) or type(e).__name__
    low = msg.lower()
    if "not ready" in low or "adapter" in low or "radio" in low:
        msg += " - is Bluetooth turned on?"
    return msg


class GuiBoard(si.Board):
    """send_images.Board that also tells the window when the connection drops."""

    def __init__(self, on_lost):
        super().__init__()
        self.on_lost = on_lost

    def _lost(self, client):
        super()._lost(client)
        self.on_lost()


class App:
    def __init__(self, root):
        self.root = root
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.events = queue.Queue()       # (function, args) to run in the window thread
        self.board = None
        self.state = "idle"               # idle, connecting, connected, sending, disconnecting
        self.stop_requested = False
        self.paths = {}                   # list row id -> image path
        self.results = {}                 # list row id -> result dict, or error text
        self.photo = None                 # keeps the preview image alive
        self.scale = root.winfo_fpixels("1i") / 96   # for high-DPI screens
        self.build()
        self.update_controls()
        root.after(50, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        default = os.path.join(HERE, "test_images")
        if os.path.isdir(default):
            self.add_paths(si.image_files([default]))

    # ----- talking between the window and the Bluetooth thread -----

    def call_ui(self, fn, *args):
        """Run fn(*args) in the window thread (safe to call from any thread)."""
        self.events.put((fn, args))

    def poll(self):
        try:
            while True:
                fn, args = self.events.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    # ----- building the window -----

    def build(self):
        root = self.root
        root.title("Send images to ESP32 OCR")
        root.minsize(int(820 * self.scale), int(560 * self.scale))
        base = font.nametofont("TkDefaultFont").actual()["family"]
        self.big_font = font.Font(family=base, size=40, weight="bold")
        self.bold_font = font.Font(family=base, size=10, weight="bold")
        pad = dict(padx=8, pady=4)

        # Board / connection
        top = ttk.LabelFrame(root, text="Board")
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(top, text="Bluetooth name:").pack(side="left", **pad)
        self.name_var = tk.StringVar(value=si.BT_NAME)
        self.name_entry = ttk.Entry(top, textvariable=self.name_var, width=16)
        self.name_entry.pack(side="left", pady=6)
        ttk.Label(top, text="Address (optional):").pack(side="left", **pad)
        self.addr_var = tk.StringVar()
        self.addr_entry = ttk.Entry(top, textvariable=self.addr_var, width=20)
        self.addr_entry.pack(side="left", pady=6)
        self.connect_btn = ttk.Button(top, text="Connect", command=self.toggle_connect, width=12)
        self.connect_btn.pack(side="left", **pad)
        self.dot = tk.Canvas(top, width=14, height=14, highlightthickness=0)
        self.dot.pack(side="left", padx=(8, 2))
        self.dot_id = self.dot.create_oval(2, 2, 12, 12, fill="#999", outline="")
        self.conn_label = ttk.Label(top, text="Not connected")
        self.conn_label.pack(side="left", padx=(2, 8))

        # Bottom first, so it keeps its space when the window is small
        bottom = ttk.Frame(root)
        bottom.pack(side="bottom", fill="x", padx=10, pady=(4, 10))
        status_row = ttk.Frame(bottom)
        status_row.pack(fill="x")
        self.status = ttk.Label(status_row, text="Connect to the board, then send images.")
        self.status.pack(side="left")
        self.progress = ttk.Progressbar(status_row, length=int(260 * self.scale), maximum=100)
        self.progress.pack(side="right")
        log_frame = ttk.Frame(bottom)
        log_frame.pack(fill="x", pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=6, wrap="word", state="disabled",
                                relief="solid", borderwidth=1, font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="x", expand=True)

        # Middle: image list (left) and preview/result (right)
        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True, padx=10, pady=4)

        right = ttk.LabelFrame(mid, text="Preview")
        right.pack(side="right", fill="y", padx=(8, 0))
        self.prev_size = int(240 * self.scale)
        box = tk.Frame(right, width=self.prev_size, height=self.prev_size, bg="#f2f2f2")
        box.pack_propagate(False)          # fixed size, whatever the image is
        box.pack(padx=8, pady=8)
        self.preview = tk.Label(box, text="No image selected", bg="#f2f2f2", fg="#777")
        self.preview.pack(fill="both", expand=True)
        self.result_label = ttk.Label(right, text="", font=self.big_font, anchor="center")
        self.result_label.pack(fill="x", padx=8)
        self.detail_label = ttk.Label(right, text="", anchor="center", justify="center",
                                      wraplength=self.prev_size)
        self.detail_label.pack(fill="x", padx=8, pady=(0, 8))

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)
        cols = ("file", "status", "result", "conf", "time")
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="extended")
        for col, title, width, anchor in (("file", "Image", 200, "w"), ("status", "Status", 110, "w"),
                                          ("result", "Read as", 90, "center"),
                                          ("conf", "Confidence", 80, "center"),
                                          ("time", "Board time", 80, "center")):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=int(width * self.scale), anchor=anchor,
                             stretch=(col == "file"))
        self.tree.tag_configure("ok", foreground="#1a7f37")
        self.tree.tag_configure("error", foreground="#c62828")
        self.tree.tag_configure("active", foreground="#0b5cad")
        tree_scroll = ttk.Scrollbar(list_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: (self.show_selected(), self.update_controls()))
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Delete>", lambda e: self.remove_selected())

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(6, 0))
        self.add_btn = ttk.Button(buttons, text="Add images...", command=self.add_images)
        self.folder_btn = ttk.Button(buttons, text="Add folder...", command=self.add_folder)
        self.remove_btn = ttk.Button(buttons, text="Remove", command=self.remove_selected)
        self.clear_btn = ttk.Button(buttons, text="Clear", command=self.clear)
        for b in (self.add_btn, self.folder_btn, self.remove_btn, self.clear_btn):
            b.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop)
        self.send_all_btn = ttk.Button(buttons, text="Send all", command=self.send_all)
        self.send_sel_btn = ttk.Button(buttons, text="Send selected", command=self.send_selected)
        for b in (self.stop_btn, self.send_all_btn, self.send_sel_btn):
            b.pack(side="right", padx=(6, 0))

    def update_controls(self):
        s = self.state
        busy = s in ("connecting", "sending", "disconnecting")
        has_sel = bool(self.tree.selection())
        self.connect_btn.configure(text="Disconnect" if s in ("connected", "sending") else "Connect",
                                   state="disabled" if busy else "normal")
        entry_state = "normal" if s == "idle" else "disabled"
        self.name_entry.configure(state=entry_state)
        self.addr_entry.configure(state=entry_state)
        can_send = "normal" if s == "connected" else "disabled"
        self.send_sel_btn.configure(state=can_send if has_sel else "disabled")
        self.send_all_btn.configure(state=can_send if self.paths else "disabled")
        self.stop_btn.configure(state="normal" if s == "sending" and not self.stop_requested
                                else "disabled")
        edit = "disabled" if s == "sending" else "normal"
        self.remove_btn.configure(state=edit if has_sel else "disabled")
        self.clear_btn.configure(state=edit if self.paths else "disabled")
        colour = {"idle": "#999", "connecting": "#e0a100", "disconnecting": "#e0a100",
                  "connected": "#2ea043", "sending": "#0b5cad"}[s]
        self.dot.itemconfigure(self.dot_id, fill=colour)

    # ----- log and status -----

    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", time.strftime("%H:%M:%S  ") + msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_status(self, msg):
        self.status.configure(text=msg)

    # ----- connecting -----

    def toggle_connect(self):
        if self.state in ("connected", "sending"):
            self.disconnect()
        elif self.state == "idle":
            self.connect()

    def connect(self):
        name = self.name_var.get().strip() or si.BT_NAME
        address = self.addr_var.get().strip() or None
        self.state = "connecting"
        msg = "Connecting to {}...".format(address) if address else "Looking for '{}'...".format(name)
        self.conn_label.configure(text=msg)
        self.set_status(msg)
        self.log(msg)
        self.update_controls()
        self.run_async(self._connect(name, address))

    async def _connect(self, name, address):
        board = GuiBoard(lambda: self.call_ui(self.on_lost, board))
        try:
            await board.connect(name, address)
        except Exception as e:
            try:
                await board.close()
            except Exception:
                pass
            self.call_ui(self.on_connect_failed, describe(e))
            return
        self.call_ui(self.on_connected, board, address or name)

    def on_connected(self, board, name):
        self.board = board
        self.state = "connected"
        self.conn_label.configure(text="Connected to " + name)
        self.set_status("Connected. Select images and press Send.")
        self.log("Connected to {} (packet size {} bytes)".format(name, board.chunk))
        self.update_controls()

    def on_connect_failed(self, msg):
        self.state = "idle"
        self.conn_label.configure(text="Not connected")
        self.set_status("Could not connect.")
        self.log("Could not connect: " + msg)
        self.update_controls()
        messagebox.showerror("Could not connect", msg, parent=self.root)

    def disconnect(self):
        board, self.board = self.board, None
        self.stop_requested = True
        self.state = "disconnecting"
        self.conn_label.configure(text="Disconnecting...")
        self.update_controls()
        fut = self.run_async(board.close())
        fut.add_done_callback(lambda f: self.call_ui(self.on_disconnected))

    def on_disconnected(self):
        if self.state == "disconnecting":
            self.state = "idle"
        self.conn_label.configure(text="Not connected")
        self.set_status("Disconnected.")
        self.log("Disconnected.")
        self.update_controls()

    def on_lost(self, board):
        if board is not self.board:        # we disconnected on purpose
            return
        self.board = None
        self.stop_requested = True
        if self.state != "sending":        # otherwise on_send_finished cleans up
            self.state = "idle"
        self.conn_label.configure(text="Connection lost")
        self.set_status("Connection lost - press Connect to try again.")
        self.log("Connection to the board was lost.")
        self.update_controls()

    # ----- the image list -----

    def add_paths(self, paths):
        known = set(self.paths.values())
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if p in known:
                continue
            item = self.tree.insert("", "end", values=(os.path.basename(p), "", "", "", ""))
            self.paths[item] = p
            known.add(p)
            added += 1
        self.update_controls()
        return added

    def add_images(self):
        start = os.path.join(HERE, "test_images")
        files = filedialog.askopenfilenames(parent=self.root, title="Choose images to send",
                                            initialdir=start if os.path.isdir(start) else HERE,
                                            filetypes=FILE_TYPES)
        if files:
            self.add_paths(files)

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title="Choose a folder of images",
                                         initialdir=HERE)
        if folder:
            n = self.add_paths(si.image_files([folder]))
            self.log("Added {} image(s) from {}".format(n, folder))

    def remove_selected(self):
        if self.state == "sending":
            return
        for item in self.tree.selection():
            self.tree.delete(item)
            self.paths.pop(item, None)
            self.results.pop(item, None)
        self.show_selected()
        self.update_controls()

    def clear(self):
        if self.state == "sending":
            return
        self.tree.delete(*self.tree.get_children())
        self.paths.clear()
        self.results.clear()
        self.show_selected()
        self.update_controls()

    def set_row(self, item, status=None, result=None, conf=None, ms=None, tag=None):
        if not self.tree.exists(item):
            return
        vals = list(self.tree.item(item, "values"))
        for i, v in ((1, status), (2, result), (3, conf), (4, ms)):
            if v is not None:
                vals[i] = v
        self.tree.item(item, values=vals, tags=(tag,) if tag else ())

    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if item and self.state == "connected":
            self.start_sending([item])

    # ----- preview -----

    def show_selected(self):
        sel = self.tree.selection()
        item = sel[0] if sel else None
        if item is None:
            self.photo = None
            self.preview.configure(image="", text="No image selected")
            self.result_label.configure(text="")
            self.detail_label.configure(text="")
            return
        self.show_image(self.paths[item])
        self.show_result(item)

    def show_image(self, path):
        size = self.prev_size
        try:
            if Image:
                img = Image.open(path)
                img.load()
                if img.mode not in ("1", "L", "RGB", "RGBA"):
                    img = img.convert("RGBA")
                k = min(size / img.width, size / img.height)
                new = (max(1, int(img.width * k)), max(1, int(img.height * k)))
                # small images are blown up with sharp pixels so they stay readable
                img = img.resize(new, Image.NEAREST if k > 1 else Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            else:                          # without pillow Tk can show PNG/GIF/PPM/PGM
                photo = tk.PhotoImage(file=path)
                big = max(photo.width(), photo.height())
                if big > size:
                    photo = photo.subsample(-(-big // size))
                elif big * 2 <= size:
                    photo = photo.zoom(size // big)
        except Exception:
            self.photo = None
            self.preview.configure(image="", text="(no preview for this file)")
            return
        self.photo = photo
        self.preview.configure(image=photo, text="")

    def show_result(self, item):
        res = self.results.get(item)
        if res is None:
            self.result_label.configure(text="", foreground="")
            self.detail_label.configure(text=os.path.basename(self.paths[item]))
        elif isinstance(res, str):
            self.result_label.configure(text="!", foreground="#c62828")
            self.detail_label.configure(text=res)
        else:
            text = res.get("text") or "-"
            self.result_label.configure(text=text, foreground="#1a7f37" if res.get("text") else "#777")
            parts = []
            if not res.get("text"):
                parts.append("nothing found")
            if res.get("conf") is not None:
                parts.append("{}% sure".format(res["conf"]))
            parts.append("{} ms on the board".format(res.get("ms")))
            parts.append("{}x{} px".format(res.get("w"), res.get("h")))
            self.detail_label.configure(text="  ·  ".join(parts))

    # ----- sending -----

    def send_selected(self):
        self.start_sending(list(self.tree.selection()))

    def send_all(self):
        self.start_sending(list(self.tree.get_children()))

    def stop(self):
        self.stop_requested = True
        self.set_status("Stopping after the current image...")
        self.update_controls()

    def start_sending(self, items):
        jobs = [(i, self.paths[i]) for i in items if i in self.paths]
        if not jobs or self.state != "connected":
            return
        self.state = "sending"
        self.stop_requested = False
        for item, _ in jobs:
            self.results.pop(item, None)
            self.set_row(item, status="waiting", result="", conf="", ms="")
        self.update_controls()
        self.run_async(self._send_jobs(self.board, jobs))

    async def _send_jobs(self, board, jobs):
        loop = asyncio.get_running_loop()
        done = 0
        for n, (item, path) in enumerate(jobs):
            if self.stop_requested:
                break
            self.call_ui(self.on_item_start, item, path, n, len(jobs))
            try:
                data, note = await loop.run_in_executor(None, si.prepare, path)
            except Exception as e:
                self.call_ui(self.on_item_done, item, path, None, "can't send it: {}".format(e))
                continue
            if note:
                self.call_ui(self.log, "  " + note)
            try:
                res = await board.send(
                    os.path.basename(path), data,
                    progress=lambda s, t, secs, item=item: self.call_ui(self.on_progress, item, s, t, secs))
            except Exception as e:
                self.call_ui(self.on_item_done, item, path, None, describe(e))
                if board.client is None or not board.client.is_connected:
                    break
                continue
            done += 1
            self.call_ui(self.on_item_done, item, path, res, None)
        self.call_ui(self.on_send_finished, done, len(jobs))

    def on_item_start(self, item, path, n, total):
        self.current = (n, total, os.path.basename(path))
        self.set_row(item, status="preparing...", tag="active")
        self.tree.selection_set(item)        # follow the image being sent
        self.tree.see(item)
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.set_status("Sending {}/{}: {}".format(n + 1, total, os.path.basename(path)))
        self.log("Sending " + path)

    def on_progress(self, item, sent, total, secs):
        pct = sent * 100 // total
        n, count, name = self.current
        if sent < total:
            self.progress.configure(value=pct)
            self.set_row(item, status="sending {}%".format(pct), tag="active")
            self.set_status("Sending {}/{}: {} - {}%".format(n + 1, count, name, pct))
        else:
            self.progress.configure(mode="indeterminate")
            self.progress.start(15)
            self.set_row(item, status="board reading...", tag="active")
            self.set_status("Sending {}/{}: {} - the board is reading it...".format(n + 1, count, name))
            self.log("  sent {} bytes in {:.1f} s ({:.1f} KB/s)".format(
                total, secs, total / 1024 / max(secs, 1e-3)))

    def on_item_done(self, item, path, res, error):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        if error is None and res.get("error"):
            error = "board error: " + str(res["error"])
        if error is not None:
            self.results[item] = error
            self.set_row(item, status="failed", result="!", conf="", ms="", tag="error")
            self.log("  " + error)
        else:
            self.results[item] = res
            text = res.get("text") or "(nothing)"
            conf = "{}%".format(res["conf"]) if res.get("conf") is not None else ""
            self.set_row(item, status="done", result=text, conf=conf,
                         ms="{} ms".format(res.get("ms")), tag="ok")
            self.log("  => '{}'{}  [{}x{}, {} ms on the board]".format(
                text, "  ({} sure)".format(conf) if conf else "", res.get("w"), res.get("h"), res.get("ms")))
        if item in self.tree.selection():
            self.show_result(item)

    def on_send_finished(self, done, total):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        if self.state == "sending":
            self.state = "connected" if self.board is not None else "idle"
        for item in self.tree.get_children():       # rows that never got their turn
            if self.tree.item(item, "values")[1] in ("waiting", "preparing..."):
                self.set_row(item, status="")
        if self.board is not None:
            self.set_status("Finished: {} of {} image(s) read.".format(done, total))
        self.stop_requested = False
        self.update_controls()

    # ----- closing -----

    def on_close(self):
        board, self.board = self.board, None
        if board is not None:
            try:
                self.run_async(board.close()).result(timeout=5)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()


def main():
    if sys.platform == "win32":
        try:                               # sharp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
