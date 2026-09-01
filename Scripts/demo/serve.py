#!/usr/bin/env python3
"""Vision demo: launches the 16K server, warms it, and serves the browser UI.

One command owns the whole session. It starts `TurboFieldfareServer` with the
16K configuration from `docs/SERVER_RUNBOOK.md` §1(a), waits for the port,
warms the decode path and the vision tower so the first picture the user picks
is not also the run that pays for the tower's first call, serves
`index.html` on http://127.0.0.1:8799, and stops the server it started when the
command exits.

    python3 Scripts/demo/serve.py

Why a local proxy rather than the browser talking to 8091 directly: the server
sends no CORS headers (it is a loopback inference server, not a web backend),
and the demo needs two things the Chat Completions API does not carry -- the
prompt token count *before* prefill starts, and the request phase transitions.
Both are already on the server's stderr (`request … prepared prompt=N`,
`… generating`), so this process tails that stream and republishes it on
`/api/events`. The browser correlates by response id, which streaming hands it
in the first SSE chunk, before prefill.

Routed experts come off the mmap path (`docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md`),
which is the server's default: the layer files stay mapped, a Metal residency set
replaces the private slot copies, and the prefill misses are handed to
`F_RDADVISE` up front. Measured on the CLI at this operating point (M3 Pro, 32
slots, 256 tokens) that is ttft x0.77 and tok/s x1.33 against the private-slot
path at 3.2 GB less peak; nobody has measured it through the server yet. `--expert-io pread` starts the server
on the old path instead, which is how the two are compared by feel: run the same
picture twice, once each way, and watch the first-token time.

Only the standard library is used, and no process is ever terminated that this
command did not start (AGENTS.md). If another TurboFieldfare process is already
running, the demo attaches to it read-only and leaves it running at exit.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
import webbrowser
import zlib
from collections import deque
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
# 画像の実体はリポジトリの外にある (`docs/mtp/40-HANDOFF.md` §3、`.gitignore:53-54`)。
# bench のドライバと同じ規約に揃える: `TF_SAMPLE_IMGS`、既定 `~/Pictures/sample_imgs`。
# どちらも無いときだけリポジトリ内の `sample_imgs/` に落ちる。
def _images_dir() -> Path:
    override = os.environ.get("TF_SAMPLE_IMGS")
    if override:
        return Path(override).expanduser()
    home = Path.home() / "Pictures" / "sample_imgs"
    return home if home.is_dir() else REPO_ROOT / "sample_imgs"


IMAGES_DIR = _images_dir()
SERVER_BIN = REPO_ROOT / ".build" / "release" / "TurboFieldfareServer"
DEFAULT_MODEL = "scratch/gemma4-qat-sym.gturbo"

# SERVER_RUNBOOK.md §1(a): the 16K configuration, the fastest one that fits.
MAX_CONTEXT = 8192
# 32 slots is the operating point and the ceiling the front ends accept
# (docs/mtp/40-HANDOFF.md §3). It is the setting where decode is bound by expert
# I/O rather than by the GPU: a verify block reads about 470 MB from the file.
#
# Since the mmap path became the default (docs/mtp/52 §8) a slot copies nothing,
# so peak sits near 1.3 GB here instead of the ~5 GB the older notes quote, and
# slots no longer trade footprint for speed. What they still cost is pages asked
# to stay resident, which is why the ceiling is 32 rather than higher (52 §9).
# `--expert-io pread` puts the old private-slot trade back.
EXPERT_CACHE_SLOTS = 32
DRAFT_BLOCK_SIZE = 4
# Mirrors RuntimeConfiguration.allowedExpertCacheSlots: 32 is the operating
# point and the ceiling (docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md §9).
ALLOWED_EXPERT_CACHE_SLOTS = (8, 16, 24, 32)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# AGENTS.md: one model process at a time, and this is how you find the others.
PGREP_PATTERN = ("TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|"
                 "TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|"
                 "mlx_lm|mlx-lm")

READY_TIMEOUT_S = 240.0
WARMUP_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 25.0


def now() -> float:
    return time.time()


# ---------------------------------------------------------------- event bus

class EventBus:
    """Fan-out of phase changes and server log lines to browser SSE clients.

    Keeps a short history so a page opened (or reloaded) mid-startup still sees
    how the boot went instead of an empty log.
    """

    def __init__(self, history: int = 400) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._history: deque[dict] = deque(maxlen=history)

    def publish(self, event: dict) -> None:
        event.setdefault("t", now())
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self._subscribers.append(q)
            return q, list(self._history)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


# ------------------------------------------------------------- model server

def warmup_png(width: int = 96, height: int = 96) -> bytes:
    """A small PNG built here so warmup needs no image file and no Pillow.

    Its content does not matter -- what matters is that the tower runs once
    (resize, patchify, projector) and that the soft-token path is exercised
    before the user's first real picture.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # PNG filter type: none
        for x in range(width):
            raw += bytes((x * 255 // (width - 1), y * 255 // (height - 1), 128))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


class ModelServer:
    """Owns (or attaches to) the TurboFieldfareServer process."""

    def __init__(self, bus: EventBus, model: str, port: int,
                 expert_cache_slots: int = EXPERT_CACHE_SLOTS,
                 expert_io: str | None = None) -> None:
        self.bus = bus
        self.model = model
        self.port = port
        self.expert_cache_slots = expert_cache_slots
        # None: whatever the binary defaults to (mmap). "mmap"/"pread" pin it.
        # Reported back from the server's own ready line, not from this value,
        # so an attached server is described by what it actually runs.
        self.expert_io_requested = expert_io
        self.expert_io: str | None = None
        self.process: subprocess.Popen | None = None
        self.owned = False
        self.model_id: str | None = None
        self.phase = "starting"
        self.error: str | None = None
        self.warmup: dict = {}
        self.timings: dict = {}

    # -- lifecycle ---------------------------------------------------------

    def set_phase(self, phase: str, **detail) -> None:
        self.phase = phase
        self.bus.publish({"kind": "phase", "phase": phase, "detail": detail})

    def existing_processes(self) -> list[str]:
        result = subprocess.run(["pgrep", "-fl", PGREP_PATTERN],
                                capture_output=True, text=True)
        return [line for line in result.stdout.splitlines() if line.strip()]

    def start(self) -> None:
        try:
            self._start()
        except Exception as exc:  # surfaced in the browser, not just the terminal
            self.error = str(exc)
            self.set_phase("failed", message=str(exc))

    def _start(self) -> None:
        self.set_phase("preflight")
        existing = self.existing_processes()
        if existing:
            # Never stop a process this command did not start. If it answers on
            # the port it is a usable server; otherwise stop here and say why.
            if self._probe_models() is not None:
                self.owned = False
                self.set_phase("attached", processes=existing, port=self.port)
                self._after_ready()
                return
            raise RuntimeError(
                "another TurboFieldfare process is running and port "
                f"{self.port} does not answer: " + "; ".join(existing)
                + " -- stop it yourself, then rerun this command")

        if not SERVER_BIN.exists():
            raise RuntimeError(
                f"{SERVER_BIN.relative_to(REPO_ROOT)} is missing; run "
                "`swift build -c release --product TurboFieldfareServer` first")
        model_path = (REPO_ROOT / self.model) if not os.path.isabs(self.model) else Path(self.model)
        if not model_path.exists():
            raise RuntimeError(f"model not found: {model_path}")

        argv = [str(SERVER_BIN),
                "--model", str(model_path),
                "--port", str(self.port),
                "--ctx-size", str(MAX_CONTEXT),
                "--expert-cache-slots", str(self.expert_cache_slots),
                "--verification", "trusted-install",
                "--draft-block-size", str(DRAFT_BLOCK_SIZE)]
        env = os.environ.copy()
        if self.expert_io_requested is not None:
            env["TF_EXPERT_MMAP"] = "1" if self.expert_io_requested == "mmap" else "0"
        self.set_phase("launching", command=" ".join(argv))
        started = time.monotonic()
        # start_new_session: the child does not receive the terminal's Control-C,
        # so it is stopped exactly once, by stop() below.
        self.process = subprocess.Popen(
            argv, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True, env=env)
        self.owned = True
        threading.Thread(target=self._pump_output, daemon=True).start()

        self.set_phase("loading", note="the port opens after the model loads (20-40 s)")
        self._wait_ready()
        self.timings["load_s"] = round(time.monotonic() - started, 2)
        self._after_ready()

    def _after_ready(self) -> None:
        self.set_phase("ready", model_id=self.model_id, load_s=self.timings.get("load_s"))
        self._warm()
        self.set_phase("warm", model_id=self.model_id, warmup=self.warmup,
                       load_s=self.timings.get("load_s"))

    def _pump_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip("\n")
            if line:
                # The ready line names the routed-expert path the server chose.
                if "expert_io=" in line:
                    self.expert_io = line.split("expert_io=", 1)[1].split()[0]
                self.bus.publish({"kind": "log", "line": line})
        code = self.process.wait()
        self.bus.publish({"kind": "log", "line": f"[demo] server exited with code {code}"})
        if self.phase not in ("stopping", "stopped"):
            self.error = f"the server exited with code {code}"
            self.set_phase("failed", message=self.error)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"the server exited during load (code {self.process.returncode}); "
                    "see the log above")
            model_id = self._probe_models()
            if model_id is not None:
                self.model_id = model_id
                return
            time.sleep(0.5)
        raise RuntimeError(f"the server did not open port {self.port} within "
                           f"{READY_TIMEOUT_S:.0f} s")

    def _probe_models(self) -> str | None:
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=3)
            conn.request("GET", "/v1/models")
            response = conn.getresponse()
            body = response.read()
            conn.close()
            if response.status != 200:
                return None
            data = json.loads(body)
            entries = data.get("data") or []
            if entries:
                self.model_id = entries[0].get("id")
                return self.model_id
        except Exception:
            return None
        return None

    # -- warmup ------------------------------------------------------------

    def _warm(self) -> None:
        """Two calls: text first, then an image.

        The text call pays for the first decode (kernel pipelines, expert cache
        misses, the drafter). The image call pays for the vision tower's first
        run. After both, picking a picture in the UI goes straight into prefill,
        which is the thing the demo is trying to show.
        """
        self.set_phase("warming", step="text")
        text_s = self._warm_request([{"type": "text", "text": "Reply with exactly READY."}])
        self.warmup["text_s"] = text_s

        self.set_phase("warming", step="image")
        import base64
        uri = "data:image/png;base64," + base64.b64encode(warmup_png()).decode("ascii")
        image_s = self._warm_request([
            {"type": "text", "text": "Reply with exactly READY."},
            {"type": "image_url", "image_url": {"url": uri}},
        ])
        self.warmup["image_s"] = image_s

    def _warm_request(self, content: list) -> float | None:
        payload = json.dumps({
            "model": self.model_id or "gemma-4-26b-a4b-it",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 8,
        })
        started = time.monotonic()
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=WARMUP_TIMEOUT_S)
            conn.request("POST", "/v1/chat/completions", body=payload.encode("utf-8"),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            body = response.read()
            conn.close()
            if response.status != 200:
                self.bus.publish({"kind": "log",
                                  "line": f"[demo] warmup returned {response.status}: "
                                          f"{body[:300].decode('utf-8', 'replace')}"})
                return None
        except Exception as exc:
            self.bus.publish({"kind": "log", "line": f"[demo] warmup failed: {exc}"})
            return None
        return round(time.monotonic() - started, 2)

    # -- shutdown ----------------------------------------------------------

    def stop(self) -> None:
        if not self.owned or self.process is None or self.process.poll() is not None:
            return
        self.set_phase("stopping")
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
        self.set_phase("stopped")


# ------------------------------------------------------------- HTTP surface

class DemoHandler(BaseHTTPRequestHandler):
    server_version = "TurboFieldfareVisionDemo/1.0"
    protocol_version = "HTTP/1.1"

    app: "DemoApp" = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        if self.app.verbose:
            sys.stderr.write("[demo] %s - %s\n" % (self.address_string(), fmt % args))

    def handle(self) -> None:
        # A browser that navigates away mid-stream (or a cancelled request)
        # resets the socket, and the keep-alive read that follows raises. It is
        # the normal end of a streaming connection, not something to print a
        # traceback about.
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

    # -- helpers -----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str,
              extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_file(DEMO_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(200, self.app.status())
        elif path == "/api/images":
            self._send_json(200, {"images": self.app.images()})
        elif path.startswith("/api/images/"):
            self._serve_image(unquote(path[len("/api/images/"):]))
        elif path == "/api/events":
            self._serve_events()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._proxy_chat()
        else:
            self._send_json(404, {"error": "not found"})

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json(404, {"error": f"missing {path.name}"})
            return
        self._send(200, body, content_type)

    def _serve_image(self, name: str) -> None:
        # Basename only: the demo serves sample_imgs/, not the filesystem.
        candidate = (IMAGES_DIR / Path(name).name)
        if (candidate.suffix.lower() not in IMAGE_SUFFIXES
                or not candidate.is_file()
                or candidate.parent != IMAGES_DIR):
            self._send_json(404, {"error": "no such image"})
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send(200, candidate.read_bytes(), content_type,
                   extra={"Cache-Control": "public, max-age=300"})

    def _serve_events(self) -> None:
        q, history = self.app.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for event in history:
                self._write_event(event)
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.bus.unsubscribe(q)

    def _write_event(self, event: dict) -> None:
        data = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _proxy_chat(self) -> None:
        """Streaming passthrough to /v1/chat/completions.

        `read1` rather than `read`: the point of the demo is that tokens appear
        as they are produced, and `read(n)` would wait for n bytes to accumulate.
        """
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.app.server.phase in ("failed", "stopped", "stopping"):
            self._send_json(503, {"error": {"message": self.app.server.error
                                            or "the model server is not running"}})
            return
        try:
            conn = HTTPConnection("127.0.0.1", self.app.server.port, timeout=900)
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
        except Exception as exc:
            self._send_json(502, {"error": {"message": f"proxy to the model server failed: {exc}"}})
            return

        content_type = response.getheader("Content-Type", "application/json")
        streaming = "text/event-stream" in content_type
        try:
            if not streaming:
                payload = response.read()
                self._send(response.status, payload, content_type)
                return
            self.send_response(response.status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                piece = response.read1(65536)
                if not piece:
                    break
                self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


class DemoApp:
    def __init__(self, server: ModelServer, bus: EventBus, verbose: bool) -> None:
        self.server = server
        self.bus = bus
        self.verbose = verbose

    def images(self) -> list[dict]:
        """Pictures from `IMAGES_DIR` -- see `_images_dir()` for where that is."""
        if not IMAGES_DIR.is_dir():
            return []
        entries = []
        for path in sorted(IMAGES_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                entries.append({"name": path.name, "bytes": path.stat().st_size})
        return entries

    def status(self) -> dict:
        return {
            "phase": self.server.phase,
            "error": self.server.error,
            "owned": self.server.owned,
            "model_id": self.server.model_id,
            "model_port": self.server.port,
            "warmup": self.server.warmup,
            "load_s": self.server.timings.get("load_s"),
            "config": {
                "max_context": MAX_CONTEXT,
                "expert_cache_slots": self.server.expert_cache_slots,
                "draft_block_size": DRAFT_BLOCK_SIZE,
                "expert_io": self.server.expert_io,
                "images_dir": str(IMAGES_DIR),
                "max_images": 4,
                "image_tokens": 280,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"model to serve (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-port", type=int, default=8091,
                        help="port for TurboFieldfareServer (default: 8091)")
    parser.add_argument("--port", type=int, default=8799,
                        help="port for this demo UI (default: 8799)")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--expert-cache-slots", type=int, default=EXPERT_CACHE_SLOTS,
                        choices=ALLOWED_EXPERT_CACHE_SLOTS,
                        help=f"expert-cache slots (default and ceiling: "
                             f"{EXPERT_CACHE_SLOTS}; fewer slots trade hit rate "
                             "for working set)")
    parser.add_argument("--expert-io", choices=("mmap", "pread"), default=None,
                        help="routed-expert path (default: the server's own, mmap; "
                             "pread is the private-slot path from before docs/mtp/52)")
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = parser.parse_args()

    bus = EventBus()
    model_server = ModelServer(bus, model=args.model, port=args.model_port,
                               expert_cache_slots=args.expert_cache_slots,
                               expert_io=args.expert_io)
    app = DemoApp(model_server, bus, verbose=args.verbose)
    DemoHandler.app = app

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[demo] UI on {url} (the page shows startup progress while the model loads)")

    # The UI comes up first so the browser can watch the model load instead of
    # staring at a connection error for half a minute.
    threading.Thread(target=model_server.start, daemon=True).start()
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    stopping = threading.Event()

    def shutdown(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        print(f"\n[demo] stopping (signal {signum})")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        httpd.server_close()
        model_server.stop()
        if model_server.owned:
            print("[demo] the model server this command started has been stopped")
        elif model_server.process is None and model_server.phase != "failed":
            print("[demo] left the already-running server alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
