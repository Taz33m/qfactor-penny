"""Render the GSAP README preview composition to assets/qfactor-penny-demo-loop.gif.

The renderer launches one headless Chrome instance and drives the composition
through the Chrome DevTools Protocol, so the GSAP HTML remains the source of
truth and rendering stays fast/reproducible without a heavyweight dependency.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPOSITION = ROOT / "video" / "qfactor-penny-readme-gif" / "index.html"
OUTPUT = ROOT / "assets" / "qfactor-penny-demo-loop.gif"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
WIDTH = 1280
HEIGHT = 720
FPS = 12
DURATION = 8.4
START_TIME = 0.5
END_TIME = 8.28
FRAMES = int(FPS * (END_TIME - START_TIME))


def main() -> None:
    if not CHROME.exists():
        raise SystemExit(f"Chrome not found at {CHROME}")
    if not COMPOSITION.exists():
        raise SystemExit(f"Composition not found: {COMPOSITION}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to render the README GIF")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qfactor-readme-gif-") as tmp:
        temp = Path(tmp)
        frame_dir = temp / "frames"
        frame_dir.mkdir()
        with _chrome_session(temp) as client:
            _prepare_page(client)
            _capture_frames(client, frame_dir)
        _encode_gif(ffmpeg, frame_dir, temp / "palette.png")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


def _chrome_session(temp: Path) -> "_ChromeClient":
    return _ChromeClient(temp)


class _ChromeClient:
    def __init__(self, temp: Path) -> None:
        self.port = _free_port()
        self.process = subprocess.Popen(
            [
                str(CHROME),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                f"--window-size={WIDTH},{HEIGHT}",
                "--force-device-scale-factor=1",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={temp / 'chrome-profile'}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.ws: _WebSocket | None = None
        self.next_id = 0

    def __enter__(self) -> "_ChromeClient":
        self._wait_for_browser()
        target = self._new_target()
        self.ws = _WebSocket(target["webSocketDebuggerUrl"])
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            if self.ws:
                self.send("Browser.close")
        except Exception:
            pass
        try:
            if self.ws:
                self.ws.close()
        finally:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def send(self, method: str, params: dict | None = None) -> dict:
        if not self.ws:
            raise RuntimeError("Chrome websocket is not connected")
        self.next_id += 1
        msg_id = self.next_id
        self.ws.send_json({"id": msg_id, "method": method, "params": params or {}})
        while True:
            payload = self.ws.recv_json()
            if payload.get("id") == msg_id:
                if "error" in payload:
                    raise RuntimeError(f"CDP error from {method}: {payload['error']}")
                return payload.get("result", {})

    def _wait_for_browser(self) -> None:
        deadline = time.time() + 12
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/version", timeout=0.5):
                    return
            except Exception:
                if self.process.poll() is not None:
                    raise RuntimeError("Chrome exited before DevTools was available")
                time.sleep(0.1)
        raise TimeoutError("Timed out waiting for Chrome DevTools")

    def _new_target(self) -> dict:
        url = COMPOSITION.resolve().as_uri() + "?t=0.12"
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/json/new?{urllib.parse.quote(url, safe=':/?=&%')}",
            method="PUT",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))


class _WebSocket:
    def __init__(self, ws_url: str) -> None:
        parsed = urllib.parse.urlparse(ws_url)
        if parsed.scheme != "ws":
            raise ValueError(f"Only ws:// URLs are supported: {ws_url}")
        self.sock = socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=8)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        headers = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(headers.encode("ascii"))
        response = self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket handshake failed: {response[:120]!r}")

    def send_json(self, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendall(_masked_frame(data))

    def recv_json(self) -> dict:
        message = self._recv_message()
        return json.loads(message.decode("utf-8"))

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def _recv_message(self) -> bytes:
        parts: list[bytes] = []
        while True:
            header = _read_exact(self.sock, 2)
            first, second = header[0], header[1]
            fin = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", _read_exact(self.sock, 2))[0]
            elif length == 127:
                length = struct.unpack("!Q", _read_exact(self.sock, 8))[0]
            mask = _read_exact(self.sock, 4) if masked else b""
            payload = _read_exact(self.sock, length) if length else b""
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 8:
                raise RuntimeError("WebSocket closed")
            if opcode == 9:
                continue
            if opcode in (0, 1):
                parts.append(payload)
                if fin:
                    return b"".join(parts)


def _prepare_page(client: _ChromeClient) -> None:
    client.send("Page.enable")
    client.send("Runtime.enable")
    client.send(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": WIDTH,
            "height": HEIGHT,
            "deviceScaleFactor": 1,
            "mobile": False,
        },
    )
    client.send("Page.navigate", {"url": COMPOSITION.resolve().as_uri() + "?t=0.12"})
    deadline = time.time() + 8
    while time.time() < deadline:
        ready = client.send(
            "Runtime.evaluate",
            {
                "expression": "Boolean(window.__timelines && window.__timelines['qfactor-readme-gif'])",
                "returnByValue": True,
            },
        )
        if ready.get("result", {}).get("value"):
            return
        time.sleep(0.05)
    raise TimeoutError("Composition timeline was not registered")


def _capture_frames(client: _ChromeClient, frame_dir: Path) -> None:
    for index in range(FRAMES):
        timestamp = START_TIME + index / FPS
        client.send(
            "Runtime.evaluate",
            {
                "expression": (
                    "window.__timelines['qfactor-readme-gif'].time("
                    f"{timestamp:.4f}, false); document.body.offsetHeight;"
                ),
                "returnByValue": True,
            },
        )
        screenshot = client.send(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False, "fromSurface": True},
        )
        (frame_dir / f"frame_{index:04d}.png").write_bytes(base64.b64decode(screenshot["data"]))


def _encode_gif(ffmpeg: str, frame_dir: Path, palette: Path) -> None:
    input_pattern = str(frame_dir / "frame_%04d.png")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            input_pattern,
            "-vf",
            "fps=12,scale=1280:-1:flags=lanczos,palettegen=max_colors=112:stats_mode=diff",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            input_pattern,
            "-i",
            str(palette),
            "-lavfi",
            "fps=12,scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a",
            str(OUTPUT),
        ],
        check=True,
    )


def _masked_frame(data: bytes) -> bytes:
    mask = os.urandom(4)
    length = len(data)
    if length < 126:
        header = bytes([0x81, 0x80 | length])
    elif length < 65536:
        header = bytes([0x81, 0x80 | 126]) + struct.pack("!H", length)
    else:
        header = bytes([0x81, 0x80 | 127]) + struct.pack("!Q", length)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
    return header + mask + masked


def _read_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("Unexpected EOF from websocket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
