#!/usr/bin/env python3
"""llama-swap の経路を真似る中継。

`TsugumiToolLoopCheck --endpoint` は llama-swap を前提に `/upstream/<id>/<path>` を叩く。
素の llama-server にはその経路が無いので、`/upstream/<id>/<path>` を `/<path>` に送り直す。
それ以外の経路 (`/v1/chat/completions` など) はそのまま通す。SSE は届いた分だけ流す。

    python3 upstream_relay.py [listen_port=8081] [server_port=8080]
"""
import http.client
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
SERVER = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
UPSTREAM = re.compile(r"^/upstream/[^/]+(/.*)$")


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, method):
        m = UPSTREAM.match(self.path)
        path = m.group(1) if m else self.path
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        conn = http.client.HTTPConnection("127.0.0.1", SERVER, timeout=7200)
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            self.close_connection = True

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.command, self.path))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", LISTEN), Relay).serve_forever()
