"""Offline checks for persistent PAN downloads using a local HTTP server."""

import contextlib
from dataclasses import replace
import hashlib
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import re
import socket
import tempfile
import threading
import unittest

from download_seo_pan import (
    IntegrityError,
    ProtocolError,
    RemoteFile,
    download_file,
    parse_pointer,
)


PAYLOAD = bytes(range(256)) * 16384 + b"the-last-part-of-the-file"
REQUEST_BYTES = 2 * 1024 * 1024
DROP_BYTES = 1024 * 1024 + 137


class RangeServer:
    """Serve deterministic bytes and deliberately violate selected responses."""

    def __init__(self, payload=PAYLOAD):
        self.payload = payload
        self.mode = "normal"
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                range_header = self.headers.get("Range", "")
                owner.requests.append({
                    "range": range_header,
                    "encoding": self.headers.get("Accept-Encoding"),
                    "if_range": self.headers.get("If-Range"),
                })
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                if match is None:
                    self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else len(owner.payload) - 1
                end = min(end, len(owner.payload) - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(owner.payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                mode = owner.mode
                body = owner.payload[start:end + 1]
                declared_start, declared_end, total = start, end, len(owner.payload)
                status = 206
                if mode == "ignore_range":
                    status, body = 200, owner.payload
                elif mode == "wrong_start":
                    declared_start += 1
                elif mode == "wrong_total":
                    total += 17
                elif mode == "wrong_end":
                    declared_end += 1

                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", '"test-content-v1"')
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Connection", "close")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {declared_start}-{declared_end}/{total}")
                if mode == "encoded":
                    self.send_header("Content-Encoding", "gzip")
                self.end_headers()
                self.close_connection = True
                try:
                    if mode == "drop":
                        self.wfile.write(body[:DROP_BYTES])
                        self.wfile.flush()
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                    else:
                        self.wfile.write(body)
                except OSError:
                    # A rejected header can cause the downloader to close early.
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/sample.tar.gz.ab"


class PersistentDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.info = RemoteFile(
            name="sample.tar.gz.ab",
            revision="1234567890abcdef",
            size=len(PAYLOAD),
            sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        )
        self.part = self.output / (self.info.name + ".http.part")
        self.sidecar = self.output / (self.info.name + ".http.json")
        self.final = self.output / self.info.name

    def call_download(self, server, info=None, **kwargs):
        arguments = dict(retries=0, delay=0, timeout=3, request_bytes=REQUEST_BYTES)
        arguments.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return download_file(info or self.info, self.output, server.url, **arguments)

    def make_partial(self, server):
        server.mode = "drop"
        with self.assertRaises((OSError, HTTPException, RuntimeError)):
            self.call_download(server)
        self.assertTrue(self.part.is_file(), "A failed transfer must preserve its prefix")
        self.assertTrue(self.sidecar.is_file(), "A partial file needs an identity sidecar")
        prefix = self.part.read_bytes()
        self.assertGreater(len(prefix), 0)
        self.assertLess(len(prefix), len(PAYLOAD))
        self.assertEqual(prefix, PAYLOAD[:len(prefix)])
        self.assertFalse(self.final.exists())
        return prefix

    def test_pointer_parser_reads_lfs_identity(self):
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{self.info.sha256}\n"
            f"size {self.info.size}\n"
        )
        self.assertEqual(parse_pointer(self.info.name, self.info.revision, pointer), self.info)

    def test_pointer_parser_rejects_non_pointer_response(self):
        with self.assertRaises(ProtocolError):
            parse_pointer(self.info.name, self.info.revision, "<html>proxy error</html>")

    def test_interrupted_prefix_is_retained_and_resumed(self):
        with RangeServer() as server:
            prefix = self.make_partial(server)
            count = len(server.requests)
            server.mode = "normal"
            result = self.call_download(server)
            self.assertEqual(result, self.final)
            self.assertEqual(self.final.read_bytes(), PAYLOAD)
            self.assertFalse(self.part.exists())
            first_resume = server.requests[count]["range"]
            self.assertTrue(first_resume.startswith(f"bytes={len(prefix)}-"))
            self.assertTrue(all(r["encoding"] == "identity" for r in server.requests))

    def test_ignored_range_does_not_overwrite_or_append_prefix(self):
        with RangeServer() as server:
            prefix = self.make_partial(server)
            server.mode = "ignore_range"
            with self.assertRaises(ProtocolError):
                self.call_download(server)
            self.assertEqual(self.part.read_bytes(), prefix)
            self.assertFalse(self.final.exists())

    def test_invalid_range_headers_do_not_append_data(self):
        for mode in ("wrong_start", "wrong_total", "wrong_end", "encoded"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                original_output, original_part, original_sidecar, original_final = (
                    self.output, self.part, self.sidecar, self.final
                )
                self.output = Path(temp)
                self.part = self.output / (self.info.name + ".http.part")
                self.sidecar = self.output / (self.info.name + ".http.json")
                self.final = self.output / self.info.name
                try:
                    with RangeServer() as server:
                        prefix = self.make_partial(server)
                        server.mode = mode
                        with self.assertRaises(ProtocolError):
                            self.call_download(server)
                        self.assertEqual(self.part.read_bytes(), prefix)
                        self.assertFalse(self.final.exists())
                finally:
                    self.output, self.part, self.sidecar, self.final = (
                        original_output, original_part, original_sidecar, original_final
                    )

    def test_hash_mismatch_never_creates_final_file(self):
        with RangeServer() as server:
            wrong = replace(self.info, sha256="0" * 64)
            with self.assertRaises(IntegrityError):
                self.call_download(server, info=wrong)
            self.assertFalse(self.final.exists())

    def test_partial_identity_mismatch_stops_before_network(self):
        with RangeServer() as server:
            prefix = self.make_partial(server)
            count = len(server.requests)
            server.mode = "normal"
            changed = replace(self.info, revision="a-different-commit")
            with self.assertRaises((ProtocolError, IntegrityError)):
                self.call_download(server, info=changed)
            self.assertEqual(self.part.read_bytes(), prefix)
            self.assertEqual(len(server.requests), count)
            self.assertFalse(self.final.exists())


if __name__ == "__main__":
    unittest.main()
