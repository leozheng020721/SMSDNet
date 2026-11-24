"""Download S-EO PAN parts with persistent, validated HTTP Range transfers."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = "emasquil/shadow-eo"
REVISION = "67c967c9fe58fbe35948ef681f86d0c18728ffba"
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException,
                  TimeoutError, ConnectionError, ssl.SSLError)


class ProtocolError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    name: str
    revision: str
    size: int
    sha256: str


def parse_pointer(name, revision, text):
    digest = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size = re.search(r"^size ([0-9]+)$", text, re.MULTILINE)
    if not text.startswith("version https://git-lfs.github.com/spec/v1\n") or not digest or not size:
        raise ProtocolError("The metadata endpoint did not return a valid Git LFS pointer")
    return RemoteFile(name, revision, int(size.group(1)), digest.group(1))


def _request(url, **headers):
    return urllib.request.Request(url, headers={
        "User-Agent": "S-EO-resumable-downloader/1.0",
        "Accept-Encoding": "identity", "Cache-Control": "no-cache", **headers,
    })


def _retryable(error):
    return not isinstance(error, urllib.error.HTTPError) or error.code in {408, 429, 500, 502, 503, 504}


def _message(error):
    # Do not print expiring signed redirect URLs or proxy credentials.
    if isinstance(error, urllib.error.HTTPError):
        return "HTTP " + str(error.code)
    if isinstance(error, urllib.error.URLError):
        return f"URLError ({type(error.reason).__name__})"
    return type(error).__name__


def read_remote(endpoint, revision, name, retries=12, delay=10, timeout=60):
    url = f"{endpoint}/datasets/{REPO}/raw/{revision}/{name}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as response:
                data = response.read(4097)
            if len(data) > 4096:
                raise ProtocolError("Unexpectedly large metadata response")
            return parse_pointer(name, revision, data.decode("ascii").replace("\r\n", "\n"))
        except NETWORK_ERRORS as error:
            if attempt == retries or not _retryable(error):
                raise
            print(f"Metadata retry {attempt + 1}/{retries}: {_message(error)}", flush=True)
            time.sleep(delay)


def _write_record(path, record):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _file_lock(path):
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another downloader is using this part") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _hash_prefix(path):
    digest = hashlib.sha256()
    if path.exists():
        print(f"Reading saved bytes for SHA256: {path.name} ({path.stat().st_size:,} bytes)", flush=True)
        last = time.monotonic()
        count = 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
                if time.monotonic() - last >= 10:
                    print(f"  SHA256 prefix: {count / 1e9:.2f} GB", flush=True)
                    last = time.monotonic()
    return digest


def _hub_completed(output, info):
    metadata = output / ".cache" / "huggingface" / "download" / (info.name + ".metadata")
    try:
        lines = metadata.read_text(encoding="utf-8").splitlines()
        stamp = float(lines[2])
        return (lines[0] == info.revision and lines[1].strip('"').lower() == info.sha256
                and stamp >= (output / info.name).stat().st_mtime)
    except (OSError, ValueError, IndexError):
        return False


def _receive_range(info, part, url, digest, timeout, request_bytes):
    offset = part.stat().st_size if part.exists() else 0
    end = min(offset + request_bytes, info.size) - 1
    request = _request(url, Range=f"bytes={offset}-{end}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise ProtocolError("Server ignored HTTP Range; saved bytes were preserved. Try another --endpoint")
        start, actual_end, total = map(int, match.groups())
        if start != offset or total != info.size or not start <= actual_end <= end:
            raise ProtocolError("Server returned an unexpected Content-Range; saved bytes were preserved")
        length = actual_end - start + 1
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        if encoding not in {"", "identity"}:
            raise ProtocolError("Compressed HTTP responses cannot be resumed safely")
        if response.headers.get("Content-Length") not in {None, str(length)}:
            raise ProtocolError("Content-Length does not match Content-Range")
        remaining = length
        last = time.monotonic()
        with part.open("ab") as destination:
            while remaining:
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    raise ConnectionError("Response ended before the requested range was complete")
                destination.write(block)
                digest.update(block)
                remaining -= len(block)
                if time.monotonic() - last >= 5:
                    position = actual_end + 1 - remaining
                    print(f"  {info.name}: {position / 1e9:.2f}/{info.size / 1e9:.2f} GB ({position / info.size:.1%})", flush=True)
                    last = time.monotonic()


def download_file(info, output, url, *, retries=12, delay=10, timeout=60,
                  request_bytes=64 * 1024 * 1024, verify_existing=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / info.name
    part = output / (info.name + ".http.part")
    sidecar = output / (info.name + ".http.json")
    identity = asdict(info)
    with _file_lock(output / (info.name + ".http.lock")):
        record = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        if record and any(record.get(key) != value for key, value in identity.items()):
            raise IntegrityError("Saved download identity differs from the requested version; nothing was overwritten")
        if final.exists():
            if final.stat().st_size != info.size:
                raise IntegrityError("Existing final file has the wrong size; nothing was overwritten")
            recorded = record.get("complete") and record.get("mtime_ns") == final.stat().st_mtime_ns
            if not verify_existing and (recorded or _hub_completed(output, info)):
                print(f"SKIP {info.name}: size and previous completion record match (SHA256 not re-read)", flush=True)
                return final
            if _hash_prefix(final).hexdigest() != info.sha256:
                raise IntegrityError("Existing final file failed SHA256; nothing was overwritten")
            _write_record(sidecar, {**identity, "complete": True, "mtime_ns": final.stat().st_mtime_ns})
            print(f"VERIFIED {info.name}", flush=True)
            return final
        if part.exists() and not record:
            raise IntegrityError("Partial file has no identity record; refusing to append")
        if part.exists() and part.stat().st_size > info.size:
            raise IntegrityError("Partial file is larger than the expected file")
        _write_record(sidecar, {**identity, "complete": False})
        digest = _hash_prefix(part)
        failures = 0
        while (part.stat().st_size if part.exists() else 0) < info.size:
            before = part.stat().st_size if part.exists() else 0
            print(f"GET {info.name} from {before / 1e9:.2f} GB", flush=True)
            try:
                _receive_range(info, part, url, digest, timeout, request_bytes)
                failures = 0
            except NETWORK_ERRORS as error:
                if not _retryable(error):
                    raise
                if part.exists() and part.stat().st_size > before:
                    failures = 0
                if failures >= retries:
                    raise
                failures += 1
                saved = part.stat().st_size if part.exists() else 0
                print(f"Network retry {failures}/{retries}: {_message(error)}; keeping {saved:,} bytes", flush=True)
                time.sleep(delay)
        if digest.hexdigest() != info.sha256:
            raise IntegrityError("SHA256 mismatch: partial file was preserved and was NOT promoted to a final file")
        if final.exists():
            raise IntegrityError("A final file appeared during this download; refusing to overwrite it")
        part.rename(final)
        _write_record(sidecar, {**identity, "complete": True, "mtime_ns": final.stat().st_mtime_ns})
        print(f"COMPLETE {info.name}: size and SHA256 verified", flush=True)
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("E:/Dataset/SEO"))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--parts", nargs="+", choices=("aa", "ab", "ac"), default=["aa", "ab", "ac"])
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--request-mib", type=int, default=64)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Read metadata and two 16-byte ranges per part; write no files")
    args = parser.parse_args()
    if args.retries < 0 or args.retry_delay < 0 or args.timeout <= 0 or args.request_mib <= 0:
        parser.error("Retry, timeout and range size values must be valid positive values")
    endpoint = args.endpoint.rstrip("/")
    if urllib.parse.urlsplit(endpoint).scheme != "https":
        parser.error("The endpoint must use HTTPS")
    for suffix in args.parts:
        name = f"panchromatic_crops.part.tar.gz.{suffix}"
        info = read_remote(endpoint, REVISION, name, args.retries, args.retry_delay, args.timeout)
        url = f"{endpoint}/datasets/{REPO}/resolve/{REVISION}/{name}"
        print(f"REMOTE {name}: {info.size:,} bytes; SHA256 {info.sha256}", flush=True)
        if args.probe:
            for start in (0, min(1024 * 1024, info.size - 16)):
                end = start + 15
                for attempt in range(args.retries + 1):
                    try:
                        with urllib.request.urlopen(_request(url, Range=f"bytes={start}-{end}"), timeout=args.timeout) as response:
                            expected = f"bytes {start}-{end}/{info.size}"
                            if (response.status != 206 or response.headers.get("Content-Range") != expected
                                    or response.headers.get("Content-Encoding", "identity") not in {"", "identity"}
                                    or len(response.read(17)) != 16):
                                raise ProtocolError("The download endpoint did not pass the 16-byte Range probe")
                        print(f"  Range {start}-{end}: OK", flush=True)
                        break
                    except NETWORK_ERRORS as error:
                        if attempt == args.retries or not _retryable(error):
                            raise
                        print(f"Probe retry {attempt + 1}/{args.retries}: {_message(error)}", flush=True)
                        time.sleep(args.retry_delay)
            print("PROBE OK: HTTPS and HTTP Range work; no files written", flush=True)
        else:
            download_file(info, args.output, url, retries=args.retries, delay=args.retry_delay,
                          timeout=args.timeout, request_bytes=args.request_mib * 1024 * 1024,
                          verify_existing=args.verify_existing)
    if not args.probe:
        print("All requested parts are present. Merge/extraction has not been started.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Saved .http.part bytes are retained for the next run.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, ValueError, http.client.HTTPException) as error:
        print(f"ERROR: {_message(error) if isinstance(error, NETWORK_ERRORS) else error}", file=sys.stderr)
        print("Any existing .http.part bytes were retained. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(1)
