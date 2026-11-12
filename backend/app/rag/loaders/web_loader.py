import asyncio
import hashlib
import os
import re
import time
import socket
import ipaddress
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Optional
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from app.core.exceptions import ScrapingError

MAX_REDIRECTS = 3
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
_STATUS_RE = re.compile(r"^HTTP/\S+\s+(\d{3})")


def _max_web_bytes() -> int:
    try:
        return int(os.getenv("MAX_WEB_BYTES", str(5 * 1024 * 1024)))
    except ValueError:
        return 5 * 1024 * 1024


def _is_public(ip_obj: ipaddress._BaseAddress) -> bool:
    """True only for globally routable unicast addresses (R-23).

    `is_global` rejects private, loopback, link-local (incl. 169.254.169.254),
    CGNAT 100.64.0.0/10, reserved, unspecified and IPv6 ULA/link-local ranges.
    IPv4-mapped/compatible IPv6 addresses are judged by their IPv4 part.
    """
    if isinstance(ip_obj, ipaddress.IPv6Address):
        mapped = ip_obj.ipv4_mapped or ip_obj.sixtofour
        if mapped is not None:
            ip_obj = mapped
    return ip_obj.is_global and not ip_obj.is_multicast


def resolve_and_check(url: str) -> tuple[str, int, str]:
    """SSRF protection: resolve the hostname and block private/local IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ScrapingError("Security Policy Violation: URL scheme is not permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise ScrapingError("Security Policy Violation: No hostname provided.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise ScrapingError("Security Policy Violation: invalid port.")

    try:
        # IPv4 and IPv6: every address the name resolves to is checked, so a
        # host returning one public and one private address is rejected.
        info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        addresses = []
        for entry in info:
            addr = entry[4][0].split("%", 1)[0]  # drop IPv6 zone id
            if addr not in addresses:
                addresses.append(addr)
    except Exception:
        raise ScrapingError("DNS resolution failed. The host could not be found.")
    if not addresses:
        raise ScrapingError("DNS resolution failed. The host could not be found.")

    for addr in addresses:
        try:
            ip_obj = ipaddress.ip_address(addr)
        except ValueError:
            raise ScrapingError(f"Security Policy Violation: {addr} is a restricted IP.")
        if not _is_public(ip_obj):
            raise ScrapingError(f"Security Policy Violation: {addr} is a restricted IP.")

    return addresses[0], port, hostname


def _last_status_and_headers(header_data: str) -> tuple[Optional[int], dict]:
    """Parse the final header block written by `curl -D` (handles HTTP/1.x and HTTP/2)."""
    status, headers = None, {}
    for raw in header_data.splitlines():
        line = raw.strip()
        m = _STATUS_RE.match(line)
        if m:
            status, headers = int(m.group(1)), {}  # new block (e.g. after 100 Continue)
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return status, headers


def _curl_fetch(url: str, hostname: str, port: int, ip: str, max_bytes: int) -> tuple[bytes, str]:
    """Run curl pinned to the validated IP, streaming stdout with a hard byte cap."""
    import subprocess
    import tempfile

    pinned = f"[{ip}]" if ":" in ip else ip
    with tempfile.NamedTemporaryFile() as header_file:
        cmd = [
            "curl", "-s",
            "--proto", "=http,https",
            "--proto-redir", "=http,https",
            "--max-time", "10",
            "--max-filesize", str(max_bytes),
            "--resolve", f"{hostname}:{port}:{pinned}",
            "-D", header_file.name,
            url,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        chunks, total = [], 0
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    proc.kill()
                    raise ScrapingError(f"Response exceeded max size of {max_bytes // (1024 * 1024)}MB")
                chunks.append(chunk)
            returncode = proc.wait(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
        if returncode == 63:  # curl: maximum file size exceeded
            raise ScrapingError(f"Response exceeded max size of {max_bytes // (1024 * 1024)}MB")
        if returncode != 0:
            raise RuntimeError(f"curl exit {returncode}")
        header_file.seek(0)
        header_data = header_file.read().decode("utf-8", errors="ignore")
    return b"".join(chunks), header_data


def _process_web_sync(url: str) -> List[Document]:
    import urllib.parse

    current_url = url
    content = b""
    max_bytes = _max_web_bytes()

    for hop in range(MAX_REDIRECTS + 1):
        # Every hop is re-resolved and re-validated (redirects to internal hosts are blocked).
        ip, port, hostname = resolve_and_check(current_url)

        body, header_data = b"", ""
        for attempt, delay in enumerate([1, 2, 4, 0]):
            try:
                body, header_data = _curl_fetch(current_url, hostname, port, ip, max_bytes)
                break
            except ScrapingError:
                raise
            except Exception:
                if delay == 0:
                    raise ScrapingError(f"Failed to scrape {current_url} after retries.")
                time.sleep(delay)

        status, headers = _last_status_and_headers(header_data)
        if status in (301, 302, 303, 307, 308) and headers.get("location"):
            current_url = urllib.parse.urljoin(current_url, headers["location"])
            continue

        ctype = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if ctype and ctype not in ALLOWED_CONTENT_TYPES:
            raise ScrapingError(f"Unsupported content type: {ctype}")
        content = body
        break
    else:
        raise ScrapingError(f"Failed to scrape {url}: too many redirects")

    if not content:
        raise ScrapingError(f"Failed to scrape {url}: no content returned")

    soup = BeautifulSoup(content, "html.parser")

    # Remove unwanted tags
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # R-50: <title> may be empty or contain nested markup (string is None then).
    title = (soup.title.get_text(strip=True) if soup.title else "") or "Unknown"
    page_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    metadata = {
        "url": url,
        "title": title,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "web",
        "hash": page_hash,
    }

    return [Document(page_content=text, metadata=metadata)]


async def load_web(url: str) -> List[Document]:
    """Asynchronously load a web page and extract text with exponential backoff."""
    return await asyncio.to_thread(_process_web_sync, url)
