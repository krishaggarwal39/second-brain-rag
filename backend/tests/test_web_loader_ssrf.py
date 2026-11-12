"""R-23 / R-50: SSRF filter, redirect handling and download limits (no real network)."""
import socket
from unittest.mock import patch

import pytest

from app.core.exceptions import ScrapingError
from app.rag.loaders import web_loader
from app.rag.loaders.web_loader import _last_status_and_headers, resolve_and_check


def _dns(*addrs):
    out = []
    for a in addrs:
        fam = socket.AF_INET6 if ":" in a else socket.AF_INET
        out.append((fam, socket.SOCK_STREAM, 6, "", (a, 443)))
    return out


@pytest.mark.parametrize("addr", [
    "::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "100.64.0.1", "169.254.169.254",
    "fd00::1", "fe80::1", "0.0.0.0", "224.0.0.1", "192.0.2.10", "10.1.2.3",
])
def test_blocks_non_global(addr):
    with patch("socket.getaddrinfo", return_value=_dns(addr)):
        with pytest.raises(ScrapingError, match="restricted IP"):
            resolve_and_check("https://evil.example")


def test_mixed_public_and_private_rejected():
    with patch("socket.getaddrinfo", return_value=_dns("93.184.216.34", "10.0.0.5")):
        with pytest.raises(ScrapingError, match="restricted IP"):
            resolve_and_check("https://rebind.example")


def test_public_ipv6_allowed():
    with patch("socket.getaddrinfo", return_value=_dns("2606:2800:220:1:248:1893:25c8:1946")):
        ip, port, host = resolve_and_check("https://example.com")
    assert ip.startswith("2606:") and port == 443 and host == "example.com"


def test_uses_af_unspec():
    with patch("socket.getaddrinfo", return_value=_dns("93.184.216.34")) as dns:
        resolve_and_check("https://example.com")
    assert dns.call_args.args[2] == socket.AF_UNSPEC


def test_status_parsing_http2_and_continue():
    hdr = "HTTP/1.1 100 Continue\r\n\r\nHTTP/2 301\r\nlocation: /next\r\n\r\n"
    assert _last_status_and_headers(hdr) == (301, {"location": "/next"})
    assert _last_status_and_headers("HTTP/2 200\r\ncontent-type: text/html; charset=utf-8\r\n")[0] == 200


class _FakeStdout:
    def __init__(self, data, chunk=65536):
        self.data, self.pos, self.chunk = data, 0, chunk

    def read(self, n):
        out = self.data[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def close(self):
        pass


class _FakeProc:
    def __init__(self, data, rc=0):
        self.stdout = _FakeStdout(data)
        self.rc, self.killed = rc, False

    def wait(self, timeout=None):
        return self.rc

    def poll(self):
        return self.rc

    def kill(self):
        self.killed = True


def _popen_factory(responses):
    """responses: list of (headers_text, body). Writes headers to the -D file."""
    calls = []

    def factory(cmd, stdout=None, stderr=None):
        headers, body = responses[len(calls)]
        calls.append(cmd)
        path = cmd[cmd.index("-D") + 1]
        with open(path, "w") as f:
            f.write(headers)
        return _FakeProc(body)

    return factory, calls


def test_body_without_content_length_is_capped(monkeypatch):
    monkeypatch.setenv("MAX_WEB_BYTES", "1000")
    factory, _ = _popen_factory([("HTTP/1.1 200 OK\r\ncontent-type: text/html\r\n", b"a" * 5000)])
    with patch("socket.getaddrinfo", return_value=_dns("93.184.216.34")), \
         patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(ScrapingError, match="exceeded max size"):
            web_loader._process_web_sync("https://example.com")


def test_rejects_non_text_content_type():
    factory, _ = _popen_factory([("HTTP/2 200\r\ncontent-type: application/pdf\r\n", b"%PDF")])
    with patch("socket.getaddrinfo", return_value=_dns("93.184.216.34")), \
         patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(ScrapingError, match="Unsupported content type"):
            web_loader._process_web_sync("https://example.com")


def test_redirect_to_private_host_blocked():
    factory, _ = _popen_factory([("HTTP/2 302\r\nlocation: http://internal.example/\r\n", b"")])

    def dns(host, *a, **k):
        return _dns("10.0.0.7") if host == "internal.example" else _dns("93.184.216.34")

    with patch("socket.getaddrinfo", side_effect=dns), patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(ScrapingError, match="restricted IP"):
            web_loader._process_web_sync("https://example.com")


def test_too_many_redirects():
    loop = [("HTTP/2 301\r\nlocation: /again\r\n", b"")] * 10
    factory, calls = _popen_factory(loop)
    with patch("socket.getaddrinfo", return_value=_dns("93.184.216.34")), \
         patch("subprocess.Popen", side_effect=factory):
        with pytest.raises(ScrapingError, match="too many redirects"):
            web_loader._process_web_sync("https://example.com")
    assert len(calls) == web_loader.MAX_REDIRECTS + 1


def test_curl_flags_and_ipv6_pinning():
    html = b"<html><head><title>  Hi  </title></head><body>Hello world</body></html>"
    factory, calls = _popen_factory([("HTTP/2 200\r\ncontent-type: text/html\r\n", html)])
    with patch("socket.getaddrinfo", return_value=_dns("2606:2800:220:1:248:1893:25c8:1946")), \
         patch("subprocess.Popen", side_effect=factory):
        docs = web_loader._process_web_sync("https://example.com")
    cmd = calls[0]
    assert cmd[cmd.index("--proto") + 1] == "=http,https"
    assert cmd[cmd.index("--proto-redir") + 1] == "=http,https"
    assert cmd[cmd.index("--resolve") + 1] == "example.com:443:[2606:2800:220:1:248:1893:25c8:1946]"
    # R-50: nested markup in <title> no longer yields None
    assert docs[0].metadata["title"] == "Hi"
    assert "Hello world" in docs[0].page_content
