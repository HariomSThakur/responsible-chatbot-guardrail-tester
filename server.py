"""Local-only HTTP server for the Responsible Chatbot Guardrail Tester."""
from __future__ import annotations

import json
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path
import os
import re
from urllib.parse import urlsplit

from analyzer import DEFAULT_LLAMA_MODEL, DEFAULT_OLLAMA_HOST, analyze
from evaluation import run_adversarial, run_benchmark
from policy_store import load_policies, reset_policies, save_policies

ROOT = Path(__file__).resolve().parent

class Handler(BaseHTTPRequestHandler):
    server_version = "GuardrailTester"
    sys_version = ""

    def _local_request_allowed(self) -> bool:
        """Reject DNS-rebinding hosts and cross-origin browser requests."""
        allowed_hosts = {"localhost", "127.0.0.1", "::1"}
        host = self.headers.get("Host", "").strip()
        try:
            host_name = urlsplit(f"//{host}").hostname.lower() if host else None
            if host and host_name not in allowed_hosts:
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed_origin = urlsplit(origin)
                if (not host or parsed_origin.scheme != "http"
                        or parsed_origin.hostname not in allowed_hosts
                        or parsed_origin.netloc.casefold() != host.casefold()):
                    return False
            fetch_site = self.headers.get("Sec-Fetch-Site", "").lower()
            if fetch_site and fetch_site not in {"same-origin", "none"}:
                return False
        except (AttributeError, ValueError):
            return False
        return True

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Permitted-Cross-Domain-Policies", "none")
        if content_type.startswith("text/html"):
            normalized_html = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            script_hashes = [
                base64.b64encode(hashlib.sha256(match).digest()).decode("ascii")
                for match in re.findall(rb"<script>(.*?)</script>", normalized_html, re.DOTALL)
            ]
            script_sources = " ".join(f"'sha256-{value}'" for value in script_hashes)
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
                f"script-src 'self' {script_sources}; script-src-attr 'none'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
                "font-src 'self'; form-action 'self'",
            )
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if not self._local_request_allowed():
            self._send(403, b"Forbidden", "text/plain; charset=utf-8")
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            try:
                sdk_available = find_spec("ollama") is not None
            except ModuleNotFoundError:
                sdk_available = False
            body = json.dumps({"ok": True, "ollama_sdk_available": sdk_available,
                               "ollama_host": os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
                               "model": os.getenv("OLLAMA_MODEL", DEFAULT_LLAMA_MODEL),
                               "policy_storage": "local JSON"}).encode()
            self._send(200, body, "application/json; charset=utf-8")
        elif path == "/api/cases":
            cases = (ROOT / "tests" / "source_test_cases.json").read_bytes()
            self._send(200, cases, "application/json; charset=utf-8")
        elif path == "/api/llm/cases":
            cases = (ROOT / "tests" / "llm_test_cases.json").read_bytes()
            self._send(200, cases, "application/json; charset=utf-8")
        elif path == "/api/policies":
            try:
                policies = load_policies()
                self._send(200, json.dumps(policies, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            except ValueError:
                self._send(500, b'{"error":"Saved policies are invalid. Reset policies to defaults and try again."}', "application/json; charset=utf-8")
        elif path == "/api/adversarial/cases":
            self._send(200, (ROOT / "tests" / "adversarial_cases.json").read_bytes(), "application/json; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_request_allowed():
            self._send(403, b'{"error":"Cross-origin requests are not allowed."}', "application/json; charset=utf-8")
            return
        path = self.path.split("?", 1)[0]
        if path not in {"/api/analyze", "/api/policies", "/api/policies/reset",
                        "/api/adversarial/run", "/api/benchmark/run"}:
            self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return
        try:
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                self._send(415, b'{"error":"Content-Type must be application/json."}', "application/json; charset=utf-8")
                return
            transfer_encoding = self.headers.get("Transfer-Encoding", "").strip().lower()
            if transfer_encoding not in {"", "identity"}:
                raise ValueError("Transfer-encoded request bodies are not supported.")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 60_000:
                raise ValueError("Request body is empty or exceeds the 60,000-byte limit.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if path == "/api/analyze":
                result = analyze(payload)
            elif path == "/api/policies":
                result = save_policies(payload)
            elif path == "/api/policies/reset":
                if payload != {}:
                    raise ValueError("Reset does not accept settings; send an empty JSON object.")
                result = reset_policies()
            elif path == "/api/adversarial/run":
                if not isinstance(payload, dict) or set(payload) - {"test_id"}:
                    raise ValueError("Adversarial run accepts only an optional test_id.")
                test_id = payload.get("test_id")
                if test_id is not None and not isinstance(test_id, str):
                    raise ValueError("test_id must be text.")
                result = run_adversarial(test_id)
            else:
                if not isinstance(payload, dict) or set(payload) - {"review_handling"}:
                    raise ValueError("Benchmark run accepts only review_handling.")
                review_handling = payload.get("review_handling", "exclude")
                result = run_benchmark(review_handling)
            self._send(200, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")
        except Exception as exc:  # Keep internal details/content out of client responses and logs.
            self._send(500, json.dumps({"error": "Analysis failed. Check server configuration and try again."}).encode(), "application/json; charset=utf-8")

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging URLs, query strings, or submitted data on this local prototype.
        return

if __name__ == "__main__":
    host = os.getenv("GUARDRAIL_HOST", "127.0.0.1")
    port = int(os.getenv("GUARDRAIL_PORT", "8000"))
    if host.lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("For safety, the tester only listens on localhost.")
    if not 1 <= port <= 65535:
        raise SystemExit("GUARDRAIL_PORT must be between 1 and 65535.")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Guardrail Tester listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Guardrail Tester.")
        server.server_close()
