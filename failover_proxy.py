"""Local OpenAI-compatible proxy that sends each model request to the OpenCode Go subscription first and to
OpenRouter (pinned to Baseten) while Go cannot serve, and returns to Go as soon as it can again.

Any harness or script that speaks the OpenAI chat-completions or responses API can point its base URL at the
proxy. The client's own API key and model name are ignored: the proxy reads its credentials from the
environment and names the model on each route itself.

Environment:
  OPENCODE_API_KEY    OpenCode Go key (optional; without it every request goes to OpenRouter)
  OPENROUTER_API_KEY  OpenRouter key (optional; without it a Go refusal is returned to the client)
  FAILOVER_GO_MODEL   default deepseek-v4.1-flash
  FAILOVER_OR_MODEL   default deepseek/deepseek-v4.1-flash-1017
  FAILOVER_LOG        JSONL request log path (default: none)

Use from Python:  base_url = failover_proxy.start()          # "http://127.0.0.1:<port>/v1"
Use from a shell: python3 failover_proxy.py --port 8787 --log requests.jsonl

How Go's availability is tracked (the rules Flywheel's own provider failover follows):
- Spent usage limit (429 with GoUsageLimitError): Go is held back until the reset that Go states. The proxy
  reads the reset from Go's usage report (/zen/go/v1/usage, the latest resetsAt of a rate-limited window),
  else from the error text ("resets in 1h 2m"), else from a retry-after header, else for five minutes.
- Saturation or outage (another 429, 5xx, or no connection): a short hold that starts at 60 seconds and
  doubles on each repeat up to five minutes, forgotten after 15 quiet minutes.
- When a hold ends, exactly one request probes Go (a lease); concurrent requests stay on OpenRouter until the
  probe answers. A served probe makes Go live again; a refused one sets a new hold.
A refusal is moved to OpenRouter only before any byte reaches the client; a refusal after the response
started streaming is passed through unchanged.

Log record per request: time, path, provider, status, request and response byte counts, duration, and the
Go hold that a refusal set. No keys, headers, or content are logged.
"""
import datetime, json, os, re, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GO_URL = "https://opencode.ai/inference/go/openai/v1"
GO_USAGE_URL = "https://opencode.ai/zen/go/v1/usage"
OR_URL = "https://openrouter.ai/api/v1"
OR_PROVIDER = {"only": ["baseten"], "ignore": ["baseten/fast"], "allow_fallbacks": False}
REFUSED = {429, 500, 502, 503, 504, 529}
RESET = re.compile(r"resets in (?:(\d+)h)?\s*(?:(\d+)m)?")
PATHS = ("/chat/completions", "/responses")
SAFE_RECHECK = 300.0
OUTAGE_HOLD, OUTAGE_MEMORY = 60.0, 900.0
PROBE_LEASE = 600.0
# Cloudflare in front of Go refuses urllib's default user agent (error 1010).
USER_AGENT = "failover-proxy/1"


class Router:
    def __init__(self, log_path=None):
        self.go_key = os.environ.get("OPENCODE_API_KEY")
        self.or_key = os.environ.get("OPENROUTER_API_KEY")
        self.go_model = os.environ.get("FAILOVER_GO_MODEL", "deepseek-v4.1-flash")
        self.or_model = os.environ.get("FAILOVER_OR_MODEL", "deepseek/deepseek-v4.1-flash-1017")
        self.log_path = log_path or os.environ.get("FAILOVER_LOG")
        self.session = "failover-" + os.urandom(6).hex()
        self.lock = threading.Lock()
        self.held_until = 0.0        # Go is not tried before this moment
        self.held = False            # a hold was set and no request has served from Go since
        self.probe_until = 0.0       # a probe request holds the lease until this moment
        self.outage_hold = 0.0       # the current outage hold length, doubling on repeats
        self.outage_ended = 0.0

    def log(self, **record):
        if self.log_path:
            line = json.dumps({"t": round(time.time(), 3), **record}) + "\n"
            with self.lock, open(self.log_path, "a") as f:
                f.write(line)

    def choose(self):
        """The provider order for one request, and whether this request holds the Go probe lease."""
        now = time.time()
        with self.lock:
            go = bool(self.go_key) and now >= self.held_until
            probe = False
            if go and self.held:
                if now < self.probe_until and self.or_key:
                    go = False
                else:
                    self.probe_until, probe = now + PROBE_LEASE, True
        order = (["go"] if go else []) + (["openrouter"] if self.or_key else [])
        return (order or (["go"] if self.go_key else [])), probe

    def served_by_go(self):
        with self.lock:
            self.held, self.probe_until, self.held_until = False, 0.0, 0.0
            if self.outage_hold:
                self.outage_ended = time.time()

    def release_probe(self):
        with self.lock:
            self.probe_until = 0.0

    def go_reset(self):
        """The latest reset of a rate-limited window in Go's usage report, as epoch seconds, or None."""
        try:
            req = urllib.request.Request(GO_USAGE_URL, headers={"Authorization": f"Bearer {self.go_key}", "Accept": "application/json", "User-Agent": USER_AGENT})
            usage = json.load(urllib.request.urlopen(req, timeout=20))["usage"]
            resets = [datetime.datetime.fromisoformat(w["resetsAt"].replace("Z", "+00:00")).timestamp()
                      for w in usage.values() if isinstance(w, dict) and w.get("status") == "rate-limited"]
            return max(resets) if resets else None
        except Exception:
            return None

    def hold_go(self, status, text, retry_after):
        """Hold Go back after a refusal, by Flywheel's rules; returns (kind, seconds)."""
        now = time.time()
        if status == 429 and "UsageLimit" in text:
            until = self.go_reset()
            kind = "usage_limit"
            if until is None:
                m = RESET.search(text)
                if m and (m.group(1) or m.group(2)):
                    until = now + int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + 60
            if until is None and retry_after:
                until = now + retry_after
            until = until if until and until > now else now + SAFE_RECHECK
        else:
            kind = "outage"
            with self.lock:
                if not self.outage_hold or (not self.held and now - self.outage_ended > OUTAGE_MEMORY):
                    self.outage_hold = OUTAGE_HOLD
                elif self.held:
                    self.outage_hold = min(self.outage_hold * 2, SAFE_RECHECK)
                hold = max(self.outage_hold, retry_after or 0)
            until = now + hold
        with self.lock:
            self.held, self.probe_until, self.held_until = True, 0.0, until
        return kind, round(until - now)

    def request(self, provider, path, body):
        payload = dict(body)
        if provider == "go":
            payload["model"] = self.go_model
            url, headers = GO_URL + path, {"Authorization": f"Bearer {self.go_key}", "x-opencode-session": self.session}
        else:
            payload["model"] = self.or_model
            payload["provider"] = OR_PROVIDER
            url, headers = OR_URL + path, {"Authorization": f"Bearer {self.or_key}"}
        data = json.dumps(payload).encode()
        headers.update({"Content-Type": "application/json", "User-Agent": USER_AGENT})
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method="POST"), timeout=1800), len(data)
        except urllib.error.HTTPError as error:
            return error, len(data)
        except (urllib.error.URLError, OSError):
            return None, len(data)


def retry_after_seconds(response):
    try:
        value = response.headers.get("retry-after-ms")
        if value:
            return float(value) / 1000
        value = response.headers.get("retry-after")
        return float(value) if value else None
    except (TypeError, ValueError, AttributeError):
        return None


def handler_for(router):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            path = self.path.split("?")[0]
            path = path[3:] if path.startswith("/v1/") else path
            if path not in PATHS:
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            order, probe = router.choose()
            try:
                for index, provider in enumerate(order):
                    started = time.time()
                    response, sent = router.request(provider, path, body)
                    status = 0 if response is None else (response.status if hasattr(response, "status") else response.code)
                    if provider == "go" and (response is None or status in REFUSED) and index + 1 < len(order):
                        text = "" if response is None else response.read().decode(errors="replace")
                        kind, hold = router.hold_go(status, text, None if response is None else retry_after_seconds(response))
                        probe = False
                        router.log(path=path, provider=provider, status=status, request_bytes=sent, response_bytes=len(text),
                                   ms=int((time.time() - started) * 1000), failover=True, hold=kind, hold_seconds=hold)
                        continue
                    if response is None:
                        self.send_error(502, "provider unreachable")
                        router.log(path=path, provider=provider, status=0, request_bytes=sent, response_bytes=0, ms=int((time.time() - started) * 1000))
                        return
                    if provider == "go" and status < 400:
                        router.served_by_go()
                        probe = False
                    self.relay(response, status, provider, path, sent, started)
                    return
                self.send_error(503, "no provider configured")
            finally:
                if probe:
                    router.release_probe()

        def relay(self, response, status, provider, path, sent, started):
            self.send_response(status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("x-failover-provider", provider)
            self.end_headers()
            total = 0
            try:
                while True:
                    chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            finally:
                router.log(path=path, provider=provider, status=status, request_bytes=sent, response_bytes=total,
                           ms=int((time.time() - started) * 1000))

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A client that drops an idle keep-alive connection is normal and not worth a traceback.
        pass


def start(port=0, log_path=None):
    """Start the proxy on 127.0.0.1 in a daemon thread and return its OpenAI base URL."""
    server = Server(("127.0.0.1", port), handler_for(Router(log_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--log")
    options = parser.parse_args()
    print(start(options.port, options.log), flush=True)
    threading.Event().wait()
