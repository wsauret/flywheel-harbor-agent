"""Local OpenAI-compatible proxy that sends each model request to OpenCode Go first and to OpenRouter
(pinned to Baseten) when Go refuses for its usage limit or saturation.

Any harness or script that speaks the OpenAI chat-completions or responses API can point its base URL at
the proxy. The client's own API key and model name are ignored: the proxy reads its credentials from the
environment and names the model on each route itself.

Environment:
  OPENCODE_API_KEY    OpenCode Go key (optional; without it every request goes to OpenRouter)
  OPENROUTER_API_KEY  OpenRouter key (optional; without it a Go refusal is returned to the client)
  FAILOVER_GO_MODEL   default deepseek-v4.1-flash
  FAILOVER_OR_MODEL   default deepseek/deepseek-v4.1-flash-1017
  FAILOVER_LOG        JSONL request log path (default: none)

Use from Python:  base_url = failover_proxy.start()          # "http://127.0.0.1:<port>/v1"
Use from a shell: python3 failover_proxy.py --port 8787 --log requests.jsonl

Failover rule: a Go answer with status 429, 502, 503, or 529 moves that request to OpenRouter before any
byte reaches the client, and later requests skip Go for a cooldown (the reset time that Go states, else
five minutes). A refusal after the response has started streaming is passed through unchanged.
Log record per request: time, path, provider, status, request and response byte counts, duration. No keys,
headers, or content are logged.
"""
import json, os, re, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GO_URL = "https://opencode.ai/inference/go/openai/v1"
OR_URL = "https://openrouter.ai/api/v1"
OR_PROVIDER = {"only": ["baseten"], "ignore": ["baseten/fast"], "allow_fallbacks": False}
SATURATED = {429, 502, 503, 529}
RESET = re.compile(r"resets in (?:(\d+)h\s*)?(?:(\d+)m)?")
PATHS = ("/chat/completions", "/responses")


class Router:
    def __init__(self, log_path=None):
        self.go_key = os.environ.get("OPENCODE_API_KEY")
        self.or_key = os.environ.get("OPENROUTER_API_KEY")
        self.go_model = os.environ.get("FAILOVER_GO_MODEL", "deepseek-v4.1-flash")
        self.or_model = os.environ.get("FAILOVER_OR_MODEL", "deepseek/deepseek-v4.1-flash-1017")
        self.go_blocked_until = 0.0
        self.lock = threading.Lock()
        self.log_path = log_path or os.environ.get("FAILOVER_LOG")
        self.session = "failover-" + os.urandom(6).hex()

    def log(self, **record):
        if not self.log_path:
            return
        line = json.dumps({"t": round(time.time(), 3), **record}) + "\n"
        with self.lock, open(self.log_path, "a") as f:
            f.write(line)

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
        headers["Content-Type"] = "application/json"
        # Cloudflare in front of Go refuses urllib's default user agent (error 1010).
        headers["User-Agent"] = "failover-proxy/1"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=1800), len(data)
        except urllib.error.HTTPError as error:
            return error, len(data)

    def block_go(self, text):
        m = RESET.search(text or "")
        seconds = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + 60) if m and (m.group(1) or m.group(2)) else 300
        with self.lock:
            self.go_blocked_until = time.time() + seconds

    def providers(self):
        order = []
        if self.go_key and time.time() >= self.go_blocked_until:
            order.append("go")
        if self.or_key:
            order.append("openrouter")
        if not order and self.go_key:
            order.append("go")
        return order


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
            order = router.providers()
            for index, provider in enumerate(order):
                started = time.time()
                response, sent = router.request(provider, path, body)
                status = response.status if hasattr(response, "status") else response.code
                if provider == "go" and status in SATURATED and index + 1 < len(order):
                    text = response.read().decode(errors="replace")
                    router.block_go(text)
                    router.log(path=path, provider=provider, status=status, request_bytes=sent, response_bytes=len(text),
                               ms=int((time.time() - started) * 1000), failover=True)
                    continue
                self.relay(response, status, provider, path, sent, started)
                return
            self.send_error(503, "no provider configured")

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
