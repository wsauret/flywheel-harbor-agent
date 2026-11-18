"""Run the official OpenCode CLI 1.19.4 through ACP without changing its prompts or tools, on DeepSeek-V4.1-Flash-1017
at max effort through failover_proxy.py: OpenCode Go first, OpenRouter pinned to Baseten when Go refuses for its
usage limit or saturation.

Credentials: OPENCODE_API_KEY and OPENROUTER_API_KEY, read by the proxy in this process only; OpenCode itself gets
neither. The provider entry matches the one Harbor's own OpenCode agent was given for OpenCode Go (openai-compatible,
interleaved reasoning_content, max variant). Native evidence goes to /logs/agent/opencode: the `opencode run --format
json` events, stderr, command.json, completion.json, OpenCode's session database, and failover-requests.jsonl (the
serving provider of each model request).
"""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid

# The bridge runs under `python -I`, which leaves its own directory off the import path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import failover_proxy  # noqa: E402

VERSION = "1.19.4"
MODEL = "deepseek-v4.1-flash"
EFFORT = "max"
KEY_VARIABLES = ("OPENCODE_API_KEY", "OPENROUTER_API_KEY")
# The platform packages that `npm install -g opencode-ai@1.19.4` installs, pinned by SHA-256.
PACKAGES = {
    "x86_64": ("https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.19.4.tgz",
               "dca7b06bbb4e4a1ab3681b799cd328057b356d9399e8e47336cf53b7ae658d0d"),
    "aarch64": ("https://registry.npmjs.org/opencode-linux-arm64/-/opencode-linux-arm64-1.19.4.tgz",
                "019c374e3eb3af5d2c2cdf1d2d6e1f9eee0bc9f3289e02570e4310d25df0a991"),
}
CONFIG_OPTIONS = [{"id": "model", "name": "Model", "category": "model", "type": "select",
                   "currentValue": MODEL, "options": [{"value": MODEL, "name": MODEL}]}]


def install(root):
    """Install the pinned OpenCode platform package in a disposable trial directory and return its executable."""
    binary = root / "package/bin/opencode"
    if not binary.exists():
        url, digest = PACKAGES[platform.machine()]
        with urllib.request.urlopen(url) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Downloaded OpenCode package checksum differs")
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(root, filter="data")
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if version != VERSION:
        raise ValueError("Unexpected OpenCode version")
    return binary


class Redactor:
    """Keeps the provider keys out of the trial's logs.

    The agent runs as the same user as this bridge, so a tool call can read the credential file, and its output
    then reaches every record of the conversation. The bridge therefore keeps the agent's home out of the logs,
    passes every line that it writes or sends through `text`, and ends with `scrub`. It matches the whole token
    and each 32-character piece of it, so a wrapped or cut copy is also removed.
    """

    MARK = "[key removed]"

    def __init__(self, tokens):
        pieces = set()
        for token in tokens:
            pieces |= {token[start:start + 32] for start in range(0, len(token) - 31, 16)} | {token, token[-32:]}
        self.pattern = re.compile("|".join(map(re.escape, sorted(pieces, key=len, reverse=True))))
        self.binary = re.compile(self.pattern.pattern.encode())

    def text(self, value):
        return self.pattern.sub(self.MARK, value)

    def copy(self, source, target):
        """Copy a text file into the logs without the token."""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.text(source.read_text(errors="replace")))

    def scrub(self, root):
        """Rewrite each text file under root that holds the token, remove each other file that holds it, and return their count."""
        count = 0
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            data = path.read_bytes()
            if not self.binary.search(data):
                continue
            count += 1
            try:
                path.write_text(self.text(data.decode()))
            except UnicodeDecodeError:
                path.unlink()
        return count


REDACTOR = None


def send(message):
    text = json.dumps(message, ensure_ascii=False)
    print(REDACTOR.text(text) if REDACTOR else text, flush=True)


def update(session_id, payload):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": payload}})


def forward(session_id, event):
    """Mirror one `opencode run --format json` event to the ACP client."""
    part = event.get("part") or {}
    if event.get("type") in ("text", "reasoning") and part.get("text"):
        update(session_id, {"sessionUpdate": "agent_message_chunk" if event["type"] == "text" else "agent_thought_chunk",
                            "content": {"type": "text", "text": part["text"]}})
    elif event.get("type") == "tool_use":
        state = part.get("state") or {}
        update(session_id, {"sessionUpdate": "tool_call", "toolCallId": part.get("callID") or part.get("id"), "title": part.get("tool"),
                            "kind": "other", "status": "failed" if state.get("status") == "error" else "completed",
                            "rawInput": state.get("input"), "rawOutput": state.get("output")})


def run_prompt(params, cwd, session_id):
    """Pass the task text unchanged to stock `opencode run`; retain native evidence."""
    global REDACTOR
    logs = Path(os.environ.get("OPENCODE_WRAPPER_LOGS", "/logs/agent/opencode"))
    logs.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent / ".runtime"
    root.mkdir(parents=True, exist_ok=True)
    binary = Path(os.environ["OPENCODE_WRAPPER_TEST_BINARY"]) if os.environ.get("OPENCODE_WRAPPER_TEST_BINARY") else install(root)
    REDACTOR = Redactor([os.environ[name] for name in KEY_VARIABLES if os.environ.get(name)])
    base_url = failover_proxy.start(log_path=str(logs / "failover-requests.jsonl"))
    home = Path(tempfile.mkdtemp(prefix="home-", dir=root))
    data = home / ".local/share/opencode"
    try:
        data.mkdir(parents=True)
        config = home / ".config/opencode"
        config.mkdir(parents=True)
        (config / "opencode.json").write_text(json.dumps({"provider": {"failover": {
            "npm": "@ai-sdk/openai-compatible", "options": {"apiKey": "unused", "baseURL": base_url},
            "models": {MODEL: {"name": "DeepSeek V4.1 Flash 1017", "limit": {"output": 384000, "context": 1000000},
                               "variants": {EFFORT: {"reasoningEffort": EFFORT}}, "reasoning": True, "tool_call": True,
                               "interleaved": {"field": "reasoning_content"}}}}}}, indent=1))
        env = {name: value for name, value in os.environ.items() if not name.startswith(("OPENAI_", "OPENCODE_", "OPENROUTER_", "XDG_"))}
        # OPENCODE_FAKE_VCS is the setting of Harbor's own OpenCode agent.
        env.update(HOME=str(home), XDG_DATA_HOME=str(home / ".local/share"), XDG_CONFIG_HOME=str(home / ".config"),
                   XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local/state"), OPENCODE_FAKE_VCS="git",
                   # OpenCode takes its working directory from PWD before the process directory.
                   PWD=cwd)
        prompt = "".join(block["text"] for block in params["prompt"] if block["type"] == "text")
        # The command line of Harbor's own OpenCode agent.
        args = [str(binary), f"--model=failover/{MODEL}", "run", "--format=json", "--variant", EFFORT, "--thinking",
                "--dangerously-skip-permissions", "--", prompt]
        (logs / "command.json").write_text(json.dumps({"args": args[1:-1], "version": VERSION}, indent=2))
        failure, steps = None, 0
        with (logs / "events.jsonl").open("w") as output, (home / "stderr.txt").open("w") as errors:
            process = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=errors, text=True, start_new_session=True)
            try:
                for line in process.stdout:
                    output.write(REDACTOR.text(line))
                    output.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "error":
                        failure = json.dumps(event)[:2000]
                    elif event.get("type") == "step_finish":
                        steps += 1
                    forward(session_id, event)
                exit_code = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
        (logs / "completion.json").write_text(REDACTOR.text(json.dumps({"exit_code": exit_code, "quota_stop": False, "steps": steps, "failure": failure}, indent=2)))
        if exit_code or failure or not steps:
            raise RuntimeError(f"OpenCode did not complete: exit {exit_code}: {failure}")
    finally:
        # OpenCode never holds a key here, so its whole session database is evidence; scrub still checks it.
        if (data / "opencode.db").is_file():
            database = sqlite3.connect(data / "opencode.db")
            database.execute("pragma wal_checkpoint(truncate)")
            database.close()
            shutil.copy(data / "opencode.db", logs / "opencode.db")
        if (home / "stderr.txt").is_file():
            REDACTOR.copy(home / "stderr.txt", logs / "stderr.txt")
        shutil.rmtree(home, ignore_errors=True)
        (logs / "key-scrub.json").write_text(json.dumps({"files_that_held_a_key": REDACTOR.scrub(logs.parent)}))


def main():
    cwd = None
    session_id = "opencode-" + uuid.uuid4().hex
    # A stop request must still run the cleanup that keeps the token out of the logs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method, params = message["method"], message.get("params", {})
        try:
            if method == "initialize":
                result = {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": [], "agentInfo": {"name": "opencode", "version": VERSION}}
            elif method == "session/new":
                if params.get("mcpServers"):
                    raise ValueError("This wrapper does not add external MCP servers")
                cwd = params["cwd"]
                result = {"sessionId": session_id, "configOptions": CONFIG_OPTIONS,
                          "models": {"currentModelId": MODEL, "availableModels": [{"modelId": MODEL, "name": MODEL}]}}
            elif method == "session/set_model":
                if params["modelId"] != MODEL:
                    raise ValueError("Only the selected model is admitted")
                result = {}
            elif method == "session/set_config_option":
                if params["configId"] != "model" or params["value"] != MODEL:
                    raise ValueError("Only the selected model is admitted")
                result = {"configOptions": CONFIG_OPTIONS}
            elif method == "session/prompt":
                run_prompt(params, cwd, session_id)
                result = {"stopReason": "end_turn"}
            else:
                raise ValueError("Unsupported ACP method")
            send({"jsonrpc": "2.0", "id": message["id"], "result": result})
        except Exception as error:
            send({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32603, "message": str(error)}})


if __name__ == "__main__":
    main()
