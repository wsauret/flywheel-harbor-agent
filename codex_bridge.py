"""Run the official Codex CLI through ACP without changing its prompts or tools.

The trial's one credential is a ChatGPT access token in CODEX_CHATGPT_ACCESS_TOKEN. The bridge writes it
to a private CODEX_HOME as an externally supplied token (auth_mode chatgptAuthTokens), which Codex never
refreshes, with no refresh token. Native evidence goes to /logs/agent/codex: the `codex exec --json`
events, stderr, the session rollouts (model, effort, and token counts of each model call), and
completion.json.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone

VERSION = "0.169.0"
MODEL = "gpt-6-luna"
EFFORT = "high"
TOKEN_VARIABLE = "CODEX_CHATGPT_ACCESS_TOKEN"
# The platform packages that `npm install -g @openai/codex@0.169.0` installs, pinned by SHA-256.
PACKAGES = {
    "x86_64": ("https://registry.npmjs.org/@openai/codex/-/codex-0.169.0-linux-x64.tgz",
               "89801b4b97faba562eb80fe0a833159ac79399be674022f4b1a0ccab1d153faa", "x86_64-unknown-linux-musl"),
    "aarch64": ("https://registry.npmjs.org/@openai/codex/-/codex-0.169.0-linux-arm64.tgz",
                "c1555c8ad3913e703d736584ff15e98eaeda685bf6eaedc83ae15943badbc1ff", "aarch64-unknown-linux-musl"),
}
# The account has paid credits, so a run past the subscription's limit could spend them. Stop before it.
USED_PERCENT_STOP = 97.0
USED_PERCENT = re.compile(rb'"used_percent":\s*([0-9.]+)')
CONFIG_OPTIONS = [{"id": "model", "name": "Model", "category": "model", "type": "select",
                   "currentValue": MODEL, "options": [{"value": MODEL, "name": MODEL}]}]


def install(root):
    """Install the pinned Codex platform package in a disposable trial directory and return its executable."""
    url, digest, triple = PACKAGES[platform.machine()]
    vendor = root / "vendor"
    binary = vendor / triple / "bin/codex"
    if not binary.exists():
        with urllib.request.urlopen(url) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Downloaded Codex package checksum differs")
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = [member for member in archive if member.name.startswith("package/vendor/")]
            for member in members:
                member.name = member.name[len("package/"):]
            archive.extractall(root, members=members, filter="data")
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if version != f"codex-cli {VERSION}":
        raise ValueError("Unexpected Codex version")
    return binary


def write_auth(home, token):
    """Write the access token as a host-supplied ChatGPT token: no refresh token, and Codex never refreshes it."""
    claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
    document = {"auth_mode": "chatgptAuthTokens", "OPENAI_API_KEY": None,
                "tokens": {"id_token": token, "access_token": token, "refresh_token": "",
                           "account_id": claims["https://api.openai.com/auth"]["chatgpt_account_id"]},
                "last_refresh": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
    descriptor = os.open(home / "auth.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(document, file)


class Redactor:
    """Keeps the access token out of the trial's logs.

    The agent runs as the same user as this bridge, so a tool call can read the credential file, and its output
    then reaches every record of the conversation. The bridge therefore keeps the agent's home out of the logs,
    passes every line that it writes or sends through `text`, and ends with `scrub`. It matches the whole token
    and each 32-character piece of it, so a wrapped or cut copy is also removed.
    """

    MARK = "[access token removed]"

    def __init__(self, token):
        pieces = sorted({token[start:start + 32] for start in range(0, len(token) - 31, 16)} | {token[-32:]})
        self.pattern = re.compile("|".join(map(re.escape, [token, *pieces])))
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


def forward(session_id, event, calls):
    """Mirror one `codex exec --json` event to the ACP client."""
    item = event.get("item")
    if event.get("type") not in ("item.started", "item.completed") or not isinstance(item, dict):
        return
    kind, completed = item.get("type"), event["type"] == "item.completed"
    if kind in ("agent_message", "reasoning"):
        if completed and item.get("text"):
            update(session_id, {"sessionUpdate": "agent_message_chunk" if kind == "agent_message" else "agent_thought_chunk",
                                "content": {"type": "text", "text": item["text"]}})
        return
    if item["id"] not in calls:
        calls.add(item["id"])
        update(session_id, {"sessionUpdate": "tool_call", "toolCallId": item["id"], "title": kind, "kind": "other",
                            "status": "pending", "rawInput": item})
    if completed:
        failed = item.get("status") == "failed" or item.get("exit_code") not in (None, 0)
        update(session_id, {"sessionUpdate": "tool_call_update", "toolCallId": item["id"],
                            "status": "failed" if failed else "completed", "rawOutput": item})


def mirror_rollouts(sessions, logs, offsets):
    """Append the new complete lines of each rollout to its copy in the logs, without the token.

    Returns the highest subscription usage that those lines state, or 0. The rollouts are the evidence of a
    trial that its timeout ends, so the copy follows the run.
    """
    highest = 0.0
    for path in sessions.rglob("rollout-*.jsonl"):
        with path.open("rb") as file:
            file.seek(offsets.get(path, 0))
            data = file.read()
        data = data[:data.rfind(b"\n") + 1]
        if not data:
            continue
        offsets[path] = offsets.get(path, 0) + len(data)
        target = logs / path.relative_to(sessions)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as copy:
            copy.write(REDACTOR.text(data.decode(errors="replace")))
        highest = max([highest, *map(float, USED_PERCENT.findall(data))])
    return highest


def run_prompt(params, cwd, session_id):
    """Pass the task text unchanged to stock `codex exec`; retain native evidence."""
    global REDACTOR
    logs = Path(os.environ.get("CODEX_WRAPPER_LOGS", "/logs/agent/codex"))
    logs.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent / ".runtime"
    root.mkdir(parents=True, exist_ok=True)
    binary = Path(os.environ["CODEX_WRAPPER_TEST_BINARY"]) if os.environ.get("CODEX_WRAPPER_TEST_BINARY") else install(root)
    token = os.environ[TOKEN_VARIABLE]
    REDACTOR = Redactor(token)
    # Codex refuses to install its helper commands when CODEX_HOME is under the temporary directory.
    home = Path(tempfile.mkdtemp(prefix="home-", dir=root))
    offsets = {}
    try:
        write_auth(home, token)
        sessions = home / "sessions"
        env = dict(os.environ, CODEX_HOME=str(home))
        for name in (TOKEN_VARIABLE, "OPENAI_API_KEY", "CODEX_API_KEY"):
            env.pop(name, None)
        prompt = "".join(block["text"] for block in params["prompt"] if block["type"] == "text")
        # The command line of Harbor's own Codex agent.
        args = [str(binary), "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
                "--model", MODEL, "--json", "--enable", "unified_exec", "-c", f"model_reasoning_effort={EFFORT}",
                "--", prompt]
        (logs / "command.json").write_text(json.dumps({"args": args[1:-1], "version": VERSION}, indent=2))
        usage, failure, blocked, used_percent = None, None, False, 0.0
        calls = set()
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
                    if event.get("type") == "turn.completed":
                        usage = event.get("usage")
                    elif event.get("type") in ("turn.failed", "error"):
                        failure = json.dumps(event)[:2000]
                    used_percent = max(used_percent, mirror_rollouts(sessions, logs / "sessions", offsets))
                    if used_percent >= USED_PERCENT_STOP or (failure and "usage limit" in failure.lower()):
                        blocked = True
                        os.killpg(process.pid, signal.SIGTERM)
                        break
                    forward(session_id, event, calls)
                exit_code = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
        (logs / "completion.json").write_text(REDACTOR.text(json.dumps({
            "exit_code": exit_code, "quota_stop": blocked, "used_percent": used_percent, "usage": usage, "failure": failure}, indent=2)))
        if blocked:
            raise RuntimeError("Stopped at the subscription usage limit")
        if exit_code or usage is None:
            raise RuntimeError(f"Codex did not complete: exit {exit_code}: {failure}")
    finally:
        if (home / "sessions").is_dir():
            mirror_rollouts(home / "sessions", logs / "sessions", offsets)
        if (home / "stderr.txt").is_file():
            REDACTOR.copy(home / "stderr.txt", logs / "stderr.txt")
        shutil.rmtree(home, ignore_errors=True)
        (logs / "token-scrub.json").write_text(json.dumps({"files_that_held_the_token": REDACTOR.scrub(logs.parent)}))


def main():
    cwd = None
    session_id = "codex-" + uuid.uuid4().hex
    # A stop request must still run the cleanup that keeps the token out of the logs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method, params = message["method"], message.get("params", {})
        try:
            if method == "initialize":
                result = {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": [], "agentInfo": {"name": "codex", "version": VERSION}}
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
