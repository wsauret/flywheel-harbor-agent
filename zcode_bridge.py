"""Run the official ZCode CLI through ACP without changing its prompts or tools."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid

# The bridge runs under `python -I`, which leaves its own directory off the import path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import failover_proxy  # noqa: E402

PACKAGE_URL = "https://cdn-zcode.z.ai/zcode/electron/releases/3.16.3/linux-x64/ZCode-3.16.3-linux-x64.deb"
PACKAGE_SHA256 = "128132b195977fd0a8c8a6c295cf42fe159222d7a5c5efefd8b5f00ef88ab8b6"
CLI_SHA256 = "97490d0f06a7a225d54d47817acf4b4859cdb268b59f2490caaeda982bef063a"
NODE_URL = "https://nodejs.org/dist/v24.14.0/node-v24.14.0-linux-x64.tar.xz"
NODE_SHA256 = "41cd79bb7877c81605a9e68ec4c91547774f46a40c67a17e34d7179ef11729df"
MODEL = "deepseek-v4.1-flash"
CONFIG_OPTIONS = [{"id": "model", "name": "Model", "category": "model", "type": "select",
                   "currentValue": MODEL, "options": [{"value": MODEL, "name": "DeepSeek V4.1 Flash 1017"}]}]


def download(url, digest):
    """Fetch a pinned public artifact and reject changed bytes before extraction."""
    with urllib.request.urlopen(url) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Downloaded artifact checksum differs")
    return data


def install(root):
    """Install original package resources and Node in a disposable trial directory."""
    resources = root / "resources"
    node = root / "node"
    if not resources.exists():
        data = io.BytesIO(download(PACKAGE_URL, PACKAGE_SHA256))
        if data.read(8) != b"!<arch>\n":
            raise ValueError("Expected a Debian ar archive")
        while header := data.read(60):
            name = header[:16].decode().strip().rstrip("/")
            size = int(header[48:58])
            payload = data.read(size)
            data.read(size % 2)
            if name == "data.tar.xz":
                with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                    for member in archive:
                        prefix = "./opt/ZCode/resources/"
                        if not member.name.startswith(prefix) or not member.isfile():
                            continue
                        relative = Path(member.name[len(prefix):])
                        if ".." in relative.parts or relative.is_absolute():
                            raise ValueError("Package resource escapes its directory")
                        target = resources / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.extractfile(member).read())
                        target.chmod(member.mode & 0o777)
                break
    cli = resources / "glm/zcode.cjs"
    if hashlib.sha256(cli.read_bytes()).hexdigest() != CLI_SHA256:
        raise ValueError("ZCode CLI checksum differs")
    if not node.exists():
        data = download(NODE_URL, NODE_SHA256)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            member = archive.getmember("node-v24.14.0-linux-x64/bin/node")
            node.write_bytes(archive.extractfile(member).read())
        node.chmod(0o755)
    version = subprocess.check_output([str(node), str(cli), "--version"], text=True).strip()
    if version != "0.17.0":
        raise ValueError("Unexpected ZCode version")
    return node, cli, resources / "config/provider/zcode-builtin.json"


def provider_config(base_url, key):
    """Select the Go model and max effort through native provider configuration."""
    return {"schemaVersion": 1, "config": {
        "providerConfigRules": {"providerRules": [{"providerId": "go", "config": {
            "group": "standard-personal", "access": {"type": "api-key", "apiKey": key},
            "api": {"type": "openai-chat-completions", "baseUrl": base_url},
            "personalModelIds": [MODEL]}}]},
        "modelConfigRules": {"providerModelRules": [{"providerId": "go", "modelId": MODEL,
            "config": {"properties": {"contextWindow": 1000000}, "optionSpecs": {
                "reasoningLevel": {"values": ["low", "high", "max"], "map": '{"reasoning_effort": reasoningLevel}'},
                "maxOutputTokens": {"max": 384000, "map": '{"max_tokens": maxOutputTokens}'}}}}],
            "manualProviderModelRules": []},
        "defaultModelSelection": {"providerId": "go", "modelId": MODEL, "options": {"reasoningLevel": "max"}}}}


def send(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def update(session_id, payload):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": payload}})


def run_prompt(params, cwd, session_id):
    """Pass the task text unchanged to stock headless ZCode; retain native evidence."""
    logs = Path(os.environ.get("ZCODE_WRAPPER_LOGS", "/logs/agent/zcode"))
    logs.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ZCODE_WRAPPER_TEST_CLI"):
        node = Path(os.environ["ZCODE_WRAPPER_TEST_NODE"])
        cli = Path(os.environ["ZCODE_WRAPPER_TEST_CLI"])
        builtin = Path(os.environ["ZCODE_WRAPPER_TEST_BUILTIN"])
    else:
        root = Path(__file__).resolve().parent / ".runtime"
        root.mkdir(parents=True, exist_ok=True)
        node, cli, builtin = install(root)
    keys = [os.environ[name] for name in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY") if os.environ.get(name)]
    # ZCode talks to the local failover proxy, which holds the real keys: Go first, OpenRouter (Baseten) on Go's limit.
    base_url = failover_proxy.start(log_path=str(logs / "failover-requests.jsonl"))
    config = provider_config(base_url, "unused")
    recorded_config = config
    (logs / "provider-config.json").write_text(json.dumps(recorded_config, indent=2))
    home = logs / "home"
    home.mkdir(exist_ok=True)
    descriptor, config_path = tempfile.mkstemp(prefix="zcode-provider-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(config, file)
        env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), ZCODE_DATA_BASE_DIR=str(home),
                   ZCODE_PERSONAL_PROVIDER_CONFIG_FILE=config_path,
                   ZCODE_BUILTIN_PROVIDER_CONFIG_FILE=str(builtin), NODE_ENV="production")
        for name in ("OPENCODE_API_KEY", "OPENROUTER_API_KEY"):
            env.pop(name, None)
        env["PATH"] = str(node.parent) + os.pathsep + env.get("PATH", "")
        prompt = "".join(block["text"] for block in params["prompt"] if block["type"] == "text")
        args = [str(node), str(cli), "--prompt", prompt, "--cwd", cwd, "--mode", "yolo", "--output-format", "stream-json", "--no-color"]
        result = None
        blocked = False
        calls = set()
        with (logs / "events.jsonl").open("w") as output, (logs / "stderr.txt").open("w") as errors:
            process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=errors, text=True, start_new_session=True)
            try:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    event = json.loads(line)
                    payload = event.get("payload", {})
                    if payload.get("type") in ("model_request_failed", "model_retry_scheduled"):
                        reason = json.dumps(payload).lower()
                        if payload.get("statusCode") == 429 or any(term in reason for term in ("quota", "rate_limit", "rate limit", "insufficient_balance")):
                            blocked = True
                            os.killpg(process.pid, signal.SIGTERM)
                            break
                    if event.get("type") == "model.streaming":
                        kind = payload.get("kind")
                        if kind in ("text_delta", "reasoning_delta") and payload.get("delta"):
                            update(session_id, {"sessionUpdate": "agent_thought_chunk" if kind == "reasoning_delta" else "agent_message_chunk", "content": {"type": "text", "text": payload["delta"]}})
                        elif kind == "tool_call":
                            calls.add(payload["toolCallId"])
                            update(session_id, {"sessionUpdate": "tool_call", "toolCallId": payload["toolCallId"], "title": payload["toolName"], "kind": "other", "status": "pending", "rawInput": payload.get("input", {})})
                    elif event.get("type") == "tool.updated" and payload.get("kind") == "result" and payload["toolCallId"] in calls:
                        tool_result = payload["result"]
                        update(session_id, {"sessionUpdate": "tool_call_update", "toolCallId": payload["toolCallId"], "status": "completed" if tool_result.get("success") else "failed", "rawOutput": tool_result, "content": [{"type": "content", "content": {"type": "text", "text": str(tool_result.get("content", ""))}}]})
                    elif event.get("type") == "result":
                        result = event
                exit_code = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
        (logs / "completion.json").write_text(json.dumps({"exit_code": exit_code, "quota_or_rate_stop": blocked, "result": result}, indent=2))
        for path in logs.rglob("*"):
            if path.is_file() and any(key.encode() in path.read_bytes() for key in keys):
                raise RuntimeError("A native log contains the Harbor secret; do not download this trial")
        if blocked:
            raise RuntimeError("Stopped on a subscription quota or rate error")
        if exit_code or result is None:
            raise RuntimeError("ZCode did not complete; inspect native artifacts")
    finally:
        os.unlink(config_path)


def main():
    cwd = None
    session_id = "zcode-" + uuid.uuid4().hex
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method, params = message["method"], message.get("params", {})
        try:
            if method == "initialize":
                result = {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": [], "agentInfo": {"name": "zcode", "version": "0.17.0"}}
            elif method == "session/new":
                if params.get("mcpServers"):
                    raise ValueError("This wrapper does not add external MCP servers")
                cwd = params["cwd"]
                result = {"sessionId": session_id, "configOptions": CONFIG_OPTIONS,
                          "models": {"currentModelId": MODEL, "availableModels": [{"modelId": MODEL, "name": "DeepSeek V4.1 Flash 1017"}]}}
            elif method == "session/set_model":
                if params["modelId"] != MODEL:
                    raise ValueError("Only the selected Go model is admitted")
                result = {}
            elif method == "session/set_config_option":
                if params["configId"] != "model" or params["value"] != MODEL:
                    raise ValueError("Only the selected Go model is admitted")
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
