"""Run the official ZCode CLI through ACP without changing its prompts or tools.

This wrapper serves gpt-6-luna at effort high through the ChatGPT subscription backend. The trial's one
credential is a ChatGPT access token in CODEX_CHATGPT_ACCESS_TOKEN: ZCode sends it as the provider's API key,
holds no refresh token, and never refreshes it.
"""
import hashlib
import base64
import io
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid

PACKAGE_URL = "https://cdn-zcode.z.ai/zcode/electron/releases/3.16.3/linux-x64/ZCode-3.16.3-linux-x64.deb"
PACKAGE_SHA256 = "128132b195977fd0a8c8a6c295cf42fe159222d7a5c5efefd8b5f00ef88ab8b6"
CLI_SHA256 = "97490d0f06a7a225d54d47817acf4b4859cdb268b59f2490caaeda982bef063a"
NODE_URL = "https://nodejs.org/dist/v24.14.0/node-v24.14.0-linux-x64.tar.xz"
NODE_SHA256 = "41cd79bb7877c81605a9e68ec4c91547774f46a40c67a17e34d7179ef11729df"
MODEL = "gpt-6-luna"
EFFORT = "high"
TOKEN_VARIABLE = "CODEX_CHATGPT_ACCESS_TOKEN"
BASE_URL = "https://chatgpt.com/backend-api/codex"
# The account has paid credits, so a run past the subscription's limit could spend them. Stop before it.
USED_PERCENT_STOP = 97.0
USED_PERCENT = re.compile(rb'"x-codex-primary-used-percent":\s*"([0-9.]+)"')
CONFIG_OPTIONS = [{"id": "model", "name": "Model", "category": "model", "type": "select",
                   "currentValue": MODEL, "options": [{"value": MODEL, "name": MODEL}]}]


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


def provider_config(token, account, session):
    """Select gpt-6-luna at effort high on the ChatGPT backend through native provider configuration.

    The backend refuses a stored response, so the request states `store: false` and asks for the encrypted
    reasoning that a stateless conversation replays. It keys its prompt cache on the session headers, which
    its own clients send, so the trial sends one session id.
    """
    return {"schemaVersion": 1, "config": {
        "providerConfigRules": {"providerRules": [{"providerId": "chatgpt", "config": {
            "group": "standard-personal", "access": {"type": "api-key", "apiKey": token},
            "api": {"type": "openai-responses", "baseUrl": BASE_URL, "headers": {
                "chatgpt-account-id": account, "originator": "zcode", "OpenAI-Beta": "responses=experimental",
                "session_id": session, "session-id": session, "x-client-request-id": session}},
            "personalModelIds": [MODEL]}}]},
        "modelConfigRules": {"providerModelRules": [{"providerId": "chatgpt", "modelId": MODEL,
            "config": {"properties": {"contextWindow": 1050000}, "optionSpecs": {
                "reasoningLevel": {"values": ["low", "medium", "high"],
                                   "map": '{"reasoning": {"effort": reasoningLevel}, "store": false, "include": ["reasoning.encrypted_content"]}'},
                "maxOutputTokens": {"max": 128000, "map": '{"max_output_tokens": null}'}}}}],
            "manualProviderModelRules": []},
        "defaultModelSelection": {"providerId": "chatgpt", "modelId": MODEL, "options": {"reasoningLevel": EFFORT}}}}


def highest_used_percent(home, offsets):
    """The highest subscription usage that the new bytes of ZCode's request records state, or 0."""
    highest = 0.0
    for path in home.glob(".zcode/cli/rollout/model-io-*.jsonl"):
        with path.open("rb") as file:
            file.seek(offsets.get(path, 0))
            data = file.read()
            offsets[path] = file.tell()
        highest = max([highest, *map(float, USED_PERCENT.findall(data))])
    return highest


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
    key = os.environ[TOKEN_VARIABLE]
    account = json.loads(base64.urlsafe_b64decode(key.split(".")[1] + "==="))["https://api.openai.com/auth"]["chatgpt_account_id"]
    session = str(uuid.uuid4())
    config = provider_config(key, account, session)
    recorded_config = provider_config("Harbor secret: " + TOKEN_VARIABLE, "account of the token", session)
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
        env.pop(TOKEN_VARIABLE)
        env["PATH"] = str(node.parent) + os.pathsep + env.get("PATH", "")
        prompt = "".join(block["text"] for block in params["prompt"] if block["type"] == "text")
        args = [str(node), str(cli), "--prompt", prompt, "--cwd", cwd, "--mode", "yolo", "--output-format", "stream-json", "--no-color"]
        result = None
        blocked = False
        calls, offsets, used_percent = set(), {}, 0.0
        with (logs / "events.jsonl").open("w") as output, (logs / "stderr.txt").open("w") as errors:
            process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=errors, text=True, start_new_session=True)
            try:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    event = json.loads(line)
                    payload = event.get("payload", {})
                    used_percent = max(used_percent, highest_used_percent(home, offsets))
                    if used_percent >= USED_PERCENT_STOP:
                        blocked = True
                        os.killpg(process.pid, signal.SIGTERM)
                        break
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
        (logs / "completion.json").write_text(json.dumps({"exit_code": exit_code, "quota_or_rate_stop": blocked, "used_percent": used_percent, "result": result}, indent=2))
        for path in logs.rglob("*"):
            if path.is_file() and not path.is_symlink() and key.encode() in path.read_bytes():
                path.unlink()
                raise RuntimeError("A native log contained the access token and was removed")
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
