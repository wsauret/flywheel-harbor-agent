# Harbor wrapper with OpenCode Go to OpenRouter failover

This branch runs a stock reference harness on DeepSeek-V4.1-Flash-1017 at max effort. Its model requests go through `failover_proxy.py`, which sends each request to the OpenCode Go subscription first and moves it to OpenRouter, pinned to Baseten with no fallback, when Go refuses for its usage limit or saturation (status 429, 502, 503, or 529). After a refusal, later requests skip Go until the reset time that Go states, or for five minutes, and then try Go again.

The harness itself never receives a key. Give the trial `OPENCODE_API_KEY` and `OPENROUTER_API_KEY` as Harbor secrets. The bridge starts the proxy in its own process and points the harness at `http://127.0.0.1:<port>/v1`. Each request is logged to `failover-requests.jsonl` in the agent logs with the serving provider, the status, the request and response byte counts, and the duration. Keys, headers, and content are never logged.

## Using the proxy from any script

`failover_proxy.py` uses only the Python standard library. Copy it next to your script, export the two keys, and then:

```python
import failover_proxy
base_url = failover_proxy.start(log_path="requests.jsonl")  # "http://127.0.0.1:<port>/v1"
# Point any OpenAI-compatible client (chat completions or responses API) at base_url with any API key.
```

Or run it as a server: `python3 failover_proxy.py --port 8787 --log requests.jsonl`. The proxy ignores the client's API key and model name and sets the model of each route itself: `FAILOVER_GO_MODEL` (default `deepseek-v4.1-flash`) and `FAILOVER_OR_MODEL` (default `deepseek/deepseek-v4.1-flash-1017`). Without `OPENROUTER_API_KEY`, a Go refusal goes back to the client; without `OPENCODE_API_KEY`, every request goes to OpenRouter.
