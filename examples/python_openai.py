"""G69: python OpenAI-style client (stdlib only)."""
import json
import os
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")


def post(path, payload, stream=False):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        if stream:
            for line in r:
                line = line.decode().strip()
                if line == "data: [DONE]":
                    break
                if line.startswith("data:"):
                    print(json.loads(line[5:])["choices"][0]["text"], end="", flush=True)
            print()
        else:
            return json.load(r)


if __name__ == "__main__":
    with urllib.request.urlopen(BASE + "/healthz") as r:
        print("health:", json.load(r))
    r = post("/v1/completions", {"prompt": "Hello", "max_tokens": 20, "temperature": 0})
    print("completion:", r["choices"][0]["text"][:80])
    r = post("/v1/chat/completions",
             {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20})
    print("chat:", r["choices"][0]["message"]["content"][:80])
    print("stream:", end=" ")
    post("/v1/completions", {"prompt": "Hello", "max_tokens": 10, "stream": True}, stream=True)
