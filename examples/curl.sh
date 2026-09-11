#!/bin/bash
# G69: curl examples against a local server (./build/llm-cpp serve --port 8080)
set -u
BASE="${BASE:-http://127.0.0.1:8080}"
echo "== health =="
curl -s "$BASE/healthz"; echo
echo "== completion =="
curl -s -X POST "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":20,"temperature":0}'; echo
echo "== chat =="
curl -s -X POST "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'; echo
echo "== stream =="
curl -s -N -X POST "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":10,"temperature":0,"stream":true}'; echo
