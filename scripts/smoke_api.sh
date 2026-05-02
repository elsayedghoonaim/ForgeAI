#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-}"

auth_args=()
if [[ -n "${API_KEY}" ]]; then
  auth_args=(-H "X-API-Key: ${API_KEY}")
fi

curl -fsS "${auth_args[@]}" "${API_BASE_URL}/healthz"
echo
curl -fsS "${auth_args[@]}" "${API_BASE_URL}/readyz"
echo
curl -fsS "${auth_args[@]}" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}]}' \
  "${API_BASE_URL}/v1/chat/completions"
echo
