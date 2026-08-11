#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:11434}"
API_KEY="${API_KEY:-}"
MODEL_TAG="${MODEL_TAG:-Qwen/Qwen2.5-7B-Instruct}"

auth_args=()
if [[ -n "${API_KEY}" ]]; then
  auth_args=(-H "Authorization: Bearer ${API_KEY}" -H "X-API-Key: ${API_KEY}")
fi

echo "=== 1. Health & Readiness Probes ==="
curl -fsS "${auth_args[@]}" "${API_BASE_URL}/healthz"
echo
curl -fsS "${auth_args[@]}" "${API_BASE_URL}/readyz"
echo

echo "=== 2. Ollama API: Tags & PS ==="
curl -fsS "${auth_args[@]}" "${API_BASE_URL}/api/tags"
echo
curl -fsS "${auth_args[@]}" "${API_BASE_URL}/api/ps"
echo

echo "=== 3. Ollama API: Chat ==="
curl -fsS "${auth_args[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_TAG}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"stream\":false}" \
  "${API_BASE_URL}/api/chat"
echo

echo "=== 4. OpenAI API: Models ==="
curl -fsS "${auth_args[@]}" "${API_BASE_URL}/v1/models"
echo

echo "=== 5. OpenAI API: Chat Completions ==="
curl -fsS "${auth_args[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_TAG}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}]}" \
  "${API_BASE_URL}/v1/chat/completions"
echo
