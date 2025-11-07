"""Utility script to verify LM Studio OpenAI-compatible server availability.

Usage (from project root, venv activated):
    python check_lmstudio.py

It will:
1. Read .env (if present) for LM_STUDIO_BASE_URL & EMBEDDING_MODEL_ID
2. Query /v1/models
3. Print whether the configured model id is among loaded models.

Exit codes:
 0 success, 1 server unreachable, 2 model not found.
"""

from __future__ import annotations
import os, json, sys, urllib.request

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "embedding-gemma-2b")

def main():
    endpoint = f"{BASE_URL}/models"
    print(f"Checking LM Studio server at: {endpoint}")
    try:
        with urllib.request.urlopen(endpoint, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[ERROR] Could not reach LM Studio: {e}")
        return 1

    models = []
    # LM Studio typically returns {"data":[{"id": "model-name", ...}, ...]}
    if isinstance(data, dict) and "data" in data:
        models = [m.get("id") for m in data["data"] if isinstance(m, dict)]
    else:
        print("[WARN] Unexpected /v1/models response:")
        print(data)

    print("Loaded models:", models or "<none>")
    if MODEL_ID not in models:
        print(f"[FAIL] Configured EMBEDDING_MODEL_ID '{MODEL_ID}' not among loaded models.")
        print("Load it in LM Studio (Developer tab) and ensure embeddings are enabled.")
        return 2

    print(f"[OK] Model '{MODEL_ID}' is available.")
    return 0

if __name__ == "__main__":
    sys.exit(main())