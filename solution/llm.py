"""OpenAI access (embeddings + gpt-4o-mini), used only where the rules fall short.

* The key comes from the environment (OPENAI_API_KEY), else from a `.env` file next to the
  submission or above it. It is checked once, when the system is built.
* No key, no package, or any API error: the call returns None and the caller keeps its rules-only
  answer. An LLM problem can never break an answer.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBED_BATCH = 256
EMBED_WORKERS = 3            # parallel requests at ingest; more trips the per-minute rate limit
EMBED_RETRIES = 4            # per batch at ingest, with back-off 2, 4, 8, 16 s
EMBED_CHARS = 1500           # chunks are short; this only guards against very long ones


def _key_from_dotenv() -> Optional[str]:
    for d in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        f = d / ".env"
        if f.is_file():
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*$", line)
                    if m:
                        v = m.group(1).strip().strip('"').strip("'")
                        if v:
                            return v
            except OSError:
                return None
    return None


class LLM:
    def __init__(self) -> None:
        self.enabled = False
        self.client = None
        self.np = None
        self.errors = 0
        self.calls = 0
        self.last_error = ""
        key = os.environ.get("OPENAI_API_KEY") or _key_from_dotenv()
        if not key:
            return
        try:
            import numpy as np
            from openai import OpenAI
            self.client = OpenAI(api_key=key, max_retries=1)
            self.np = np
            self.enabled = True
        except Exception:
            self.enabled = False

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], timeout: float = 30.0, retries: int = 0):
        """Unit-length float32 vectors (rows) for `texts`, or None if nothing could be embedded.
        Large jobs run in batches with limited parallelism; a failed batch is retried with back-off,
        and a batch that still fails gets zero rows (no match) instead of losing every batch."""
        if not self.enabled or not texts:
            return None
        np = self.np
        batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]

        def run(batch):
            for attempt in range(retries + 1):
                try:
                    self.calls += 1
                    resp = self.client.embeddings.create(
                        model=EMBED_MODEL, input=[(t or " ")[:EMBED_CHARS] for t in batch], timeout=timeout)
                    return [d.embedding for d in resp.data]
                except Exception as e:       # rate limits, timeouts: wait and try again
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                    if attempt < retries:
                        time.sleep(min(30.0, 2.0 * 2 ** attempt))
            return None

        if len(batches) == 1:
            parts = [run(batches[0])]
        else:
            with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
                parts = list(ex.map(run, batches))
        if all(p is None for p in parts):
            return None
        dim = next(len(p[0]) for p in parts if p)
        rows = []
        for batch, part in zip(batches, parts):
            rows.extend(part if part is not None else [[0.0] * dim] * len(batch))
        m = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m /= np.where(norms > 0, norms, 1.0)
        return m

    # ------------------------------------------------------------------ chat
    def chat(self, system: str, user: str, max_tokens: int = 300, timeout: float = 3.0) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            self.calls += 1
            resp = self.client.chat.completions.create(
                model=CHAT_MODEL, temperature=0, seed=7, max_tokens=max_tokens, timeout=timeout,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            self.errors += 1
            return None
