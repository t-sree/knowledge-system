"""OpenAI access (embeddings + gpt-4o-mini), used only where the rules fall short.

* The key comes from the environment (OPENAI_API_KEY), else from a `.env` file next to the
  submission or above it. It is checked once, when the system is built.
* No key, no package, or any API error: the call returns None and the caller keeps its rules-only
  answer. An LLM problem can never break an answer.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBED_BATCH = 256
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
    def embed(self, texts: list[str], timeout: float = 30.0):
        """Unit-length float32 vectors (rows) for `texts`, or None on failure."""
        if not self.enabled or not texts:
            return None
        np = self.np
        batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]

        def run(batch):
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=[(t or " ")[:EMBED_CHARS] for t in batch],
                                                 timeout=timeout)
            return [d.embedding for d in resp.data]
        try:
            self.calls += len(batches)
            if len(batches) == 1:
                rows = run(batches[0])
            else:
                with ThreadPoolExecutor(max_workers=6) as ex:
                    rows = [r for part in ex.map(run, batches) for r in part]
            m = np.asarray(rows, dtype=np.float32)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            return m
        except Exception:
            self.errors += 1
            return None

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
