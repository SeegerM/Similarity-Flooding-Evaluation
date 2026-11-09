"""The frozen PLM side: E5 embeddings, mean-pooled, L2-normalised, cached.

The language model is never trained (PAPER S1: "keeps the language model
frozen and trains only a lightweight structural encoder"), so every embedding
this project ever needs is a pure function of (checkpoint, prefix, text).  That
makes it cacheable, and caching is what makes the reconstruction affordable on
CPU: the encoder is the only expensive stage, and a table that appears in
several scenarios is embedded once.

The cache key covers the checkpoint, the prefix, the pooling and the exact text
list, so a configuration change can never silently reuse the wrong vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import SemStructConfig


def _key(cfg: SemStructConfig, texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [cfg.plm, cfg.e5_prefix, cfg.pool, cfg.l2_normalize_features,
         cfg.max_tokens, cfg.embed_dim], sort_keys=True).encode("utf-8"))
    digest.update(b"\x00")
    for text in texts:
        digest.update(text.encode("utf-8", "surrogatepass"))
        digest.update(b"\x00")
    return digest.hexdigest()


class Encoder:
    """Loads the checkpoint once; embeds text lists; optionally caches to disk."""

    def __init__(self, cfg: SemStructConfig, cache_dir: Optional[Path] = None,
                 device: str = "cpu", threads: Optional[int] = None) -> None:
        self.cfg = cfg
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._threads = threads
        self._tok = None
        self._model = None
        self._memo: Dict[str, np.ndarray] = {}

    # -- lazy load, so importing the package costs nothing ------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._threads:
            torch.set_num_threads(self._threads)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.plm)
        self._model = AutoModel.from_pretrained(self.cfg.plm).eval().to(self.device)
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n, d) float32.  Empty input returns a correctly shaped empty array."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.cfg.embed_dim), dtype=np.float32)

        # De-duplicate first: within one table the same value token recurs in
        # many rows, and the encoder should see each string once.
        order: Dict[str, int] = {}
        unique: List[str] = []
        for text in texts:
            if text not in order:
                order[text] = len(unique)
                unique.append(text)

        missing = [t for t in unique if t not in self._memo]
        if missing:
            for start in range(0, len(missing), self.cfg.embed_batch):
                batch = missing[start:start + self.cfg.embed_batch]
                vectors = self._encode_batch(batch)
                for text, vector in zip(batch, vectors):
                    self._memo[text] = vector

        return np.stack([self._memo[texts[i]] for i in range(len(texts))])

    def _encode_batch(self, batch: Sequence[str]) -> np.ndarray:
        import torch
        self._load()
        prefixed = [self.cfg.e5_prefix + t for t in batch]
        encoded = self._tok(prefixed, padding=True, truncation=True,
                            max_length=self.cfg.max_tokens, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if self.cfg.pool != "mean":
            raise ValueError(f"unsupported pooling {self.cfg.pool!r}")
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.cfg.l2_normalize_features:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    # -- disk cache over a whole text list ---------------------------------
    def encode_cached(self, texts: Sequence[str]) -> np.ndarray:
        if not self.cache_dir:
            return self.encode(texts)
        texts = list(texts)
        path = self.cache_dir / f"{_key(self.cfg, texts)}.npy"
        if path.exists():
            cached = np.load(path)
            if cached.shape == (len(texts), self.cfg.embed_dim):
                return cached
        vectors = self.encode(texts)
        # `.npy` is mandatory on the temporary too: np.save appends the suffix
        # when it is missing, and the rename would then chase a file that was
        # never written.
        temporary = path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(temporary, vectors)
        temporary.replace(path)
        return vectors

    def clear_memo(self) -> None:
        self._memo.clear()
