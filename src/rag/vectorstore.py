"""FAISS-backed dense index with on-disk persistence.

Vectors arrive L2-normalised from the embedding layer, so an inner-product
index (IndexFlatIP) gives exact cosine similarity. Flat is deliberate: at
portfolio corpus sizes an approximate index (IVF/HNSW) would trade recall for
a speed-up nobody can measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .models import Chunk, ScoredChunk

_INDEX_FILE = "dense.faiss"
_META_FILE = "chunks.jsonl"
_INFO_FILE = "index_info.json"


class VectorStore:
    def __init__(self, dim: int, embedder_name: str = "unknown") -> None:
        self.dim = dim
        self.embedder_name = embedder_name
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors; they must line up."
            )
        if not chunks:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Index has dim {self.dim} but vectors have dim {vectors.shape[1]}. "
                "Re-ingest after changing the embedding backend."
            )
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        )
        top_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:  # FAISS pads with -1 when it has fewer hits than asked for
                continue
            results.append(
                ScoredChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                    component_scores={"dense": float(score)},
                )
            )
        return results

    # --- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _INDEX_FILE))
        with (directory / _META_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (directory / _INFO_FILE).write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "embedder": self.embedder_name,
                    "count": len(self._chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        directory = Path(directory)
        info_path = directory / _INFO_FILE
        if not info_path.exists():
            raise FileNotFoundError(
                f"No index at {directory}. Run scripts/ingest.py first."
            )
        info = json.loads(info_path.read_text(encoding="utf-8"))
        store = cls(dim=info["dim"], embedder_name=info.get("embedder", "unknown"))
        store._index = faiss.read_index(str(directory / _INDEX_FILE))
        with (directory / _META_FILE).open(encoding="utf-8") as handle:
            store._chunks = [
                Chunk.from_dict(json.loads(line)) for line in handle if line.strip()
            ]
        if store._index.ntotal != len(store._chunks):
            raise ValueError(
                f"Index is corrupt: {store._index.ntotal} vectors vs "
                f"{len(store._chunks)} chunks. Re-ingest."
            )
        return store
