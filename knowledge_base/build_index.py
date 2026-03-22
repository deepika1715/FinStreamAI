"""
FinStreamAI — Week 1, Step 1
Builds a FAISS index using TF-IDF vectors.
Zero cost. Zero new installs. Uses only numpy + faiss (already installed).
Run once: python knowledge_base/build_index.py
"""

import pickle
import math
import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

BASE_DIR      = Path(__file__).parent
PATTERNS_FILE = BASE_DIR / "fraud_patterns.txt"
INDEX_FILE    = BASE_DIR / "index.faiss"
CHUNKS_FILE   = BASE_DIR / "chunks.pkl"
VOCAB_FILE    = BASE_DIR / "vocab.pkl"


def load_and_chunk(filepath: Path) -> list[str]:
    text   = filepath.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    print(f"Loaded {len(chunks)} fraud pattern chunks")
    return chunks


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_tfidf(chunks: list[str]) -> tuple[np.ndarray, list[str]]:
    tokenised = [tokenise(c) for c in chunks]

    # Build vocabulary from all tokens
    all_tokens = [t for doc in tokenised for t in doc]
    vocab      = sorted(set(all_tokens))
    word2idx   = {w: i for i, w in enumerate(vocab)}
    print(f"Vocabulary size: {len(vocab)}")

    n_docs = len(chunks)

    # Document frequency
    df = Counter()
    for doc in tokenised:
        for word in set(doc):
            df[word] += 1

    # TF-IDF matrix
    matrix = np.zeros((n_docs, len(vocab)), dtype="float32")
    for d, doc in enumerate(tokenised):
        tf = Counter(doc)
        total = len(doc)
        for word, count in tf.items():
            if word in word2idx:
                tf_val  = count / total
                idf_val = math.log((n_docs + 1) / (df[word] + 1)) + 1
                matrix[d, word2idx[word]] = tf_val * idf_val

    # L2 normalise each row
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    print(f"TF-IDF matrix shape: {matrix.shape}")
    return matrix, vocab


def build_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    # Inner product on L2-normalised vectors = cosine similarity
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    print(f"Index built — {index.ntotal} vectors stored")
    return index


def save_artifacts(index, chunks: list[str], vocab: list[str]) -> None:
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved: {INDEX_FILE}")
    print(f"Saved: {CHUNKS_FILE}")
    print(f"Saved: {VOCAB_FILE}")


def main():
    chunks         = load_and_chunk(PATTERNS_FILE)
    matrix, vocab  = build_tfidf(chunks)
    index          = build_index(matrix)
    save_artifacts(index, chunks, vocab)
    print("\nDone. knowledge_base ready.")


if __name__ == "__main__":
    main()
