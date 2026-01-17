#!/usr/bin/env python3

"""
Compare two embedding spaces using cosine similarity and k-NN overlap.

Usage:
    python compare_embeddings.py --emb_a path/to/embeddings_a.npy --emb_b path/to/embeddings_b.npy --k 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Loading
# -----------------------------

def load_embeddings(path: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".npy":
        Z = np.load(path)
    elif path.suffix == ".csv":
        Z = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Only .npy and .csv files are supported")

    if Z.ndim != 2:
        raise ValueError("Embeddings must have shape (N, D)")

    return Z


def normalize(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding detected")
    return Z / norms


# -----------------------------
# Metrics
# -----------------------------

def cosine_stats(Z1: np.ndarray, Z2: np.ndarray):
    Z1 = normalize(Z1)
    Z2 = normalize(Z2)

    cos = np.sum(Z1 * Z2, axis=1)

    return {
        "mean_cosine_similarity": float(np.mean(cos)),
        "var_cosine_similarity": float(np.var(cos)),
    }


def knn_overlap(Z1: np.ndarray, Z2: np.ndarray, k: int):
    Z1 = normalize(Z1)
    Z2 = normalize(Z2)

    nn1 = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z1)
    nn2 = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z2)

    idx1 = nn1.kneighbors(Z1, return_distance=False)
    idx2 = nn2.kneighbors(Z2, return_distance=False)

    overlaps = []
    for i in range(Z1.shape[0]):
        n1 = set(idx1[i][1:])  # drop self
        n2 = set(idx2[i][1:])
        overlaps.append(len(n1 & n2) / k)

    return {
        "knn_overlap_mean": float(np.mean(overlaps)),
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two embedding spaces (paired samples)"
    )
    parser.add_argument("--emb_a", required=True, help="Path to embeddings A (.npy or .csv)")
    parser.add_argument("--emb_b", required=True, help="Path to embeddings B (.npy or .csv)")
    parser.add_argument("--k", type=int, default=10, help="k for k-NN overlap")

    args = parser.parse_args()

    Z_A = load_embeddings(args.emb_a)
    Z_B = load_embeddings(args.emb_b)

    if Z_A.shape != Z_B.shape:
        raise ValueError(
            f"Shape mismatch: {Z_A.shape} vs {Z_B.shape}"
        )

    results = {}
    results.update(cosine_stats(Z_A, Z_B))
    results.update(knn_overlap(Z_A, Z_B, k=args.k))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
