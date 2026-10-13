"""Context selectors. All fitted transformations use eligible pool rows only."""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def assert_no_leakage(pool, query, selected=None):
    assert pool.invoice_id.is_unique and query.invoice_id.is_unique
    assert not set(pool.invoice_id) & set(query.invoice_id)
    if selected is not None:
        assert selected.invoice_id.is_unique
        assert set(selected.invoice_id) <= set(pool.invoice_id)
        assert not set(selected.invoice_id) & set(query.invoice_id)


def mixed_features(pool, query):
    """Numeric scaling and nominal one-hot encoding, fitted on pool only."""
    excluded = ["invoice_id", "paid_late", "is_fraud"]
    numeric = [c for c in pool.select_dtypes(include="number") if c not in excluded]
    categorical = [c for c in pool.columns if c not in excluded + numeric]
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ])
    pool_x = transform.fit_transform(pool)
    query_x = transform.transform(query)
    return np.asarray(pool_x, dtype=float), np.asarray(query_x, dtype=float)


def _farthest(x, size, seed, candidates=None):
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(x)) if candidates is None else np.asarray(candidates)
    selected = [int(rng.choice(candidates))]
    minimum = np.linalg.norm(x - x[selected[0]], axis=1)
    while len(selected) < size:
        available = np.setdiff1d(candidates, selected, assume_unique=False)
        next_index = int(available[np.argmax(minimum[available])])
        selected.append(next_index)
        minimum = np.minimum(minimum, np.linalg.norm(x - x[next_index], axis=1))
    return selected


def select_context(pool, target, query, strategy, context_size, random_state, pool_x=None, query_x=None):
    """Return selected rows and construction seconds. Similarity/hybrid use one query row."""
    start = time.perf_counter()
    assert_no_leakage(pool, query)
    if context_size > len(pool) or context_size < 2:
        raise ValueError("Invalid context budget")
    rng = np.random.default_rng(random_state)
    labels = pool[target].to_numpy()
    positions = np.arange(len(pool))
    if strategy == "random":
        chosen = rng.choice(positions, context_size, replace=False)
    elif strategy in ("stratified", "balanced"):
        positives = positions[labels == 1]
        negatives = positions[labels == 0]
        expected = round(context_size * len(positives) / len(pool)) if strategy == "stratified" else context_size // 2
        n_positive = min(len(positives), max(1, expected), context_size - 1)
        n_negative = context_size - n_positive
        if n_negative > len(negatives):
            n_negative = len(negatives)
            n_positive = context_size - n_negative
        chosen = np.r_[rng.choice(positives, n_positive, replace=False),
                       rng.choice(negatives, n_negative, replace=False)]
        rng.shuffle(chosen)
    elif strategy in ("diversity", "similarity", "hybrid"):
        if pool_x is None or query_x is None:
            pool_x, query_x = mixed_features(pool, query)
        if strategy == "diversity":
            chosen = _farthest(pool_x, context_size, random_state)
        else:
            if len(query) != 1:
                raise ValueError("Query-specific selection requires one query row")
            distance = np.linalg.norm(pool_x - query_x[0], axis=1)
            nearest = np.argsort(distance, kind="stable")
            if strategy == "similarity":
                chosen = nearest[:context_size]
            else:
                # Half nearest, then minority representation and diverse coverage.
                nearby = list(nearest[:context_size // 2])
                minority = list(np.intersect1d(nearest[:min(4 * context_size, len(pool))],
                    positions[labels == 1], assume_unique=False))
                minority.sort(key=lambda i: distance[i])
                minority = [i for i in minority if i not in nearby][:max(1, context_size // 4)]
                chosen = nearby + minority
                if len(chosen) < context_size:
                    candidates = [i for i in positions if i not in chosen]
                    extra = _farthest(pool_x, context_size - len(chosen), random_state, candidates)
                    chosen += extra
    else:
        raise ValueError(strategy)
    selected = pool.iloc[list(chosen)].copy()
    assert len(selected) == context_size
    assert_no_leakage(pool, query, selected)
    return selected, time.perf_counter() - start
