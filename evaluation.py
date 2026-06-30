"""Offline evaluation of recommendation lists.

A recommender that predicts 4.2 stars accurately can still hand back a useless
list. So we don't stop at rating error -- we judge the actual top-N list two
ways:

  Accuracy (did we put relevant items near the top?)
    - Precision@K, Recall@K, Hit-Rate@K, NDCG@K, MRR
  Beyond accuracy (is the list healthy, not just accurate?)
    - Catalog coverage, novelty, intra-list diversity, popularity bias

"Relevant" for a user = the items in their held-out TEST set that they rated
highly (>= REL_THRESHOLD). The model only ever sees the TRAIN set; a good list
recovers these held-out favourites without having been shown them.
"""

import numpy as np
import pandas as pd
from . import config

REL_THRESHOLD = 4.0


# ----- per-user accuracy metrics (recommended/relevant are lists/sets of ids) -----
def precision_at_k(recommended, relevant, k=config.TOP_K):
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k=config.TOP_K):
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k=config.TOP_K):
    return 1.0 if any(i in relevant for i in recommended[:k]) else 0.0


def dcg_at_k(relevances, k=config.TOP_K):
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(recommended, relevant, k=config.TOP_K):
    gains = [1.0 if i in relevant else 0.0 for i in recommended[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(recommended, relevant, k=config.TOP_K):
    for rank, i in enumerate(recommended[:k], 1):
        if i in relevant:
            return 1.0 / rank
    return 0.0


# ----- beyond-accuracy metrics (computed across all users' lists) -----
def catalog_coverage(all_recommended, all_items):
    recommended_unique = set().union(*all_recommended) if all_recommended else set()
    return len(recommended_unique) / len(set(all_items))


def novelty(all_recommended, item_popularity, n_users):
    """Mean self-information -log2(p(item)). Popular items -> low novelty."""
    vals = []
    for rec in all_recommended:
        for i in rec:
            p = item_popularity.get(i, 1) / n_users
            vals.append(-np.log2(p)) if p > 0 else None
    return float(np.mean(vals)) if vals else 0.0


def _genre_set(genres_of, item_id):
    return set(str(genres_of.get(item_id, "")).split("|")) - {"", "nan"}


def intra_list_diversity(all_recommended, genres_of):
    """1 - average pairwise genre Jaccard similarity within each list."""
    per_list = []
    for rec in all_recommended:
        if len(rec) < 2:
            continue
        sims, pairs = 0.0, 0
        for a in range(len(rec)):
            for b in range(a + 1, len(rec)):
                ga, gb = _genre_set(genres_of, rec[a]), _genre_set(genres_of, rec[b])
                union = ga | gb
                jac = len(ga & gb) / len(union) if union else 0.0
                sims += jac
                pairs += 1
        per_list.append(1 - sims / pairs if pairs else 0.0)
    return float(np.mean(per_list)) if per_list else 0.0


def popularity_bias(all_recommended, item_popularity):
    """Average popularity (rating count) of recommended items. High = biased to hits."""
    vals = [item_popularity.get(i, 0) for rec in all_recommended for i in rec]
    return float(np.mean(vals)) if vals else 0.0


# ----- driver -----
def evaluate_model(model, train, test, items, k=config.TOP_K):
    item_popularity = train[config.ITEM_COL].value_counts().to_dict()
    n_users = train[config.USER_COL].nunique()
    genres_of = dict(zip(items[config.ITEM_COL], items.get(config.GENRES_COL, pd.Series())))

    # relevant test items per user (the highly-rated held-out ones)
    rel = test[test[config.RATING_COL] >= REL_THRESHOLD]
    relevant_by_user = rel.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()

    P = R = H = N = M = 0.0
    all_lists, n_eval = [], 0
    for user_id, relevant in relevant_by_user.items():
        recs = model.recommend(user_id, train, n=k, exclude_seen=True)
        rec_ids = [i for i, _ in recs]
        if not rec_ids:
            continue
        P += precision_at_k(rec_ids, relevant, k)
        R += recall_at_k(rec_ids, relevant, k)
        H += hit_rate_at_k(rec_ids, relevant, k)
        N += ndcg_at_k(rec_ids, relevant, k)
        M += mean_reciprocal_rank(rec_ids, relevant, k)
        all_lists.append(rec_ids)
        n_eval += 1

    if n_eval == 0:
        return {}
    return {
        f"Precision@{k}": P / n_eval,
        f"Recall@{k}": R / n_eval,
        f"HitRate@{k}": H / n_eval,
        f"NDCG@{k}": N / n_eval,
        "MRR": M / n_eval,
        "Coverage": catalog_coverage(all_lists, items[config.ITEM_COL]),
        "Novelty": novelty(all_lists, item_popularity, n_users),
        "Diversity": intra_list_diversity(all_lists, genres_of),
        "PopBias": popularity_bias(all_lists, item_popularity),
        "users_evaluated": n_eval,
    }


def compare_models(models, train, test, items, k=config.TOP_K):
    rows = {name: evaluate_model(m, train, test, items, k) for name, m in models.items()}
    return pd.DataFrame(rows).T
