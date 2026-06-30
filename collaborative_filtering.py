"""Collaborative filtering: item-item and user-user.

CF ignores what a film *is* and looks only at the ratings matrix: "people who
agreed in the past will agree in the future." It needs no genres or metadata
(content-agnostic), which is its strength, but it suffers when the matrix is
sparse and is expensive on large catalogues -- the trade-offs from the course.

Both variants below use the course's mean-centered prediction:

    S(u,i) = baseline + sum_n (rating_n - mean_n) * w / sum_n |w|

and both restrict the sum to the top-k most similar neighbours (k << n), which
is the scalability trick from the computational-cost slide: instead of every
neighbour we keep only the k strongest similarities per row.
"""

import numpy as np
from . import config


def _matrix(ratings):
    """Return (R, users, items) where R is a dense users x items array with NaN
    for unobserved entries, plus the id<->index lookups."""
    users = np.sort(ratings[config.USER_COL].unique())
    items = np.sort(ratings[config.ITEM_COL].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: j for j, it in enumerate(items)}
    R = np.full((len(users), len(items)), np.nan)
    ui = ratings[config.USER_COL].map(u_idx).to_numpy()
    ii = ratings[config.ITEM_COL].map(i_idx).to_numpy()
    R[ui, ii] = ratings[config.RATING_COL].to_numpy()
    return R, users, items, u_idx, i_idx


def _keep_top_k(S, k):
    """Zero every entry of each row of S except its k largest (by value)."""
    if k is None or k >= S.shape[1]:
        return S
    out = np.zeros_like(S)
    part = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    out[rows, part] = S[rows, part]
    return out


class ItemItemCollaborativeFiltering:
    """Recommend items similar to ones the user already liked.

    Similarity = adjusted cosine (ratings centered by each user's mean before
    comparing item columns, so a generous and a harsh rater still count as
    'agreeing' when they deviate the same way). More stable and cheaper to cache
    than user-user; tends toward safe, 'obvious' picks (the slide's TRUST side).
    """

    def __init__(self, k=20):
        self.k = k

    def fit(self, ratings):
        R, self.users_, self.items_, self.u_idx_, self.i_idx_ = _matrix(ratings)
        user_mean = np.nanmean(R, axis=1)
        Rc = np.nan_to_num(R - user_mean[:, None])            # user-centered, unobserved=0
        norms = np.linalg.norm(Rc, axis=0)
        norms[norms == 0] = 1e-9
        S = (Rc.T @ Rc) / np.outer(norms, norms)              # adjusted cosine
        np.fill_diagonal(S, 0.0)
        self.S_ = _keep_top_k(S, self.k)
        self.item_mean_ = np.nan_to_num(np.nanmean(R, axis=0))
        self.observed_ = ~np.isnan(R)
        self.R_ = R
        return self

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        if user_id not in self.u_idx_:
            return []
        u = self.u_idx_[user_id]
        rated = self.observed_[u]
        e = np.where(rated, np.nan_to_num(self.R_[u]) - self.item_mean_, 0.0)  # (r_uj - rbar_j)
        num = self.S_ @ e
        den = np.abs(self.S_) @ rated.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pred = self.item_mean_ + np.where(den > 0, num / den, 0.0)
        if exclude_seen:
            pred[rated] = -np.inf
        order = np.argsort(-pred)[:n]
        return [(int(self.items_[j]), float(pred[j])) for j in order if np.isfinite(pred[j])]


class UserUserCollaborativeFiltering:
    """Recommend items liked by users similar to the target user.

    Similarity = mean-centered cosine, which is Pearson correlation computed over
    the shared rating pattern (the slide's Pearson weight). Great at surprising,
    useful discoveries, but the most expensive and the most exposed to sparsity:
    if nobody similar has rated an item, it can't score it.
    """

    def __init__(self, k=30):
        self.k = k

    def fit(self, ratings):
        R, self.users_, self.items_, self.u_idx_, self.i_idx_ = _matrix(ratings)
        self.user_mean_ = np.nanmean(R, axis=1)
        Rc = np.nan_to_num(R - self.user_mean_[:, None])
        norms = np.linalg.norm(Rc, axis=1)
        norms[norms == 0] = 1e-9
        W = (Rc @ Rc.T) / np.outer(norms, norms)              # Pearson-style weights
        np.fill_diagonal(W, 0.0)
        self.W_ = _keep_top_k(W, self.k)
        self.D_ = Rc                                          # deviations (r_vi - rbar_v)
        self.M_ = (~np.isnan(R)).astype(float)                # who rated what
        self.observed_ = ~np.isnan(R)
        return self

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        if user_id not in self.u_idx_:
            return []
        u = self.u_idx_[user_id]
        w = self.W_[u]
        num = w @ self.D_
        den = np.abs(w) @ self.M_
        with np.errstate(divide="ignore", invalid="ignore"):
            pred = self.user_mean_[u] + np.where(den > 0, num / den, 0.0)
        if exclude_seen:
            pred[self.observed_[u]] = -np.inf
        order = np.argsort(-pred)[:n]
        return [(int(self.items_[j]), float(pred[j])) for j in order if np.isfinite(pred[j])]
