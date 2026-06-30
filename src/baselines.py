"""Non-personalized baselines.

These ignore who the user is and recommend the same ranked list to everyone
(minus items they've already seen). They matter for two reasons: they solve the
cold-start problem (a brand-new user still gets sensible recommendations), and
they are the bar every personalized method has to beat in the evaluation.
"""

import numpy as np
import pandas as pd
from . import config


def _top_n_from_ranking(ranking, seen, n, exclude_seen):
    """Take the first n items from a pre-sorted (item_id, score) ranking."""
    out = []
    for item_id, score in ranking:
        if exclude_seen and item_id in seen:
            continue
        out.append((item_id, float(score)))
        if len(out) >= n:
            break
    return out


class MostPopularRecommender:
    """Top-N by interaction count: recommend what the most people have rated.

    This is the classic 'trending / most watched' list. It's a strong baseline
    precisely because popular items are popular -- but that's also its weakness:
    it amplifies popularity bias and never surfaces the long tail.
    """

    def __init__(self):
        self.ranking_ = None

    def fit(self, ratings, items=None):
        counts = ratings[config.ITEM_COL].value_counts()
        self.ranking_ = list(zip(counts.index, counts.values))
        return self

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        from .data_loading import get_seen_items
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        return _top_n_from_ranking(self.ranking_, seen, n, exclude_seen)


class WeightedAverageRatingRecommender:
    """Highest average rating, shrunk toward the global mean (IMDb-style).

    Naive 'highest average' is dominated by items with a single 5.0 rating.
    The course's weighted formula fixes this:

        WR(j) = v/(v+m) * R(j) + m/(v+m) * C

    where R(j) is item j's mean rating, v is its number of ratings, C is the
    global mean rating, and m is a smoothing constant (how many ratings an item
    needs before we trust its own average). Items with few ratings get pulled
    toward C; well-rated popular items keep their high score.
    """

    def __init__(self, m=20):
        self.m = m
        self.ranking_ = None

    def fit(self, ratings, items=None):
        grouped = ratings.groupby(config.ITEM_COL)[config.RATING_COL]
        R = grouped.mean()
        v = grouped.count()
        C = ratings[config.RATING_COL].mean()
        wr = (v / (v + self.m)) * R + (self.m / (v + self.m)) * C
        wr = wr.sort_values(ascending=False)
        self.ranking_ = list(zip(wr.index, wr.values))
        return self

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        from .data_loading import get_seen_items
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        return _top_n_from_ranking(self.ranking_, seen, n, exclude_seen)


class RandomRecommender:
    """Sanity-check baseline: recommend random unseen items.

    Any method worth keeping must clearly beat this. It's also a useful
    reference point for coverage and novelty, where random scores 'well' for
    the wrong reasons.
    """

    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        self.items_ = ratings[config.ITEM_COL].unique()
        return self

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        from .data_loading import get_seen_items
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        rng = np.random.default_rng(self.random_state + int(user_id))
        candidates = [i for i in self.items_ if i not in seen]
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(int(i), 0.0) for i in chosen]
