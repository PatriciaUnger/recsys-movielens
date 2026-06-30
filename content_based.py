"""Content-based recommender.

Idea: describe each movie by its content (here: its genres), describe each user
by the content of the movies they liked, and recommend movies whose content is
closest to that taste profile. Unlike the popularity baselines, this is
personalized -- two users with different tastes get different lists -- and it
can recommend brand-new or niche films as long as we know their genres.

Pipeline (matches the course slides):
  1. Turn each movie's genres into a TF-IDF vector.
  2. Build a user profile = sum over the movies they rated of
     (rating - user's average rating) * movie_vector.
     Centering by the user's average means a movie rated *above* their norm
     pushes the profile toward its genres, and one rated *below* pushes away.
  3. Score every movie by cosine similarity to the profile; recommend the
     highest-scoring unseen movies.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import config


class ContentBasedRecommender:
    def __init__(self, feature_col=config.GENRES_COL, use_tfidf=True):
        self.feature_col = feature_col
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.item_features_ = None      # sparse matrix (n_items x n_features)
        self.item_ids_ = None           # item id for each matrix row
        self.item_id_to_index_ = None

    def fit(self, ratings, items):
        # genres look like "Action|Comedy" -> treat as the text "Action Comedy"
        text = items[self.feature_col].fillna("").str.replace("|", " ", regex=False)
        # use_idf=False reduces this to plain (normalized) genre counts, which is
        # how we later compare "TF-IDF vs raw genre vectors" as an extension.
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+", use_idf=self.use_tfidf)
        self.item_features_ = self.vectorizer.fit_transform(text)
        self.item_ids_ = items[config.ITEM_COL].to_numpy()
        self.item_id_to_index_ = {iid: i for i, iid in enumerate(self.item_ids_)}
        return self

    def build_user_profile(self, user_id, ratings_train):
        rows = ratings_train[ratings_train[config.USER_COL] == user_id]
        if rows.empty:
            return None
        mean = rows[config.RATING_COL].mean()

        idx, weights = [], []
        for iid, r in zip(rows[config.ITEM_COL], rows[config.RATING_COL]):
            j = self.item_id_to_index_.get(iid)
            if j is not None:
                idx.append(j)
                weights.append(r - mean)
        if not idx:
            return None

        weights = np.asarray(weights)
        # if the user rated everything the same, centering kills the signal;
        # fall back to raw ratings so we still get a usable profile
        if not np.any(weights):
            weights = rows[config.RATING_COL].to_numpy()[:len(idx)]

        profile = weights @ self.item_features_[idx].toarray()  # (n_features,)
        return profile.reshape(1, -1)

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        from .data_loading import get_seen_items
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None or not np.any(profile):
            return []

        scores = cosine_similarity(profile, self.item_features_).ravel()
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()

        order = np.argsort(-scores)
        out = []
        for j in order:
            iid = self.item_ids_[j]
            if exclude_seen and iid in seen:
                continue
            out.append((int(iid), float(scores[j])))
            if len(out) >= n:
                break
        return out

    def similar_items(self, item_id, n=config.TOP_K):
        """Movies most similar in content to a given movie (for 'more like this')."""
        j = self.item_id_to_index_.get(item_id)
        if j is None:
            return []
        scores = cosine_similarity(self.item_features_[j], self.item_features_).ravel()
        order = np.argsort(-scores)
        return [(int(self.item_ids_[k]), float(scores[k])) for k in order if k != j][:n]
