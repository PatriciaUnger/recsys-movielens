"""Data loading, splitting and basic stats for the recommender prototype.

Everything downstream (baselines, content-based, CF, matrix factorization)
reads the same two DataFrames produced here, so this is the single place that
knows about file formats and column names.
"""

import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from . import config

# Where to fetch the dataset if it is missing (e.g. on a fresh cloud deploy).
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_data():
    """Make sure ratings.csv and movies.csv exist locally.

    On your own machine they are already in data/raw/. On a hosting service like
    Streamlit Community Cloud the data folder isn't committed (the MovieLens
    licence doesn't allow redistributing it), so the first run downloads
    ml-latest-small straight from GroupLens and extracts just the two files we
    use. This keeps the dataset out of the repository while still letting the
    deployed app run.
    """
    if config.RATINGS_PATH.exists() and config.ITEMS_PATH.exists():
        return
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MOVIELENS_URL, timeout=60) as resp:
        archive = resp.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in ("ratings.csv", "movies.csv"):
                with z.open(name) as src, open(config.RAW_DATA_DIR / base, "wb") as out:
                    out.write(src.read())


def load_ratings(path=config.RATINGS_PATH):
    """Load the user-item ratings table (MovieLens: userId, movieId, rating, timestamp)."""
    ensure_data()
    ratings = pd.read_csv(path)
    required = {config.USER_COL, config.ITEM_COL, config.RATING_COL}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings file is missing columns: {missing}")
    return ratings


def load_items(path=config.ITEMS_PATH):
    """Load the item metadata table (MovieLens: movieId, title, genres)."""
    ensure_data()
    items = pd.read_csv(path)
    required = {config.ITEM_COL, config.TITLE_COL}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items file is missing columns: {missing}")
    return items


def describe_dataset(ratings, items=None):
    """Return a dict of the headline statistics and print a short summary.

    These are the numbers the EDA section of the report is built on:
    how many users/items/ratings we have, how sparse the matrix is, and
    where the popularity mass sits (long tail).
    """
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_items)

    counts_per_item = ratings[config.ITEM_COL].value_counts()
    counts_per_user = ratings[config.USER_COL].value_counts()

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "avg_ratings_per_user": counts_per_user.mean(),
        "avg_ratings_per_item": counts_per_item.mean(),
        "rating_min": ratings[config.RATING_COL].min(),
        "rating_max": ratings[config.RATING_COL].max(),
        "rating_mean": ratings[config.RATING_COL].mean(),
    }

    print("Dataset summary")
    print(f"  users          : {n_users:,}")
    print(f"  items          : {n_items:,}")
    print(f"  ratings        : {n_ratings:,}")
    print(f"  sparsity       : {sparsity:.4%}")
    print(f"  ratings/user   : {stats['avg_ratings_per_user']:.1f} (avg)")
    print(f"  ratings/item   : {stats['avg_ratings_per_item']:.1f} (avg)")
    print(f"  rating range   : {stats['rating_min']} - {stats['rating_max']} "
          f"(mean {stats['rating_mean']:.2f})")
    return stats


def train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE):
    """Per-user hold-out split.

    A plain random split can leave a user with all rows in the test set and
    nothing to learn from, which makes personalized methods impossible to
    evaluate for that user. So instead we hold out `test_size` of EACH user's
    ratings. Users with very few ratings keep at least one row in train.
    """
    rng = np.random.default_rng(random_state)
    test_idx = []
    for _, group in ratings.groupby(config.USER_COL):
        n = len(group)
        if n < 2:
            continue  # too few ratings to hold any out
        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)  # always leave >=1 in train
        test_idx.extend(rng.choice(group.index.values, size=n_test, replace=False))

    test = ratings.loc[test_idx]
    train = ratings.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_seen_items(ratings, user_id):
    """Items a user has already rated -- we never recommend these back."""
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])


def build_user_item_matrix(ratings):
    """Pivot ratings into a (users x items) DataFrame; NaN = not rated.

    Used by the collaborative-filtering models. Returned as a dense pandas
    pivot for clarity; we move to sparse matrices later when we discuss
    scalability.
    """
    return ratings.pivot_table(index=config.USER_COL,
                               columns=config.ITEM_COL,
                               values=config.RATING_COL)
