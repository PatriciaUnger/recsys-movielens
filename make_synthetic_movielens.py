"""Generate a small MovieLens-format dataset for local development.

This is ONLY a stand-in so the pipeline runs before the real data is in place.
For the actual submission, download MovieLens (ml-latest-small) and put
ratings.csv and movies.csv in data/raw/. The format produced here matches
MovieLens exactly, so no code changes are needed when you swap in real data.

The generator builds genre-driven user preferences on purpose: each user likes
a couple of genres more than others, so content-based and collaborative methods
end up producing genuinely different recommendations (otherwise every method
would look the same and the comparison would be meaningless).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]

N_USERS = 610
N_MOVIES = 3000
RNG = np.random.default_rng(42)


def build_movies():
    titles, genres_list = [], []
    for mid in range(1, N_MOVIES + 1):
        k = RNG.integers(1, 4)  # 1-3 genres per movie
        g = RNG.choice(GENRES, size=k, replace=False)
        genres_list.append("|".join(sorted(g)))
        titles.append(f"Movie {mid} ({1950 + int(RNG.integers(0, 75))})")
    return pd.DataFrame({"movieId": range(1, N_MOVIES + 1),
                         "title": titles, "genres": genres_list})


def build_ratings(movies):
    # Per-movie popularity (long tail) and per-movie genre set.
    movie_pop = RNG.power(0.3, size=N_MOVIES)  # skewed: few very popular
    movie_genres = [set(g.split("|")) for g in movies["genres"]]

    rows = []
    for uid in range(1, N_USERS + 1):
        # Each user has 2 preferred genres.
        prefs = set(RNG.choice(GENRES, size=2, replace=False))
        n_rated = int(RNG.integers(20, 120))
        # Sampling weight = popularity boosted if movie matches user's genres.
        match = np.array([1.0 + 2.0 * len(prefs & mg) for mg in movie_genres])
        weights = movie_pop * match
        weights /= weights.sum()
        chosen = RNG.choice(N_MOVIES, size=n_rated, replace=False, p=weights)
        for idx in chosen:
            base = 3.0 + 0.8 * len(prefs & movie_genres[idx])  # like preferred genres more
            r = float(np.clip(RNG.normal(base, 0.8), 0.5, 5.0))
            r = round(r * 2) / 2  # MovieLens uses 0.5 steps
            rows.append((uid, idx + 1, r, int(RNG.integers(9e8, 1.6e9))))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


if __name__ == "__main__":
    movies = build_movies()
    ratings = build_ratings(movies)
    # Keep only movies that were actually rated (mirrors real data).
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())].reset_index(drop=True)
    movies.to_csv(RAW / "movies.csv", index=False)
    ratings.to_csv(RAW / "ratings.csv", index=False)
    print(f"Wrote {len(ratings)} ratings, {ratings.userId.nunique()} users, "
          f"{movies.movieId.nunique()} movies to {RAW}")
