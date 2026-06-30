"""Configuration file for paths and project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# MovieLens-style default paths. Students may change these if using another dataset.
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
ITEMS_PATH = RAW_DATA_DIR / "movies.csv"

# Default recommendation settings.
USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"
TITLE_COL = "title"
GENRES_COL = "genres"

TOP_K = 10
RANDOM_STATE = 42
