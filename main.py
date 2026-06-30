"""Console pipeline for the recommender prototype.

Run `python main.py` for an end-to-end check: load -> EDA -> split -> fit all
methods -> sample recommendations -> offline evaluation table. The interactive
demo lives in app.py.
"""

import os
import pandas as pd
from src import config
from src.data_loading import (load_ratings, load_items, describe_dataset,
                              train_test_split_ratings)
from src.baselines import (MostPopularRecommender, WeightedAverageRatingRecommender,
                           RandomRecommender)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (ItemItemCollaborativeFiltering,
                                         UserUserCollaborativeFiltering)
from src.matrix_factorization import MatrixFactorization
from src.evaluation import compare_models


def main():
    ratings = load_ratings()
    items = load_items()
    describe_dataset(ratings, items)

    train, test = train_test_split_ratings(ratings, test_size=0.2)
    print(f"\nSplit: {len(train):,} train / {len(test):,} test ratings\n")

    print("Fitting models (item-item CF can take a moment on the full catalogue)...")
    models = {
        "Most Popular": MostPopularRecommender().fit(train),
        "Weighted Average Rating": WeightedAverageRatingRecommender(m=20).fit(train),
        "Content-Based": ContentBasedRecommender().fit(train, items),
        "Item-Item CF": ItemItemCollaborativeFiltering(k=20).fit(train),
        "User-User CF": UserUserCollaborativeFiltering(k=30).fit(train),
        "Matrix Factorization": MatrixFactorization(n_factors=40, n_epochs=20).fit(train),
        "Random": RandomRecommender().fit(train),
    }
    title_of = dict(zip(items["movieId"], items["title"]))

    print("\nSample recommendations")
    for user_id in [1, 2, 3]:
        print(f"\nUser {user_id}")
        for name, model in models.items():
            recs = model.recommend(user_id, train, n=5)
            print(f"  {name:24s}: {[title_of.get(i, i) for i, _ in recs]}")

    print("\nOffline evaluation (top-10)")
    pd.set_option("display.width", 170, "display.max_columns", 20)
    df = compare_models(models, train, test, items, k=10)
    print(df.round(4).to_string())
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/metrics.csv")
    print("\nSaved results/metrics.csv")


if __name__ == "__main__":
    main()
