# Movie Recommender System (MovieLens)

Individual project for the Recommender Systems course. Implements six
recommendation methods plus a random baseline, an offline evaluation module,
and an interactive Streamlit prototype.

## Methods
Most Popular, Weighted Average (IMDb-style), Content-Based (TF-IDF on genres),
Item-Item CF (adjusted cosine), User-User CF (Pearson), Matrix Factorization
(SGD with biases), and a Random baseline.

## Run locally
```bash
pip install -r requirements.txt
python main.py          # console pipeline + evaluation table
streamlit run app.py    # interactive prototype
```

## Data
This project uses MovieLens **ml-latest-small** from GroupLens. The data is **not**
included in the repository (the MovieLens licence does not allow redistribution).
- For local use, download it from https://grouplens.org/datasets/movielens/ and
  place `ratings.csv` and `movies.csv` in `data/raw/`.
- When deployed (e.g. Streamlit Community Cloud), the app downloads the dataset
  automatically on first run.

## Structure
`src/` source modules · `app.py` Streamlit app · `main.py` console pipeline ·
`notebooks/eda.ipynb` exploratory data analysis · `requirements.txt` dependencies.
