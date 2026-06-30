"""Recommender prototype - interactive UI.

Pick a USER and a METHOD; see the top-N recommendations that method produces
for that user, next to the user's own profile for context. Every model shares
the same .recommend(user_id, train) interface, so the method dropdown only
swaps which object we call -- that's what makes the comparison fair and the
screen easy to extend with new methods.

Run with:  streamlit run app.py
"""

import html
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings
from src.baselines import (MostPopularRecommender, WeightedAverageRatingRecommender,
                           RandomRecommender)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (ItemItemCollaborativeFiltering,
                                         UserUserCollaborativeFiltering)
from src.matrix_factorization import MatrixFactorization
from src.evaluation import compare_models

st.set_page_config(page_title="Movie Recommender Prototype",
                   page_icon="🎬", layout="wide")

# What the score column means for each method -- shown above the list so the
# numbers are never mysterious.
METHOD_NOTES = {
    "Most Popular": "Score = how many people rated the film. Popular, but the same for everyone.",
    "Weighted Average Rating": "Score = average rating pulled toward the global mean, so a single 5-star vote can't win.",
    "Content-Based": "Score = how close a film's genres are to this viewer's taste profile (cosine similarity, 0–1).",
    "Item-Item CF": "Predicted rating from films similar to ones the viewer liked (adjusted cosine). Content-agnostic; safer picks.",
    "User-User CF": "Predicted rating from viewers with similar taste (Pearson). Good at discovery; can exceed 5, which is fine.",
    "Matrix Factorization": "Predicted rating from learned hidden factors (mu + b_u + b_i + p_u·q_i), trained by SGD.",
    "Random": "No score — random unseen films, kept only as a sanity-check baseline.",
}


# ---------------------------------------------------------------- styling ----
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root{
      --bg:#F6F7FB; --card:#FFFFFF; --ink:#15171E; --muted:#6B7180;
      --line:#E8EAF1; --accent:#6366F1; --accent-2:#8B5CF6; --accent-soft:#EEF0FF;
      --chip:#F1F2F8; --shadow:0 1px 2px rgba(16,24,40,.04),0 4px 12px rgba(16,24,40,.06);
    }
    .stApp{ background:var(--bg); }
    html, body, [class*="css"]{ font-family:'Inter',sans-serif; color:var(--ink); }
    #MainMenu, footer, header{ visibility:hidden; }
    .block-container{ padding-top:2rem; max-width:1280px; }

    section[data-testid="stSidebar"]{ background:var(--card); border-right:1px solid var(--line); }
    section[data-testid="stSidebar"] label{
      font-weight:600; font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); }

    .masthead{ margin-bottom:1.8rem; }
    .masthead h1{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:2.4rem;
      line-height:1.05; margin:0; letter-spacing:-.02em; }
    .masthead p{ color:var(--muted); margin:.4rem 0 0; font-size:.98rem; }

    .eyebrow{ font-weight:700; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;
      color:var(--accent); margin-bottom:.55rem; }
    .sec-title{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.4rem; margin:0 0 .35rem; }

    .profile-card{ background:var(--card); border:1px solid var(--line); border-radius:18px;
      padding:1.25rem 1.35rem; box-shadow:var(--shadow); }
    .metric-row{ display:flex; gap:.75rem; margin:.1rem 0 1.25rem; }
    .metric{ flex:1; background:var(--accent-soft); border-radius:14px; padding:.8rem .9rem; }
    .metric .num{ font-family:'Plus Jakarta Sans',sans-serif; font-size:1.7rem; font-weight:700;
      line-height:1; color:var(--accent); }
    .metric .lab{ font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-top:.35rem; }
    .field-lab{ font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin-bottom:.45rem; }
    .liked{ display:flex; justify-content:space-between; align-items:center;
      padding:.5rem 0; border-bottom:1px solid var(--line); font-size:.92rem; }
    .liked:last-child{ border-bottom:none; }
    .star{ color:var(--accent); font-weight:600; white-space:nowrap; margin-left:.6rem; }

    .chip{ display:inline-block; background:var(--chip); color:#4A4F63; border-radius:999px;
      padding:.2rem .65rem; font-size:.74rem; font-weight:500; margin:.18rem .3rem .18rem 0; }

    .rec{ display:flex; align-items:center; gap:1rem; background:var(--card);
      border:1px solid var(--line); border-radius:16px; padding:.95rem 1.1rem; margin-bottom:.7rem;
      box-shadow:var(--shadow); transition:transform .12s ease, box-shadow .12s ease; }
    .rec:hover{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(99,102,241,.14); }
    .rank{ display:flex; align-items:center; justify-content:center; width:2.1rem; height:2.1rem;
      flex:none; border-radius:50%; background:var(--accent-soft); color:var(--accent);
      font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:.95rem; }
    .rec-body{ flex:1; min-width:0; }
    .rec-title{ font-weight:600; font-size:1rem; margin-bottom:.3rem; }
    .scorewrap{ display:flex; align-items:center; gap:.6rem; margin-top:.55rem; }
    .scorebar{ flex:1; height:6px; background:var(--accent-soft); border-radius:999px; overflow:hidden; }
    .scorebar > span{ display:block; height:100%; border-radius:999px;
      background:linear-gradient(90deg,var(--accent),var(--accent-2)); }
    .scoreval{ font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
    .why{ font-size:.8rem; color:var(--muted); margin-top:.5rem; }
    .why b{ color:var(--accent); font-weight:600; }
    .note{ color:var(--muted); font-size:.9rem; margin:0 0 1.2rem; }
    </style>
    """, unsafe_allow_html=True)


def stars(rating):
    full = int(rating)
    half = "½" if rating - full >= 0.5 else ""
    return "★" * full + half


def chips(genre_str):
    parts = [g for g in str(genre_str).split("|") if g and g != "nan"]
    return "".join(f"<span class='chip'>{html.escape(g)}</span>" for g in parts)


# ------------------------------------------------------------- data + models --
@st.cache_data
def load_data():
    ratings = load_ratings()
    items = load_items()
    train, test = train_test_split_ratings(ratings)
    return ratings, items, train, test


@st.cache_resource
def build_models(_train, _items):
    return {
        "Most Popular": MostPopularRecommender().fit(_train),
        "Weighted Average Rating": WeightedAverageRatingRecommender(m=20).fit(_train),
        "Content-Based": ContentBasedRecommender().fit(_train, _items),
        "Item-Item CF": ItemItemCollaborativeFiltering(k=20).fit(_train),
        "User-User CF": UserUserCollaborativeFiltering(k=30).fit(_train),
        "Matrix Factorization": MatrixFactorization(n_factors=40, n_epochs=20).fit(_train),
        "Random": RandomRecommender().fit(_train),
    }


@st.cache_data
def evaluate(_models, _train, _test, _items):
    return compare_models(_models, _train, _test, _items, k=config.TOP_K)


inject_css()
ratings, items, train, test = load_data()
models = build_models(train, items)
title_of = dict(zip(items[config.ITEM_COL], items[config.TITLE_COL]))
genres_of = dict(zip(items[config.ITEM_COL], items.get(config.GENRES_COL, pd.Series())))


def user_top_genres(uid, n=5):
    """The genres this viewer rated most often above 4 stars."""
    rows = train[(train[config.USER_COL] == uid) & (train[config.RATING_COL] >= 4)]
    gc = {}
    for mid in rows[config.ITEM_COL]:
        for g in str(genres_of.get(mid, "")).split("|"):
            if g and g != "nan":
                gc[g] = gc.get(g, 0) + 1
    return sorted(gc, key=lambda g: -gc[g])[:n]


def closest_liked_film(uid, item_id):
    """For item-item CF: the viewer's own film most similar to a recommendation."""
    m = models["Item-Item CF"]
    if uid not in m.u_idx_ or item_id not in m.i_idx_:
        return None
    u, i = m.u_idx_[uid], m.i_idx_[item_id]
    sims = m.S_[i].copy()
    sims[~m.observed_[u]] = -1.0          # keep only films this viewer has rated
    j = int(np.argmax(sims))
    return title_of.get(int(m.items_[j])) if sims[j] > 0 else None


def why(method, uid, item_id, score):
    """A short, honest reason this item was recommended, per method."""
    g = [x for x in str(genres_of.get(item_id, "")).split("|") if x and x != "nan"]
    if method == "Most Popular":
        return f"Popular pick, rated by <b>{int(score)}</b> viewers"
    if method == "Weighted Average Rating":
        return f"Consistently well rated (<b>{score:.2f}</b>/5 weighted)"
    if method == "Content-Based":
        overlap = [x for x in user_top_genres(uid) if x in g]
        if overlap:
            return "Matches your taste for <b>" + html.escape(", ".join(overlap[:2])) + "</b>"
        return "Close to your genre profile"
    if method == "Item-Item CF":
        t = closest_liked_film(uid, item_id)
        return ("Because you liked <b>" + html.escape(str(t)) + "</b>") if t \
            else "Similar to films you rated highly"
    if method == "User-User CF":
        return "Liked by viewers with taste like yours"
    if method == "Matrix Factorization":
        return f"Predicted <b>{max(0.5, min(score, 5)):.1f}\u2605</b> for your taste"
    if method == "Random":
        return "Random pick (sanity-check baseline)"
    return ""

# ----- controls -----
st.sidebar.markdown("### Controls")
user_id = st.sidebar.selectbox("User", sorted(train[config.USER_COL].unique()))
method = st.sidebar.selectbox("Method", list(models.keys()))
top_n = st.sidebar.slider("Recommendations to show", 5, 20, config.TOP_K)

# ----- masthead -----
st.markdown("<div class='masthead'><h1>Movie Recommender</h1>"
            "<p>Choose a viewer and a method to see how the recommendations change.</p></div>",
            unsafe_allow_html=True)

tab_explore, tab_eval = st.tabs(["Explore recommendations", "Compare methods"])

with tab_explore:
    left, right = st.columns([5, 7], gap="large")

# ----- profile -----
with left:
    st.markdown(f"<div class='eyebrow'>Viewer {user_id}</div>", unsafe_allow_html=True)
    rows = train[train[config.USER_COL] == user_id]

    liked = rows[rows[config.RATING_COL] >= 4]
    gc = {}
    for mid in liked[config.ITEM_COL]:
        for g in str(genres_of.get(mid, "")).split("|"):
            if g and g != "nan":
                gc[g] = gc.get(g, 0) + 1
    fav = sorted(gc, key=lambda g: -gc[g])[:5]
    fav_html = "".join(f"<span class='chip'>{html.escape(g)}</span>" for g in fav) or \
               "<span class='note'>no clear favourites yet</span>"

    top = rows.sort_values(config.RATING_COL, ascending=False).head(7)
    liked_html = "".join(
        f"<div class='liked'><span>{html.escape(str(title_of.get(m, m)))}</span>"
        f"<span class='star'>{stars(r)} {r:g}</span></div>"
        for m, r in zip(top[config.ITEM_COL], top[config.RATING_COL]))

    st.markdown(
        "<div class='profile-card'>"
        "<div class='metric-row'>"
        f"<div class='metric'><div class='num'>{len(rows)}</div><div class='lab'>films rated</div></div>"
        f"<div class='metric'><div class='num'>{rows[config.RATING_COL].mean():.2f}</div><div class='lab'>avg rating</div></div>"
        "</div>"
        "<div class='field-lab'>Favourite genres</div>"
        f"<div style='margin-bottom:1rem'>{fav_html}</div>"
        "<div class='field-lab'>Top rated</div>"
        f"{liked_html}"
        "</div>", unsafe_allow_html=True)

# ----- recommendations -----
with right:
    st.markdown(f"<div class='eyebrow'>Recommendations</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title'>{html.escape(method)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='note'>{METHOD_NOTES.get(method, '')}</div>", unsafe_allow_html=True)

    recs = models[method].recommend(user_id, train, n=top_n, exclude_seen=True)
    scores = [s for _, s in recs]
    lo, hi = (min(scores), max(scores)) if scores else (0, 1)
    span = (hi - lo) or 1

    def fmt(s):
        return f"{s:.0f}" if float(s).is_integer() else f"{s:.3f}"

    if not recs:
        st.markdown("<div class='note'>This method can't produce a list for this "
                    "viewer (no usable history yet). Try another viewer or method.</div>",
                    unsafe_allow_html=True)

    cards = []
    for i, (mid, score) in enumerate(recs, 1):
        width = 100 * (score - lo) / span if hi != lo else 60
        cards.append(
            f"<div class='rec'><div class='rank'>{i}</div>"
            f"<div class='rec-body'><div class='rec-title'>{html.escape(str(title_of.get(mid, mid)))}</div>"
            f"<div>{chips(genres_of.get(mid, ''))}</div>"
            f"<div class='scorewrap'><div class='scorebar'><span style='width:{width:.0f}%'></span></div>"
            f"<div class='scoreval'>{fmt(score)}</div></div>"
            f"<div class='why'>{why(method, user_id, mid, score)}</div></div></div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


# ----- evaluation tab -----
with tab_eval:
    st.markdown("<div class='eyebrow'>Offline evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>How the methods compare</div>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Top-10 lists scored against each viewer's held-out, "
                "highly-rated films. Higher is better for everything except <b>PopBias</b> "
                "(lower = less herd behaviour) — accuracy alone never tells the whole story.</div>",
                unsafe_allow_html=True)

    metrics = evaluate(models, train, test, items)
    show = metrics.drop(columns=["users_evaluated"])
    st.dataframe(show.style.format("{:.4f}"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='field-lab'>Precision@10 — accuracy</div>", unsafe_allow_html=True)
        st.bar_chart(show[f"Precision@{config.TOP_K}"], color="#6366F1")
    with c2:
        st.markdown("<div class='field-lab'>Coverage — share of catalog recommended</div>", unsafe_allow_html=True)
        st.bar_chart(show["Coverage"], color="#8B5CF6")

    st.markdown("<div class='note' style='margin-top:1rem'>Read it as a trade-off, not a "
                "ranking: Most Popular leans on a few hits (high PopBias, tiny Coverage); "
                "Content-Based is accurate and covers the catalogue but stays inside one genre "
                "(low Diversity); Random is the floor that shows accuracy actually matters.</div>",
                unsafe_allow_html=True)
