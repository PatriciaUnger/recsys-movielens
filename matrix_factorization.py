"""Matrix factorization with biases, trained by SGD.

Where neighbourhood CF compares whole rows/columns, MF assumes the ratings
matrix is explained by a few hidden factors. It learns, for every user, a short
vector p_u, and for every item a short vector q_i, so that

    r_hat(u,i) = mu + b_u + b_i + p_u . q_i

  - mu     : global average rating
  - b_u    : how generous/harsh this user is overall
  - b_i    : how loved/disliked this item is overall
  - p_u.q_i: the taste match between the user's factors and the item's factors

We fit it with stochastic gradient descent: go through the observed ratings,
predict, and nudge the parameters down the error gradient (with L2
regularization so the factors stay small and generalize). Recommendations for a
user = score every unseen item with r_hat and take the top N. This is the model
behind the Netflix-Prize approach; we implement it directly so it isn't a black
box.
"""

import numpy as np
from . import config


class MatrixFactorization:
    def __init__(self, n_factors=40, lr=0.01, reg=0.05, n_epochs=25,
                 random_state=config.RANDOM_STATE):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, ratings, items=None, verbose=False):
        users = np.sort(ratings[config.USER_COL].unique())
        item_ids = np.sort(ratings[config.ITEM_COL].unique())
        self.u_idx_ = {u: i for i, u in enumerate(users)}
        self.i_idx_ = {it: j for j, it in enumerate(item_ids)}
        self.users_, self.items_ = users, item_ids
        n_users, n_items = len(users), len(item_ids)

        u = ratings[config.USER_COL].map(self.u_idx_).to_numpy()
        i = ratings[config.ITEM_COL].map(self.i_idx_).to_numpy()
        r = ratings[config.RATING_COL].to_numpy(dtype=float)

        rng = np.random.default_rng(self.random_state)
        self.mu_ = r.mean()
        self.b_u_ = np.zeros(n_users)
        self.b_i_ = np.zeros(n_items)
        self.P_ = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_items, self.n_factors))

        # track which items each user has seen (for excluding at recommend time)
        self.seen_ = {}
        for uu, ii in zip(u, i):
            self.seen_.setdefault(uu, set()).add(ii)

        order = np.arange(len(r))
        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            sse = 0.0
            for n in order:
                uu, ii, rui = u[n], i[n], r[n]
                pred = self.mu_ + self.b_u_[uu] + self.b_i_[ii] + self.P_[uu] @ self.Q_[ii]
                err = rui - pred
                sse += err * err
                # gradient step
                self.b_u_[uu] += self.lr * (err - self.reg * self.b_u_[uu])
                self.b_i_[ii] += self.lr * (err - self.reg * self.b_i_[ii])
                pu, qi = self.P_[uu].copy(), self.Q_[ii]
                self.P_[uu] += self.lr * (err * qi - self.reg * pu)
                self.Q_[ii] += self.lr * (err * pu - self.reg * qi)
            if verbose:
                print(f"  epoch {epoch+1:2d}  train RMSE = {np.sqrt(sse/len(r)):.4f}")
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.u_idx_ or item_id not in self.i_idx_:
            return self.mu_
        uu, ii = self.u_idx_[user_id], self.i_idx_[item_id]
        return self.mu_ + self.b_u_[uu] + self.b_i_[ii] + self.P_[uu] @ self.Q_[ii]

    def recommend(self, user_id, ratings_train, n=config.TOP_K, exclude_seen=True):
        if user_id not in self.u_idx_:
            return []
        uu = self.u_idx_[user_id]
        scores = self.mu_ + self.b_u_[uu] + self.b_i_ + self.Q_ @ self.P_[uu]
        if exclude_seen:
            for ii in self.seen_.get(uu, ()):
                scores[ii] = -np.inf
        order = np.argsort(-scores)[:n]
        return [(int(self.items_[j]), float(scores[j])) for j in order if np.isfinite(scores[j])]
