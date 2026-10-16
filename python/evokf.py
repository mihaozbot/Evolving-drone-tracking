import numpy as np


# ---- swarms: local linear models, merge / drop to keep budget ----
class Swarms:
    def __init__(self, budget=50, forget=2000, ridge=1e-2, dim=3, classes=0):
        self.budget, self.forget, self.ridge, self.dim = budget, forget, ridge, dim
        self.L = np.zeros((0, classes))
        self.miss = np.zeros(0)
        self.mu, self.n, self.last = np.zeros((0, dim)), np.zeros(0), np.zeros(0)
        self.C = np.zeros((0, dim, dim))
        self.A, self.B = np.zeros((0, dim + 1, dim + 1)), np.zeros((0, dim + 1, dim))
        self.k, self.xm, self.xM2 = 0, np.zeros(dim), np.zeros(dim)
        self.lo, self.hi = np.full(dim, np.inf), np.full(dim, -np.inf)
        self.cn, self.cm, self.cM2 = np.zeros(classes), np.zeros((classes, dim)), np.zeros((classes, dim))

    def special(self):
        S = np.ones((len(self.cn), self.dim))
        seen = self.cn > 1
        if seen.sum() < 2:
            return S
        for c in np.flatnonzero(seen):
            others = self.cm[seen & (np.arange(len(self.cn)) != c)]
            S[c] = np.median((others - self.cm[c]) ** 2, 0) / (self.cM2[c] / self.cn[c] + 1e-12)
            S[c] = S[c] / S[c].mean() + 1e-3
        return S

    def _w(self):
        return self.special()[np.argmax(self.L, 1)] if self.L.shape[1] else np.ones_like(self.mu)

    def _scale(self):
        s = np.sqrt(self.xM2 / self.k) if self.k > 1 else np.ones(self.dim)
        return np.maximum(s, 1e-9)

    def _d2(self, x):
        return (((x - self.mu) / self._scale()) ** 2 * self._w()).sum(1)

    def learn(self, x, y, label=None):
        self.k += 1
        y = y - x  # learn the change
        self.lo, self.hi = np.minimum(self.lo, y), np.maximum(self.hi, y)
        d = x - self.xm
        self.xm += d / self.k
        self.xM2 += d * (x - self.xm)
        if len(self.mu):
            d2 = self._d2(x)
            self.last[np.argmin(d2)] = self.k
        if label is not None:
            self.cn[label] += 1
            d = x - self.cm[label]
            self.cm[label] += d / self.cn[label]
            self.cM2[label] += d * (x - self.cm[label])
            if len(self.mu) and np.argmax(self.L[np.argmin(d2)]) != label:
                self.miss[np.argmin(d2)] += 1
        phi = np.r_[1.0, x]
        self.mu, self.n, self.last = np.vstack([self.mu, x]), np.append(self.n, 1.0), np.append(self.last, self.k)
        self.C = np.concatenate([self.C, np.zeros((1, self.dim, self.dim))])
        self.A = np.concatenate([self.A, (self.ridge * np.eye(self.dim + 1) + np.outer(phi, phi))[None]])
        self.B = np.concatenate([self.B, np.outer(phi, y)[None]])
        self.L = np.vstack([self.L, np.eye(self.L.shape[1])[label] if label is not None else np.zeros(self.L.shape[1])])
        self.miss = np.append(self.miss, 0.0)
        while len(self.n) > self.budget:
            self.keep_budget()

    # drop stale one or merge 2 closest
    def keep_budget(self):
        X = self.mu / self._scale()
        D = ((X[:, None] - X[None]) ** 2 * self._w()[:, None]).sum(-1) + np.diag(np.full(len(X), np.inf))
        if self.L.shape[1]:  # only same class merges
            c = np.argmax(self.L, 1)
            D[c[:, None] != c[None]] = np.inf
            if not np.isfinite(D).any():
                self._drop(np.argmin(self.n))
                return
        nearest = D.min(1)
        stale = np.flatnonzero((self.k - self.last > self.forget) & (nearest < np.median(nearest)))
        if len(stale):
            self._drop(stale[np.argmin(self.last[stale])])
            return
        a, b = np.unravel_index(np.argmin(D), D.shape)
        keep, drop = (a, b) if self.n[a] >= self.n[b] else (b, a)
        w = self.n[keep] * self.n[drop] / (self.n[keep] + self.n[drop])
        d = self.mu[keep] - self.mu[drop]
        self.C[keep] += self.C[drop] + w * np.outer(d, d)
        self.mu[keep] = (self.n[keep] * self.mu[keep] + self.n[drop] * self.mu[drop]) / (self.n[keep] + self.n[drop])
        self.n[keep] += self.n[drop]
        self.last[keep] = max(self.last[keep], self.last[drop])
        self.A[keep] += self.A[drop] - self.ridge * np.eye(self.dim + 1)
        self.B[keep] += self.B[drop]
        self.L[keep] += self.L[drop]
        self.miss[keep] += self.miss[drop]
        self._drop(drop)

    def _drop(self, i):
        keep = np.arange(len(self.n)) != i
        self.mu, self.n, self.last, self.C = self.mu[keep], self.n[keep], self.last[keep], self.C[keep]
        self.A, self.B, self.L, self.miss = self.A[keep], self.B[keep], self.L[keep], self.miss[keep]

    def cov(self):
        return self.C / np.maximum(self.n, 1)[:, None, None]

    def classify(self, x, n_near=8):
        d2 = self._d2(x)
        o = np.argsort(d2)[:n_near]
        w = np.exp(-(d2[o] - d2[o[0]]) / (np.median(d2[o]) - d2[o[0]] + 1e-9))
        trust = self.n[o] / (self.n[o] + self.miss[o])  # often wrong -> trust less
        return ((w * trust) @ self.L[o]) / np.maximum(self.L.sum(0), 1) + 1e-6

    def motion(self, x, n_near=8):
        d2 = self._d2(x)
        o = np.argsort(d2)[:n_near]
        w = np.exp(-(d2[o] - d2[o[0]]) / (np.median(d2[o]) - d2[o[0]] + 1e-9))
        theta = np.linalg.solve(self.ridge * np.eye(self.dim + 1) + np.einsum("c,cij->ij", w, self.A[o]),
                                np.einsum("c,cij->ij", w, self.B[o]))
        return theta[1:].T, theta[0]

    def next(self, x):
        A, b = self.motion(x)
        margin = 0.5 * (self.hi - self.lo)
        return x + np.clip(A @ x + b, self.lo - margin, self.hi + margin)


# ---- kalman on (p, v), swarms learn how v changes ----
class EvoKF(Swarms):

    def __init__(self, dt, q=10.0, r=0.1 ** 2, **swarms):
        super().__init__(**swarms)
        self.dt, self.r = dt, r
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])

    def start(self, z0):
        self.x = np.vstack([z0, np.zeros(3)])
        self.P = np.repeat(np.diag([self.r, 10.0])[:, :, None], 3, axis=2)
        self.steps, self.v_prev, self.surprise = 0, None, 1.0

    def step(self, z, learn=True):
        self.x = self.F @ self.x
        self.P = np.einsum("ij,jka,lk->ila", self.F, self.P, self.F) + self.Q[:, :, None]
        S = self.P[0, 0] + self.r
        self.surprise = 0.95 * self.surprise + 0.05 * ((z - self.x[0]) ** 2 / S).mean()
        K = self.P[:, 0] / S
        self.x = self.x + K * (z - self.x[0])
        self.P = self.P - K[:, None] * self.P[0][None]
        self.steps += 1
        if learn and self.steps > 5:
            if self.v_prev is not None:
                self.learn(self.v_prev, self.x[1])
            self.v_prev = self.x[1].copy()

    def coast(self):  # no measurement -> learned model
        if len(self.mu):
            self.x[1] = self.next(self.x[1])
        self.x = self.F @ self.x
        self.P = np.einsum("ij,jka,lk->ila", self.F, self.P, self.F) + self.Q[:, :, None]
        self.v_prev = None

    def predict(self, H, learned=True):
        p, v, path = self.x[0], self.x[1], []
        for _ in range(H):
            if learned and len(self.mu):
                v = self.next(v)
            p = p + self.dt * v
            path.append(p)
        return np.array(path)
