# demo: 2 drones at a time, sensor loses both once, then who is who
import copy
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import download_mmaud
from evokf import EvoKF, Swarms

HERE = Path(__file__).resolve().parent
SEMINAR = HERE.parent / "Seminarska"

DT, H, WINDOW = 0.2, 20, 50
GAP = 20
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc4"]
FEATURES = ["hitrost", "najvišja hitrost", "pospeševanje", "zaviranje", "bočni pospešek",
            "največji bočni pospešek", "ostrina zavoja", "sunkovitost", "dviganje", "lebdenje"]
FX, FY = 3, 5


# ---- helpers ----
def special(clf, c, top=3):
    if clf.cn[c] < 2:
        return ""
    S, others = clf.special()[c], np.delete(clf.cm, c, axis=0)[np.delete(clf.cn, c) > 1]
    return ", ".join(f"{'↑' if clf.cm[c, f] > np.median(others[:, f]) else '↓'} {FEATURES[f]}"
                     for f in np.argsort(-S)[:top])


def who_roji(clf):
    return (clf.mu[:, [FX, FY]].copy(), clf.cov()[:, [FX, FY]][:, :, [FX, FY]], clf.n.copy(), clf.L.copy(),
            [special(clf, c) for c in range(len(clf.cn))])


def how_roji(track):
    go = []
    for r in range(len(track.n)):
        theta = np.linalg.solve(track.A[r], track.B[r])
        go.append(track.mu[r, :2] + (theta[1:].T @ track.mu[r] + theta[0])[:2] / DT)
    return track.mu[:, :2].copy(), track.cov()[:, :2, :2], track.n.copy(), np.array(go).reshape(-1, 2)


def features(p):
    v = np.diff(p, axis=0) / DT
    a = np.diff(v, axis=0) / DT
    v = v[1:]
    s = np.hypot(v[:, 0], v[:, 1])
    along = (v[:, 0] * a[:, 0] + v[:, 1] * a[:, 1]) / np.maximum(s, 1e-6)
    side = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.maximum(s, 1e-6)
    jerk = np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1) / DT
    mv = s > 0.5 if (s > 0.5).sum() > 2 else np.ones(len(s), bool)
    return np.array([s.mean(), np.percentile(s, 90), along[mv].clip(0).mean(), (-along[mv]).clip(0).mean(),
                     side[mv].mean(), np.percentile(side[mv], 90),
                     np.percentile(side[mv] / np.maximum(s[mv], 0.5) ** 2, 90), np.median(jerk),
                     np.percentile(np.abs(v[:, 2]), 90), 1 - (s > 0.5).mean()])


def miss(model, p):
    v, m = (p[5:] - p[:-5]) / (5 * DT), 0.0
    for a, b in zip(v[:-5], v[5:]):
        for _ in range(5):
            a = model.next(a) if len(model.mu) else a
        m += ((a - b) ** 2).sum()
    return m


def sure(L, right):
    c = np.clip(right[0] / right[1], 0.5, 0.95)
    p = np.exp(L - L.max(1, keepdims=True))
    return np.log((2 * c - 1) * p / p.sum(1, keepdims=True) + 1 - c)


def name(L):
    return L[0, 1] + L[1, 0] > L[0, 0] + L[1, 1]


def turn(i):
    c, s = np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# ---- main loop ----
def main(seconds=30):
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    names = list(download_mmaud.DRONES)
    flights = [download_mmaud.load(d, DT) for d in names]
    centre = np.vstack(flights).mean(0)
    for i, f in enumerate(flights):
        flights[i] = (f - centre) @ turn(i) + centre
    allp = np.vstack(flights)
    lo, hi = allp.min(0), allp.max(0)
    z = [f + 0.05 * rng.standard_normal(f.shape) for f in flights]  # fake sensor noise
    track = [EvoKF(DT, q=0.1, budget=6) for _ in names]
    for e in range(len(names)):
        track[e].start(z[e][0])
    clf = Swarms(budget=12, dim=len(FEATURES), classes=len(names))
    pairs = [list(c) for c in itertools.combinations(range(len(names)), 2)]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    at = [1] * len(names)
    later, log, ids, frames, clock = [], [], [], [], 0
    windows, flown = [], [[] for _ in names]
    record = {"vedenje": [1, 2], "gibanje": [1, 2]}
    for pair in itertools.cycle(pairs):
        pair = [e for e in pair if at[e] + int(seconds / DT) + H < len(flights[e])]
        if sum(at[e] + int(seconds / DT) + H < len(flights[e]) for e in range(len(names))) < 2:
            break
        if len(pair) < 2:
            continue
        L0 = max(range(30, 85, 5), key=lambda j: sum(np.linalg.norm(flights[e][at[e] + j + 1, :2] - flights[e][at[e] + j, :2])
                                                     for e in pair))
        L1 = L0 + GAP  # lose them where they fly fastest
        who, seen, Lp = list(pair), [0, 0], np.zeros((2, 2))
        for j in range(int(seconds / DT)):
            clock += 1
            lost = L0 <= j < L1
            if j == L0:
                known = [copy.deepcopy(track[e]) for e in pair]  # models at the moment of loss
            if j == L1:  # back, no names -> compare positions
                for i, d in enumerate(pair):
                    seen[i] = at[d]
                    for m, e in enumerate(pair):
                        S = (track[e].P[0, 0] + track[e].r) * track[e].surprise
                        Lp[i, m] = -0.5 * ((z[d][at[d]] - track[e].x[0]) ** 2 / S + np.log(S)).sum()
            if j >= L1 and j % 5 == 0:
                Lb, Lm = np.zeros((2, 2)), np.zeros((2, 2))
                fly = at[pair[0]] - seen[0] >= 15 and len(clf.n)
                if fly:
                    for i, d in enumerate(pair):
                        p = z[d][max(seen[i], at[d] - WINDOW):at[d] + 1]
                        Lb[i] = np.log(clf.classify(features(p))[pair])
                        Lm[i] = [miss(m, p) for m in known]
                    Lb, Lm = sure(Lb, record["vedenje"]), sure(-0.5 * Lm / (Lm.min(1, keepdims=True) / (len(p) - 10)),
                                                              record["gibanje"])
                    for key, L in (("vedenje", Lb), ("gibanje", Lm)):
                        later.append((clock + WINDOW, key, not name(L)))
                every = name(Lp + Lb + Lm)
                who = pair[::-1] if every else list(pair)
                ids.append((clock * DT, not every, not name(Lp),
                            not name(Lb) if fly else np.nan, not name(Lm) if fly else np.nan))
            for i, d in enumerate(pair):
                if lost:
                    track[pair[i]].coast()
                else:
                    track[who[i]].step(z[d][at[d]])
                    flown[who[i]].append(track[who[i]].x[1, :2].copy())
            while later and later[0][0] <= clock:
                _, f, e = later.pop(0)
                if isinstance(f, str):
                    record[f][0] += e; record[f][1] += 1
                else:
                    clf.learn(f, f, e)
                    windows.append((f[FX], f[FY], e))
            if j % 5 == 0:
                state = []
                for i, e in enumerate(pair):
                    k = at[e]
                    ours, straight = track[e].predict(H), track[e].predict(H, learned=False)
                    if k >= WINDOW and (j <= L0 or j >= L1 + WINDOW):
                        later.append((clock + WINDOW, features(z[e][k - WINDOW:k + 1]), e))
                    log.append((e, np.linalg.norm(ours[-1] - flights[e][k + H]),
                                np.linalg.norm(straight[-1] - flights[e][k + H])))
                    state.append((e, k, lost, None if lost else who[i], track[e].x[0].copy(), ours))
                frames.append((clock, state, who_roji(clf), np.array(windows[-300:]).reshape(-1, 3),
                               {e: (how_roji(track[e]), np.array(flown[e][-150:]).reshape(-1, 2)) for e in pair}))
            for e in pair:
                at[e] += 1
    log, ids = np.array(log, dtype=float), np.array(ids, dtype=float)
    print("after the radar lost them, right name (2 in the air, chance 50 %):")
    for col, what in ((2, "position only"), (3, "behaviour only"), (4, "travel model only"), (1, "all together")):
        print(f"  {what:21s} {100 * np.nanmean(ids[:, col]):3.0f} %")
    for e, n in enumerate(names):
        m = log[:, 0] == e
        print(f"  {n:9s} miss 4 s ahead: EvoKF {log[m, 1].mean():5.2f} m, constant velocity {log[m, 2].mean():5.2f} m")

    secs = np.unique(ids[:, 0])
    acc = np.array([[100 * np.nanmean(ids[(ids[:, 0] > s - 60) & (ids[:, 0] <= s), c]) for c in (1, 2, 3, 4)]
                    for s in secs])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    accuracy(ax, secs, acc)
    fig.tight_layout()
    fig.savefig(HERE / "behaviour.png", dpi=150, bbox_inches="tight")
    per = int(seconds / DT) // 5
    show = [names.index("mavic2"), names.index("avata")]
    seminar(frames, flights, names, show, per, clf, track, windows, flown)
    animate(frames, flights, names, secs, acc, lo, hi)


# ---- plots ----
def accuracy(ax, secs, acc, upto=np.inf):
    m = secs <= upto
    lines = [ax.plot(secs[m], acc[m, c], color=col, lw=lw, ls=ls, label=lab)[0] for c, col, lw, ls, lab in
             ((0, COLORS[0], 2, "-", "vse skupaj"), (1, "grey", 1.2, "--", "samo napovedan položaj"),
              (2, COLORS[1], 1.2, ":", "samo vedenje"), (3, COLORS[2], 1.2, "-.", "samo model gibanja"))]
    ax.axhline(50, color="grey", ls=":", lw=1)
    ax.set_xlim(0, secs[-1]); ax.set_ylim(0, 105)
    ax.set_xlabel("čas [s]"); ax.set_ylabel("pravo ime po izgubi, zadnjih 60 s [%]")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return lines


def draw(a, cl, rg, frame, flights, names, lims, notes=True, trail=25):
    clock, state, (mu, S, n, L, why), pts, reg = frame
    lo, hi, wx, wy = lims
    t = np.linspace(0, 2 * np.pi, 40)
    for ax in (a, cl, rg):
        ax.cla()
    a.set_xlim(lo[0] - 2, hi[0] + 2); a.set_ylim(lo[1] - 2, hi[1] + 2); a.set_zlim(0, hi[2] + 2)
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("višina [m]")
    a.view_init(elev=22, azim=-60)
    for e, k, lost, guess, est, ours in state:
        q = flights[e]
        a.plot(*q[max(0, k - trail):k + 1].T, color=COLORS[e], lw=2, label=names[e])
        a.plot(*np.vstack([est, ours]).T, color=COLORS[e], lw=1.5, ls="--")
        a.plot([q[k, 0]], [q[k, 1]], [q[k, 2]], "o", ms=8, color=COLORS[e], mfc="none" if lost else COLORS[e])
        if guess is not None:
            a.text(*(q[k] + [0.4, 0.4, 0.6]), names[guess], fontsize=8, fontweight="bold",
                   color="#1a7f37" if guess == e else "#c62828")
    a.text2D(0.02, 0.95, f"t = {clock * DT:.0f} s" + ("   radar ju ne vidi" if state[0][2] else ""),
             transform=a.transAxes, fontsize=10, fontweight="bold")
    a.legend(frameon=False, fontsize=8, loc="upper right",
             title="- - napoved 4 s; prazno = izgubljen;\nime: zeleno = prav, rdeče = narobe" if notes else None)
    cl.set_xlim(*wx); cl.set_ylim(*wy)
    cl.set_xlabel(f"{FEATURES[FX]} [m/s²]"); cl.set_ylabel(f"{FEATURES[FY]} [m/s²]")
    if len(pts):
        cl.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.35, lw=0, color=[COLORS[int(e)] for e in pts[:, 2]])
    for r in range(len(n)):
        c = int(np.argmax(L[r]))
        roj(cl, mu[r], S[r], COLORS[c])
    if notes:
        cl.text(0.02, 0.02, "razvrščevalnik: vsak roj je od enega; pike = 10 s okna, iz katerih se je učil",
                transform=cl.transAxes, va="bottom", fontsize=9)
    cl.legend(handles=[Line2D([], [], color=COLORS[e], lw=2, label=f"{nm}: {why[e]}" if why[e] else nm)
                       for e, nm in enumerate(names)], frameon=False, fontsize=8, loc="upper right",
              title="posebnost (↑ več / ↓ manj kot drugi)" if notes else None)
    big = 1.0
    for e, ((mu, S, n, go), flown) in reg.items():
        if len(flown):
            rg.scatter(flown[:, 0], flown[:, 1], s=6, alpha=0.35, lw=0, color=COLORS[e])
            big = max(big, np.abs(flown).max())
        for r in range(len(n)):
            roj(rg, mu[r], S[r], COLORS[e])
            rg.annotate("", xy=go[r], xytext=mu[r], arrowprops=dict(arrowstyle="->", color=COLORS[e], lw=1.5))
    rg.axhline(0, color="grey", lw=0.5); rg.axvline(0, color="grey", lw=0.5)
    rg.set_xlim(-1.15 * big, 1.15 * big); rg.set_ylim(-1.15 * big, 1.15 * big); rg.set_aspect("equal")
    rg.set_xlabel("hitrost v smeri x [m/s]"); rg.set_ylabel("hitrost v smeri y [m/s]")
    if notes:
        rg.text(0.02, 0.98, "regresija: roji dveh, ki letita;\nrazdalja od 0 = hitrost, smer = kot;\n"
                "puščica = kam roj pelje hitrost v 1 s", transform=rg.transAxes, va="top", fontsize=9)


def limits(frames, lo, hi):
    allw = np.vstack([f[3] for f in frames if len(f[3])])
    return lo, hi, (0, np.percentile(allw[:, 0], 99.5) * 1.15), (0, np.percentile(allw[:, 1], 99.5) * 1.15)


def animate(frames, flights, names, secs, acc, lo, hi):
    fig = plt.figure(figsize=(15, 11))
    a = fig.add_subplot(2, 2, 1, projection="3d")
    cl, rg, c = fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
    lims = limits(frames, lo, hi)
    curves = accuracy(c, secs, acc, upto=-1)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.97, wspace=0.18, hspace=0.18)

    def show(j):
        draw(a, cl, rg, frames[j], flights, names, lims)
        m = secs <= frames[j][0] * DT
        for col, curve in enumerate(curves):
            curve.set_data(secs[m], acc[m, col])

    FuncAnimation(fig, show, frames=len(frames), blit=False).save(HERE / "behaviour.gif", writer=PillowWriter(fps=5))


# ---- seminar figures ----
def roj(ax, c, S, color, logx=False, lw=1.5):
    t = np.linspace(0, 2 * np.pi, 60)
    c = np.array(c, float)
    if logx:
        J = np.diag([1 / (c[0] * np.log(10)), 1.0])
        S, c = J @ S @ J, np.array([np.log10(c[0]), c[1]])
    vals, vecs = np.linalg.eigh(S + np.eye(2) * 0.03 ** 2)
    x, y = c[:, None] + 1.5 * vecs @ (np.sqrt(vals)[:, None] * np.array([np.cos(t), np.sin(t)]))
    if logx:
        x = 10 ** x
    ax.fill(x, y, color=color, alpha=0.15)
    ax.plot(x, y, color=color, lw=lw)


def seminar(frames, flights, names, show, per, clf, track, windows, flown):
    box = dict(left=0.17, right=0.97, bottom=0.15, top=0.95)
    size, big_text = (4.2, 4.2), {"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 9,
                                  "ytick.labelsize": 9, "legend.fontsize": 8.5}
    with plt.rc_context(big_text):
        fig = plt.figure(figsize=(1.5 * size[0], size[1]))
        a = fig.add_subplot(projection="3d")
        for e in show:
            q = flights[e]
            mine = [(j, st) for j, f in enumerate(frames) for st in f[1] if st[0] == e]
            ks = [st[1] for _, st in mine]
            a.plot(*q[min(ks):max(ks) + 1].T, color=COLORS[e], lw=2)
            for s in sorted({j // per for j, _ in mine}):
                seg = [(j, st) for j, st in mine if j // per == s]
                gap = [(j, st) for j, st in seg if st[2]]
                g0, g1 = gap[0][1][1], gap[-1][1][1] + 5
                a.plot(*q[g0:g1 + 1].T, color="white", lw=3.5)
                a.plot(*q[g0:g1 + 1].T, color=COLORS[e], lw=2.5, ls=":")
                a.plot(*np.vstack([gap[0][1][4], gap[0][1][5]]).T, color="k", lw=2, ls="--")
                u = (q[g0] - q[g0 - 3]) / max(np.linalg.norm(q[g0] - q[g0 - 3]), 1e-9)
                a.quiver(*(q[g0] - 0.9 * u), *(0.9 * u), color=COLORS[e], lw=2.5, arrow_length_ratio=1.0)
                a.plot(*q[g1], "o", ms=8, color=COLORS[e])
        a.set_xlabel("x [m]", labelpad=-2); a.set_ylabel("y [m]", labelpad=-2); a.set_zlabel("višina [m]", labelpad=-4)
        a.tick_params(pad=-2)
        for axis in (a.xaxis, a.yaxis, a.zaxis):
            axis.set_pane_color((1, 1, 1, 1)); axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
        a.view_init(elev=25, azim=-60)
        a.set_box_aspect((1.5, 1.1, 0.75), zoom=1.15)
        a.set_zticks([0, 5, 10, 15])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        a.legend(handles=[Line2D([], [], color=COLORS[e], lw=2.5, label=names[e]) for e in show]
                   + [Line2D([], [], color="grey", lw=2.5, ls=":", label="senzor ga ne vidi"),
                      Line2D([], [], color="k", lw=2, ls="--", label="napoved EvoKF"),
                      Line2D([], [], color="grey", marker=r"$\rightarrow$", lw=0, ms=14, label="tu ga izgubi"),
                      Line2D([], [], color="grey", marker="o", lw=0, ms=8, label="spet najden")],
                 loc="upper left", ncol=2, frameon=False, columnspacing=1.0)
        fig.savefig(SEMINAR / "seminar_sledenje.pdf")
        fig.savefig(HERE / "sledenje.png", dpi=120)

        fig, cl = plt.subplots(figsize=size)
        w = np.array(windows)
        cl.scatter(w[:, 0], w[:, 1], s=10, alpha=0.4, lw=0, color=[COLORS[int(e)] for e in w[:, 2]])
        S = clf.cov()[:, [FX, FY]][:, :, [FX, FY]]
        for r in range(len(clf.n)):
            roj(cl, clf.mu[r, [FX, FY]], S[r], COLORS[int(np.argmax(clf.L[r]))], lw=2.2)
        cl.set_xlabel(f"{FEATURES[FX]} [m/s²]"); cl.set_ylabel(f"{FEATURES[FY]} [m/s²]")
        cl.legend(handles=[Line2D([], [], color=COLORS[e], lw=2.5, label=f"{n}: {special(clf, e, top=1)}")
                           for e, n in enumerate(names)], loc="upper left", ncol=2, frameon=False,
                  columnspacing=1.0, handlelength=1.5)
        lo, hi = cl.get_ylim()
        cl.set_ylim(lo, hi + 0.35 * (hi - lo))
        fig.subplots_adjust(**box)
        fig.savefig(SEMINAR / "seminar_razvrscevalnik.pdf")

        fig, rg = plt.subplots(figsize=size)
        big = 0.0
        for e, n in enumerate(names):
            R = turn(e)[:2, :2].T
            f = np.array(flown[e]) @ R
            rg.scatter(f[:, 0], f[:, 1], s=5, alpha=0.25, lw=0, color=COLORS[e])
            mu, S, _, go = how_roji(track[e])
            mu, go, S = mu @ R, go @ R, R.T @ S @ R
            for r in range(len(mu)):
                roj(rg, mu[r], S[r], COLORS[e], lw=2.2)
                rg.annotate("", xy=go[r], xytext=mu[r],
                            arrowprops=dict(arrowstyle="->", color=COLORS[e], lw=2, mutation_scale=14))
                big = max(big, np.abs(mu[r]).max() + 1.5 * np.sqrt(np.linalg.eigvalsh(S[r]).max()))
            big = max(big, np.percentile(np.abs(f), 99.5))
        rg.set_xlim(-1.05 * big, 1.05 * big); rg.set_ylim(-1.05 * big, 1.05 * big); rg.set_aspect("equal")
        rg.axhline(0, color="grey", lw=0.8); rg.axvline(0, color="grey", lw=0.8)
        rg.set_xlabel("hitrost x [m/s]"); rg.set_ylabel("hitrost y [m/s]")
        rg.legend(handles=[Line2D([], [], color=COLORS[e], lw=2.5, label=n) for e, n in enumerate(names)],
                  loc="upper left", frameon=False)
        fig.subplots_adjust(**box)
        fig.savefig(SEMINAR / "seminar_regresija.pdf")
        fig.savefig(HERE / "regresija.png", dpi=120)


if __name__ == "__main__":
    main()
