import json, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
snaps = sorted(glob.glob(os.path.join(HERE, "freq-before-*.json")))
names = ["warm"] + [os.path.basename(s)[15:-5] for s in snaps[:-1]]
D = [np.array(json.load(open(s))["decode"], dtype=np.float64) for s in snaps]
P = [np.array(json.load(open(s))["prefill"], dtype=np.float64) for s in snaps]
last = json.load(open(snaps[-1]))
per = [D[0]] + [D[i] - D[i - 1] for i in range(1, len(D))]
pper = [P[0]] + [P[i] - P[i - 1] for i in range(1, len(P))]
tot = D[-1]; L, E = tot.shape
LOCKED = [0, 1, 2, 3, 37, 38, 39]
free = [l for l in range(L) if l not in LOCKED]
MIB = 17280.0 / (L * E)
print("shape", tot.shape, "MiB/expert %.3f" % MIB)
print("decode hits per layer (min/max):", tot.sum(1).min(), tot.sum(1).max(), " => tokens ~", tot.sum(1).max() / 8)
for n, d, p in zip(names, per, pper):
    print("  %-10s decode tok %6.0f  prefill hits/layer %8.0f" % (n, d.sum(1).max() / 8, p.sum(1).max()))


def need(v, frac):
    s = np.sort(v)[::-1]; c = np.cumsum(s)
    return int(np.searchsorted(c, frac * c[-1]) + 1)


print("\nper-layer decode skew (all tasks): layer never-used need50 need90 need99 top32share")
rows = []
for l in range(L):
    v = tot[l]
    rows.append(((v == 0).sum(), need(v, .5), need(v, .9), need(v, .99), np.sort(v)[::-1][:32].sum() / v.sum()))
    if l % 4 == 0 or l in LOCKED:
        print("  L%02d %4d %4d %4d %4d  %.2f%s" % (l, *rows[-1], " (locked)" if l in LOCKED else ""))
r = np.array(rows)[free]
print("  mean over unlocked layers: never %.0f need50 %.0f need90 %.0f need99 %.0f top32 %.2f" % tuple(r.mean(0)))

print("\nglobal ranking over unlocked layers (decode): coverage vs RAM kept")
v = tot[free].ravel(); s = np.sort(v)[::-1]; c = np.cumsum(s) / s.sum()
total_gib = len(v) * MIB / 1024
for g in [1, 2, 3, 4, 6, 8, 10, 12]:
    k = int(g * 1024 / MIB)
    if k < len(v):
        print("  hottest %5.1f GiB (%5d experts): %.4f of hits, miss/token %.2f of %d" % (g, k, c[k - 1], (1 - c[k - 1]) * 8 * len(free), 8 * len(free)))
print("  unlocked total %.2f GiB; never used in decode: %d of %d (%.2f GiB)" % (total_gib, (v == 0).sum(), len(v), (v == 0).sum() * MIB / 1024))

print("\nleave-one-task-out: hot set chosen from the OTHER tasks, coverage on the held-out task")
for g in [2, 4, 6, 8, 10]:
    k = int(g * 1024 / MIB); out = []
    for i in range(1, len(per)):
        if per[i].sum() < 1000: continue
        oth = (tot - per[i])[free].ravel(); idx = np.argsort(oth)[::-1][:k]
        h = per[i][free].ravel(); out.append((names[i], h[idx].sum() / h.sum()))
    print("  %2d GiB: " % g + " ".join("%s %.3f" % o for o in out) + "  | min %.3f" % min(o[1] for o in out))

print("\nper-task working set (distinct experts used per unlocked layer, mean) and GiB")
for n, d in zip(names, per):
    if d.sum() < 1000: continue
    ws = (d[free] > 0).sum()
    print("  %-10s %5.1f/layer  %.2f GiB" % (n, ws / len(free), ws * MIB / 1024))

print("\nprefill: tokens", last.get("prefill_tokens"))
pd_ = np.array(last["prefill_distinct"], dtype=np.float64)
print("  mean distinct experts per prefill call per layer (all calls incl. tiny): %.1f" % (pd_[:, 0].sum() / max(pd_[:, 1].sum(), 1)))
for n, p in zip(names, pper):
    if p.sum() < 50000: continue
    u = (p[free] > 0).sum(1)
    print("  %-10s distinct experts/layer min %d mean %.0f max %d ; need90 mean %.0f" % (n, u.min(), u.mean(), u.max(), np.mean([need(p[l], .9) for l in free])))
pt = P[-1][free].ravel(); dt = tot[free].ravel()
print("  corr(prefill counts, decode counts) %.3f" % np.corrcoef(pt, dt)[0, 1])
both = np.argsort(dt + pt)[::-1]
cc = np.cumsum((dt + pt)[both]) / (dt + pt).sum()
for g in [4, 6, 8, 10]:
    k = int(g * 1024 / MIB); print("  decode+prefill hits covered by hottest %d GiB: %.4f" % (g, cc[k - 1]))
