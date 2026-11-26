"""موديل LightGBM متصدّر لتريز عادية: التنبؤ والـ SHAP بـ Python/numpy بس، من غير مكتبة lightgbm.

models/blend/lgbm_blend_trees.json بيتولّد من lgbm_blend.txt بـ tools/export_lgbm_trees.py
(والنتايج اتقارنت بـ lightgbm الأصلي).
"""
import json
from pathlib import Path

import numpy as np


class LgbmTrees:
    def __init__(self, path):
        j = json.loads(Path(path).read_text())
        self.feature_names = list(j["feature_names"])   # بعد استبدال المسافات بـ _
        self.trees = j["trees"]
        for t in self.trees:   # قوائم Python أسرع من numpy للعناصر المفردة
            t["leaf_cover"] = t["leaf_count"]

    def _leaf(self, t, x):
        if t["single"]:
            return -1
        node = 0
        feat, thr, dl, L, R = t["feat"], t["thr"], t["default_left"], t["left"], t["right"]
        while node >= 0:
            v = x[feat[node]]
            if v != v:
                go_left = dl[node]
            else:
                go_left = v <= thr[node]
            node = L[node] if go_left else R[node]
        return ~node

    def predict_row(self, x) -> float:
        x = [float(v) for v in np.asarray(x, dtype=float).ravel()]
        s = 0.0
        for t in self.trees:
            s += t["leaf_value"][0] if t["single"] else t["leaf_value"][self._leaf(t, x)]
        return s

    def shap_row(self, x):
        x = [float(v) for v in np.asarray(x, dtype=float).ravel()]
        n = len(x)
        phi = np.zeros(n)
        base = 0.0
        for t in self.trees:
            if t["single"]:
                base += t["leaf_value"][0]
                continue
            base += _expected(t)
            _tree_shap(t, x, phi)
        return phi, base


def _expected(t):
    lc = t["leaf_count"]
    tot = float(sum(lc))
    return sum(v * c for v, c in zip(t["leaf_value"], lc)) / tot


def _tree_shap(t, x, phi):
    """TreeSHAP الأصلي (path-dependent) بنفس cover اللي LightGBM بيستخدمه (عدد العينات)."""
    feat, thr, dl, L, R = t["feat"], t["thr"], t["default_left"], t["left"], t["right"]
    icount, lcount, lval = t["internal_count"], t["leaf_count"], t["leaf_value"]

    def cnt(c):
        return lcount[~c] if c < 0 else icount[c]

    def extend(path, zf, of, fi):
        path = [q[:] for q in path] + [[fi, zf, of, 1.0 if not path else 0.0]]
        d = len(path) - 1
        for i in range(d - 1, -1, -1):
            path[i + 1][3] += of * path[i][3] * (i + 1) / (d + 1)
            path[i][3] = zf * path[i][3] * (d - i) / (d + 1)
        return path

    def unwound_sum(path, pi):
        d = len(path) - 1
        of, zf = path[pi][2], path[pi][1]
        nxt = path[d][3]
        total = 0.0
        for i in range(d - 1, -1, -1):
            if of != 0:
                tmp = nxt * (d + 1) / ((i + 1) * of)
                total += tmp
                nxt = path[i][3] - tmp * zf * (d - i) / (d + 1)
            else:
                total += (path[i][3] / zf) / ((d - i) / (d + 1))
        return total

    def unwind(path, pi):
        d = len(path) - 1
        of, zf = path[pi][2], path[pi][1]
        nxt = path[d][3]
        path = [p[:] for p in path]
        for i in range(d - 1, -1, -1):
            if of != 0:
                tmp = path[i][3]
                path[i][3] = nxt * (d + 1) / ((i + 1) * of)
                nxt = tmp - path[i][3] * zf * (d - i) / (d + 1)
            else:
                path[i][3] = path[i][3] * (d + 1) / (zf * (d - i))
        for i in range(pi, d):
            path[i][0], path[i][1], path[i][2] = path[i + 1][0], path[i + 1][1], path[i + 1][2]
        return path[:-1]

    def recurse(node, path, pz, po, pf):
        path = extend(path, pz, po, pf)
        if node < 0:
            v = lval[~node]
            for i in range(1, len(path)):
                w = unwound_sum(path, i)
                phi[path[i][0]] += w * (path[i][2] - path[i][1]) * v
            return
        f = feat[node]
        xv = x[f]
        go_left = dl[node] if xv != xv else xv <= thr[node]
        hot, cold = (L[node], R[node]) if go_left else (R[node], L[node])
        tot = cnt(L[node]) + cnt(R[node])
        hz, cz = cnt(hot) / tot, cnt(cold) / tot
        iz, io = 1.0, 1.0
        k = next((i for i in range(1, len(path)) if path[i][0] == f), None)
        if k is not None:
            iz, io = path[k][1], path[k][2]
            path = unwind(path, k)
        recurse(hot, path, hz * iz, io, f)
        recurse(cold, path, cz * iz, 0.0, f)

    recurse(0, [], 1.0, 1.0, -1)
