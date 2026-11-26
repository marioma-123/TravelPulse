"""موديل CatBoost متصدّر لتريز عادية: التنبؤ والـ SHAP بـ numpy بس، من غير مكتبة catboost.

الملف models/blend/cat_blend_trees.json اتولّد من cat_blend.cbm بـ tools/export_cat_trees.py
(والنتايج اتقارنت بـ catboost الأصلي).
"""
import json
from pathlib import Path

import numpy as np


class CatTrees:
    def __init__(self, path):
        j = json.loads(Path(path).read_text())
        self.feature_names = list(j["feature_names"])
        self.scale = float(j["scale"])
        self.bias = float(j["bias"])
        self.depth = int(j["depth"])
        self.feat = np.asarray(j["feat"], dtype=np.int64)          # (T, D)
        self.border = np.asarray(j["border"], dtype=np.float64)    # (T, D)
        self.leaf = np.asarray(j["leaf"], dtype=np.float64)        # (T, 2**D)
        self.weight = np.asarray(j["weight"], dtype=np.float64)    # (T, 2**D)
        self._pow = (1 << np.arange(self.depth)).astype(np.int64)

    def predict_row(self, x) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        bits = (x[self.feat] > self.border).astype(np.int64)
        idx = (bits * self._pow).sum(axis=1)
        vals = self.leaf[np.arange(len(idx)), idx]
        return float(vals.sum() * self.scale + self.bias)

    def shap_row(self, x):
        """(مساهمات كل عمود، القيمة الأساسية) - مجموعهم = predict_row."""
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        phi = np.zeros(n)
        base = 0.0
        D = self.depth
        for t in range(self.feat.shape[0]):
            ph, b = _tree_shap(x, self.feat[t], self.border[t], self.leaf[t], self.weight[t], D, n)
            phi += ph
            base += b
        return phi * self.scale, base * self.scale + self.bias


def _tree_shap(x, feat, border, leaf, weight, D, n):
    """Shapley الدقيق (path-dependent) لشجرة oblivious: الـ levels الأقل أهمية = البِت الأقل.
    نمرّ على الـ subsets لـ features الفريدة (<= D) ونحسب القيمة المتوقعة بأوزان الأوراق."""
    feats = sorted(set(int(f) for f in feat))
    k = len(feats)
    pos = {f: i for i, f in enumerate(feats)}
    L = 1 << D
    leaves = np.arange(L)
    # لكل ورقة: هل مسارها بيطابق x عند كل level؟
    match = np.empty((D, L), dtype=bool)
    for d in range(D):
        go_right = x[feat[d]] > border[d]
        match[d] = (((leaves >> d) & 1) == 1) == go_right
    # cover لكل عقدة: levels بتتقسم من الأعلى (D-1) للأدنى (0) لأن بِت d بيمثّل level d
    # (ترتيب الـ splits في catboost: split 0 هو الأقل وزناً في index الورقة)
    # عقدة على level d بتتحدد بالبِتات d..D-1 من الورقة، و cover = مجموع أوزان الأوراق اللي ليها نفس البتات دي.
    def cover_at(d):
        # cover العقدة التي تحتوي كل ورقة، بعد تثبيت البِتات من d لفوق
        grp = leaves >> d
        sums = np.bincount(grp, weights=weight)
        return sums[grp]
    cov = [cover_at(d) for d in range(D + 1)]  # cov[D] = كل الأوراق، cov[0] = الورقة نفسها
    # احتمال ذهاب المسار لورقة، لما features في S معروفة: نمشي من level D-1 لـ 0
    # الأب هو cov[d+1] والابن cov[d]
    ratio = [np.where(cov[d + 1] > 0, cov[d] / np.where(cov[d + 1] > 0, cov[d + 1], 1), 0.0) for d in range(D)]
    # v(S)
    nsub = 1 << k
    v = np.zeros(nsub)
    for S in range(nsub):
        p = np.ones(L)
        for d in range(D):
            if (S >> pos[int(feat[d])]) & 1:
                p = p * match[d]
            else:
                p = p * ratio[d]
        v[S] = float((p * leaf).sum())
    # Shapley من v
    from math import factorial
    phi_u = np.zeros(k)
    for i in range(k):
        for S in range(nsub):
            if (S >> i) & 1:
                continue
            s = bin(S).count("1")
            w = factorial(s) * factorial(k - s - 1) / factorial(k)
            phi_u[i] += w * (v[S | (1 << i)] - v[S])
    ph = np.zeros(n)
    for f, i in pos.items():
        ph[f] += phi_u[i]
    return ph, v[0]
