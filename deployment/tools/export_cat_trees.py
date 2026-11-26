"""بيحوّل models/blend/cat_blend.cbm إلى cat_blend_trees.json (محتاج مكتبة catboost مرة واحدة بس).
الاستخدام: python3 tools/export_cat_trees.py [مجلد models]"""
import json, sys, tempfile, os
from pathlib import Path
from catboost import CatBoostRegressor

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "models")
m = CatBoostRegressor(); m.load_model(str(d / "blend/cat_blend.cbm"))
tmp = tempfile.mktemp(suffix=".json"); m.save_model(tmp, format="json")
j = json.load(open(tmp)); os.remove(tmp)
fi = j["features_info"]["float_features"]
names = m.feature_names_
by_idx = {f["flat_feature_index"]: f for f in fi}
sc, bias = m.get_scale_and_bias()
feat, border, leaf, weight = [], [], [], []
for t in j["oblivious_trees"]:
    feat.append([s["float_feature_index"] for s in t["splits"]])
    border.append([s["border"] for s in t["splits"]])
    leaf.append(t["leaf_values"]); weight.append(t["leaf_weights"])
D = len(feat[0])
assert all(len(f) == D for f in feat)
out = dict(feature_names=list(names), scale=sc, bias=bias, depth=D, feat=feat, border=border, leaf=leaf, weight=weight)
json.dump(out, open(d / "blend/cat_blend_trees.json", "w"))
print("exported", len(feat), "trees depth", D)
