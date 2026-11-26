"""بيحوّل models/blend/lgbm_blend.txt إلى lgbm_blend_trees.json (محتاج مكتبة lightgbm مرة واحدة بس).
الاستخدام: python3 tools/export_lgbm_trees.py [مجلد models]"""
import json, sys
from pathlib import Path
import lightgbm as lgb

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "models")
b = lgb.Booster(model_file=str(d / "blend/lgbm_blend.txt"))
dump = b.dump_model()
assert dump["objective"].startswith("regression"), dump["objective"]
trees = []
for ti in dump["tree_info"]:
    root = ti["tree_structure"]
    t = dict(feat=[], thr=[], default_left=[], left=[], right=[], internal_count=[],
             leaf_value=[], leaf_count=[], single=False)
    if "leaf_value" in root:
        t.update(single=True, leaf_value=[root["leaf_value"]], leaf_count=[1])
        trees.append(t); continue
    def walk(n):
        if "leaf_index" in n:
            t["leaf_value"].append(None); t["leaf_count"].append(None)
            i = n["leaf_index"]
            while len(t["leaf_value"]) <= i:
                t["leaf_value"].append(None); t["leaf_count"].append(None)
            t["leaf_value"][i] = n["leaf_value"]; t["leaf_count"][i] = n["leaf_count"]
            return ~i
        assert n["decision_type"] == "<=", n["decision_type"]
        i = len(t["feat"])
        t["feat"].append(n["split_feature"]); t["thr"].append(n["threshold"])
        t["default_left"].append(bool(n["default_left"]))
        t["left"].append(0); t["right"].append(0); t["internal_count"].append(n["internal_count"])
        t["left"][i] = walk(n["left_child"]); t["right"][i] = walk(n["right_child"])
        return i
    walk(root)
    # أزل الـ None الزائدة (الورقة اتضافت أكتر من مرة بالـ padding)
    nl = ti["num_leaves"]
    t["leaf_value"] = t["leaf_value"][:nl]; t["leaf_count"] = t["leaf_count"][:nl]
    assert None not in t["leaf_value"]
    trees.append(t)
json.dump(dict(feature_names=b.feature_name(), trees=trees), open(d / "blend/lgbm_blend_trees.json", "w"))
print("exported", len(trees), "trees")
