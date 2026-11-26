# Staged pipeline: reproduce the Colab tabular blend, train deployment models, export.
# شغّله من أي مكان: python3 tools/retrain_blend.py <stage>
# يحتاج: data/processed/listings_final.csv + machine_learning/export_for_deployment/parity_sample.csv
# Stages بالترتيب: prep | baseline | es_xgb es_lgbm es_cat | oof_<model>_<0..4> | weights
#                  | dep_xgb dep_lgbm dep_cat | finalize
# Stages: prep | baseline | es_{xgb,lgbm,cat} | oof_{xgb,lgbm,cat}_{0..4} | weights
#         | dep_{xgb,lgbm,cat} | finalize
import json, os, sys, gc, math, time
import numpy as np
import pandas as pd
import joblib

RANDOM_STATE = 42
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]          # TravelPulse repo root
OUT = BASE / "deployment"
CK = Path(os.getenv("BLEND_CKPT", "/tmp/blend_ckpt"))
os.makedirs(CK, exist_ok=True)
os.makedirs(f"{OUT}/models/blend", exist_ok=True)
DATA = f"{BASE}/data/processed/listings_final.csv"
ART = json.load(open(f"{BASE}/deployment/models/artifacts.json"))

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

FIXED = {"xgb": dict(learning_rate=0.05),
         "lgbm": dict(learning_rate=0.05, subsample_freq=1),
         "cat": dict(learning_rate=0.08)}
TUNED = {
 "xgb": {**FIXED["xgb"], 'max_depth': 5, 'min_child_weight': 10, 'subsample': 0.892797576724562,
         'colsample_bytree': 0.7993292420985183, 'reg_lambda': 0.02938027938703535,
         'reg_alpha': 0.0029375384576328283, 'gamma': 0.11616722433639892},
 "lgbm": {**FIXED["lgbm"], 'num_leaves': 93, 'min_child_samples': 10, 'subsample': 0.6609079086683075,
          'colsample_bytree': 0.5347932177377853, 'reg_lambda': 0.010535179945522732,
          'reg_alpha': 0.5072957033687833},
 "cat": {**FIXED["cat"], 'depth': 7, 'l2_leaf_reg': 1.3787764619353768,
         'bagging_temperature': 0.29214464853521815, 'random_strength': 1.4159046082342293},
}
BEST_IT = {"xgb": 506, "lgbm": 311, "cat": 614}
KEYS = ["xgb", "lgbm", "cat"]

def load_ck():
    d = joblib.load(f"{CK}/matrices.joblib")
    return d["X_train"], d["X_test"], d["y_tr"], d["y_te"], d["groups"], d["city_test"], d["fit_idx"], d["val_idx"]

def fixed_model(kind, params, n):
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=n, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, **params)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=n, random_seed=RANDOM_STATE, verbose=0, thread_count=-1, **params)

def add_report(d):
    rp = f"{CK}/report.json"
    rep = json.load(open(rp)) if os.path.exists(rp) else {}
    rep.update(d)
    json.dump(rep, open(rp, "w"), indent=2, default=float)

stage = sys.argv[1]

if stage == "prep":
    from sklearn.model_selection import GroupShuffleSplit
    USECOLS = ["id","host_id","price","price_missing","price_outlier","min_nights_outlier",
    "amenities","property_type","last_scraped","first_review","last_review","city",
    "neighbourhood_cleansed","room_type","latitude","longitude","dist_center_km",
    "dist_nearest_landmark_km","landmarks_1km","landmarks_2km","accommodates","bedrooms",
    "beds","bathrooms","bathroom_shared","has_license","host_is_superhost",
    "host_identity_verified","host_tenure_years","calculated_host_listings_count",
    "availability_30","availability_365","minimum_nights_avg_ntm","number_of_reviews",
    "number_of_reviews_ltm","reviews_per_month","has_reviews","review_scores_rating",
    "review_scores_location","review_scores_cleanliness","review_scores_value"]
    df = pd.read_csv(DATA, low_memory=False, usecols=USECOLS,
                     parse_dates=["last_scraped", "first_review", "last_review"])
    log("csv loaded", df.shape)
    bad = (df["price_missing"].astype(bool) | df["price_outlier"].astype(bool) | df["min_nights_outlier"].astype(bool))
    df = df[~bad].copy()

    def count_amenities(s):
        try:
            return len(json.loads(s))
        except Exception:
            return 0
    am_raw = df["amenities"].fillna("[]")
    am = am_raw.str.lower()
    df["n_amenities"] = am_raw.apply(count_amenities)
    key_amenities = {"has_wifi": "wifi", "has_kitchen": "kitchen", "has_ac": "air conditioning",
                     "has_pool": "pool", "has_parking": "parking", "has_washer": "washer",
                     "has_tv": "tv", "has_heating": "heating", "has_elevator": "elevator"}
    for c, kw in key_amenities.items():
        df[c] = am.str.contains(kw, regex=False).astype(int)
    del am_raw, am; gc.collect()

    top_types = df["property_type"].value_counts().head(10).index
    df["property_type"] = df["property_type"].where(df["property_type"].isin(top_types), "Other")
    ref_date = df["last_scraped"].max()
    df["days_since_last_review"] = (ref_date - df["last_review"]).dt.days
    df["listing_age_days"]       = (ref_date - df["first_review"]).dt.days
    df["nbh_key"] = df["city"] + " | " + df["neighbourhood_cleansed"].fillna("unknown")
    for c in ["host_is_superhost", "host_identity_verified", "has_license", "bathroom_shared", "has_reviews"]:
        df[c] = df[c].fillna(False).astype(int)

    TARGET, GROUP = "price", "host_id"
    num_features = [
        "latitude", "longitude", "dist_center_km", "dist_nearest_landmark_km", "landmarks_1km", "landmarks_2km",
        "accommodates", "bedrooms", "beds", "bathrooms", "bathroom_shared", "has_license",
        "n_amenities", *key_amenities.keys(),
        "host_is_superhost", "host_identity_verified", "host_tenure_years", "calculated_host_listings_count",
        "availability_30", "availability_365", "minimum_nights_avg_ntm",
        "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
        "has_reviews", "days_since_last_review", "listing_age_days",
        "review_scores_rating", "review_scores_location", "review_scores_cleanliness", "review_scores_value",
    ]
    cat_onehot = ["city", "room_type", "property_type"]
    model_df = df[[GROUP, TARGET] + num_features + cat_onehot + ["nbh_key"]].copy()
    model_df["log_price"] = np.log1p(model_df[TARGET])
    del df; gc.collect()
    log("model_df", model_df.shape, "ref_date", ref_date.date())

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df[GROUP]))
    train = model_df.iloc[train_idx].copy()
    test  = model_df.iloc[test_idx].copy()
    log(f"Train: {len(train):,} | Test: {len(test):,} | shared hosts:", len(set(train[GROUP]) & set(test[GROUP])))
    assert len(train) == 22934 and len(test) == 4958, "split mismatch vs Colab run"

    def fill_with_group_median(train, test, col, by):
        med = train.groupby(by)[col].median().rename("_m").reset_index()
        overall = train[col].median()
        for d in (train, test):
            m = d[by].merge(med, on=by, how="left")["_m"].values
            d[col] = d[col].fillna(pd.Series(m, index=d.index)).fillna(overall)
    for c in ["bedrooms", "beds"]:
        fill_with_group_median(train, test, c, by=["room_type", "accommodates"])
    for d in (train, test):
        d["reviews_per_month"] = d["reviews_per_month"].fillna(0)
    for c in num_features:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)

    def smoothed_mean(y, cat, prior, m=20):
        stats = y.groupby(cat).agg(["mean", "count"])
        return (stats["count"] * stats["mean"] + m * prior) / (stats["count"] + m)
    from sklearn.model_selection import GroupKFold
    col = "nbh_key"
    global_mean = train["log_price"].mean()
    train["nbh_te"] = np.nan
    gkf = GroupKFold(n_splits=5)
    loc = train.columns.get_loc("nbh_te")
    for tr, val in gkf.split(train, groups=train[GROUP]):
        enc = smoothed_mean(train.iloc[tr]["log_price"], train.iloc[tr][col], global_mean)
        train.iloc[val, loc] = train.iloc[val][col].map(enc).fillna(global_mean).values
    enc_full = smoothed_mean(train["log_price"], train[col], global_mean)
    test["nbh_te"] = test[col].map(enc_full).fillna(global_mean)

    city_test = test["city"].values
    feature_cols = num_features + ["nbh_te"]
    X_train = pd.get_dummies(train[feature_cols + cat_onehot], columns=cat_onehot, dtype=int)
    X_test  = pd.get_dummies(test[feature_cols + cat_onehot],  columns=cat_onehot, dtype=int)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_train, y_test = train["log_price"], test["log_price"]
    log("X_train:", X_train.shape, "X_test:", X_test.shape)
    assert X_train.shape == (22934, 59) and X_test.shape == (4958, 59)
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0

    # validate vs original export artifacts
    checks = {}
    checks["ref_date_matches"] = str(ref_date.date()) == ART["reference_date"]
    checks["num_features_match"] = list(num_features) == ART["num_features"]
    checks["feature_columns_match"] = list(X_train.columns) == ART["feature_columns"]
    checks["top_types_match"] = set(top_types) == set(ART["top_property_types"])
    checks["nbh_global_mean_diff"] = abs(float(global_mean) - ART["nbh_global_mean"])
    enc_s = pd.Series(enc_full); art_enc = pd.Series(ART["nbh_encoding"])
    common = enc_s.index.intersection(art_enc.index)
    checks["nbh_encoding_keys"] = f"train={len(enc_s)} artifacts={len(art_enc)} common={len(common)}"
    checks["nbh_encoding_maxdiff"] = float((enc_s[common] - art_enc[common]).abs().max()) if len(common) else None
    for k, v in checks.items():
        log("CHECK", k, "=", v)
    assert checks["feature_columns_match"] and checks["num_features_match"]
    assert checks["nbh_global_mean_diff"] < 1e-6
    assert checks["nbh_encoding_maxdiff"] is not None and checks["nbh_encoding_maxdiff"] < 1e-6

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(gss2.split(X_train, groups=train[GROUP]))
    joblib.dump({"X_train": X_train, "X_test": X_test,
                 "y_tr": np.asarray(y_train), "y_te": np.asarray(y_test),
                 "groups": np.asarray(train[GROUP]), "city_test": city_test,
                 "fit_idx": fit_idx, "val_idx": val_idx}, f"{CK}/matrices.joblib")
    add_report({"checks": checks})
    log("prep saved")

elif stage == "baseline":
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error
    X_train, X_test, y_tr, y_te, groups, city_test, fit_idx, val_idx = load_ck()
    m = XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=3,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                     tree_method="hist", early_stopping_rounds=50, n_jobs=-1, random_state=RANDOM_STATE)
    m.fit(X_train.iloc[fit_idx], y_tr[fit_idx], eval_set=[(X_train.iloc[val_idx], y_tr[val_idx])], verbose=False)
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(m.predict(X_test)))
    log(f"baseline xgb: best_iter={m.best_iteration} test MAE {mae:.3f} (Colab 41.552 / original local 41.36)")
    add_report({"baseline_repro": {"best_iter": int(m.best_iteration), "test_MAE": mae}})

elif stage.startswith("es_"):
    kind = stage[3:]
    from sklearn.metrics import mean_absolute_error
    X_train, X_test, y_tr, y_te, groups, city_test, fit_idx, val_idx = load_ck()
    Xa, ya = X_train.iloc[fit_idx], y_tr[fit_idx]
    Xb, yb = X_train.iloc[val_idx], y_tr[val_idx]
    if kind == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=5000, tree_method="hist", random_state=RANDOM_STATE,
                         n_jobs=-1, early_stopping_rounds=50, **TUNED[kind])
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        n = m.best_iteration + 1
    elif kind == "lgbm":
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(n_estimators=5000, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **TUNED[kind])
        m.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)])
        n = m.best_iteration_
    else:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=5000, random_seed=RANDOM_STATE, verbose=0,
                              thread_count=-1, early_stopping_rounds=100, **TUNED[kind])
        m.fit(Xa, ya, eval_set=(Xb, yb), use_best_model=True)
        n = m.get_best_iteration() + 1
    pred = m.predict(X_test)
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(pred))
    np.save(f"{CK}/es_pred_{kind}.npy", pred)
    log(f"{kind}_tuned ES: best_iter={n} test MAE {mae:.3f} (Colab: xgb 41.516 / lgbm 41.753 / cat 41.049)")
    add_report({f"{kind}_es": {"best_iter": int(n), "test_MAE": mae}})

elif stage.startswith("oof_"):
    _, kind, fold_s = stage.split("_")
    fold_i = int(fold_s)
    from sklearn.model_selection import GroupKFold
    X_train, X_test, y_tr, y_te, groups, city_test, fit_idx, val_idx = load_ck()
    folds5 = list(GroupKFold(n_splits=5).split(X_train, y_tr, groups))
    tr_i, va_i = folds5[fold_i]
    m = fixed_model(kind, TUNED[kind], BEST_IT[kind]).fit(X_train.iloc[tr_i], y_tr[tr_i])
    np.save(f"{CK}/oof_{kind}_{fold_i}.npy", np.column_stack([va_i, m.predict(X_train.iloc[va_i])]))
    log(f"OOF {kind} fold {fold_i+1}/5 done")

elif stage == "weights":
    from scipy.optimize import nnls
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    X_train, X_test, y_tr, y_te, groups, city_test, fit_idx, val_idx = load_ck()
    oof = {}
    for k in KEYS:
        o = np.zeros(len(X_train))
        for i in range(5):
            arr = np.load(f"{CK}/oof_{k}_{i}.npy")
            o[arr[:, 0].astype(int)] = arr[:, 1]
        oof[k] = o
        add_report({f"oof_{k}_mae": mean_absolute_error(np.expm1(y_tr), np.expm1(o))})
        log(f"OOF {k}: MAE(USD) {mean_absolute_error(np.expm1(y_tr), np.expm1(o)):.3f} (Colab: 43.72/43.46/43.32)")
    O = np.column_stack([oof[k] for k in KEYS])
    w, _ = nnls(O, y_tr); w = w / w.sum()
    Wd = dict(zip(KEYS, [float(x) for x in w]))
    log("weights:", {k: round(v, 4) for k, v in Wd.items()}, "(Colab printed .148/.385/.467)")
    np.save(f"{CK}/weights.npy", w)
    T_es = np.column_stack([np.load(f"{CK}/es_pred_{k}.npy") for k in KEYS])
    blend_es = T_es @ w
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(blend_es))
    rmse = root_mean_squared_error(np.expm1(y_te), np.expm1(blend_es))
    r2 = r2_score(np.expm1(y_te), np.expm1(blend_es))
    log(f"ES blend test: MAE {mae:.3f} RMSE {rmse:.3f} R2 {r2:.4f} (Colab 40.927/60.686/0.761)")
    add_report({"weights": Wd, "es_blend_test": {"MAE": mae, "RMSE": rmse, "R2": r2}})

elif stage.startswith("dep_"):
    kind = stage[4:]
    from sklearn.metrics import mean_absolute_error
    X_train, X_test, y_tr, y_te, groups, city_test, fit_idx, val_idx = load_ck()
    m = fixed_model(kind, TUNED[kind], BEST_IT[kind]).fit(X_train, y_tr)
    pred = m.predict(X_test)
    np.save(f"{CK}/dep_pred_{kind}.npy", pred)
    if kind == "xgb":
        m.get_booster().save_model(f"{OUT}/models/blend/xgb_blend.json")
    elif kind == "lgbm":
        m.booster_.save_model(f"{OUT}/models/blend/lgbm_blend.txt")
    else:
        m.save_model(f"{OUT}/models/blend/cat_blend.cbm")
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(pred))
    log(f"deployment {kind}: fitted on full train ({BEST_IT[kind]} iters), solo test MAE {mae:.3f}, exported")
    add_report({f"dep_{kind}_solo_mae": mae})

elif stage == "finalize":
    import shutil
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    import xgboost, lightgbm, catboost
    X_train, X_test, y_tr, y_te, groups, city_test, fit_idx, val_idx = load_ck()
    w = np.load(f"{CK}/weights.npy")
    W = dict(zip(KEYS, [float(x) for x in w]))
    T_dep = np.column_stack([np.load(f"{CK}/dep_pred_{k}.npy") for k in KEYS])
    blend_dep = T_dep @ w
    yt, p = np.expm1(y_te), np.expm1(blend_dep)
    met = {"MAE": mean_absolute_error(yt, p), "RMSE": root_mean_squared_error(yt, p), "R2": r2_score(yt, p)}
    for c in np.unique(city_test):
        met[f"MAE_{c}"] = mean_absolute_error(yt[city_test == c], p[city_test == c])
    log("DEPLOYMENT blend test:", {k: round(v, 3) for k, v in met.items()})

    xb = xgboost.Booster(); xb.load_model(f"{OUT}/models/blend/xgb_blend.json")
    assert xb.feature_names == ART["feature_columns"], "xgb feature names mismatch"
    lb = lightgbm.Booster(model_file=f"{OUT}/models/blend/lgbm_blend.txt")
    # LightGBM sanitizes names (space -> underscore) when saving; compare sanitized
    sanitized = [c.replace(" ", "_") for c in ART["feature_columns"]]
    assert lb.feature_name() == sanitized, "lgbm feature names mismatch"
    cb = catboost.CatBoostRegressor(); cb.load_model(f"{OUT}/models/blend/cat_blend.cbm")
    assert list(cb.feature_names_) == ART["feature_columns"], "cat feature names mismatch"
    log("exported model feature names all match app artifacts")

    resid = np.sort(np.abs(y_te - blend_dep))
    n = len(resid)
    def q(level):
        k = min(math.ceil((n + 1) * level), n)
        return float(resid[k - 1])
    conformal = {"n_calibration": n, "quantiles": {"0.8": q(0.8), "0.9": q(0.9)}, "model": "blend"}
    log("conformal:", conformal)
    pd.DataFrame({"y_true_log": y_te, "y_pred_log": blend_dep}).to_csv(f"{OUT}/models/calibration_blend.csv", index=False)

    import subprocess
    for exp in ("export_lgbm_trees.py", "export_cat_trees.py"):  # الـ app بيشتغل بدون lightgbm/catboost
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / exp), f"{OUT}/models"], check=True)

    art = dict(ART)
    art["blend"] = {
        "files": {"xgb": "blend/xgb_blend.json", "lgbm_trees": "blend/lgbm_blend_trees.json", "cat_trees": "blend/cat_blend_trees.json"},
        "source_files": {"lgbm": "blend/lgbm_blend.txt", "cat": "blend/cat_blend.cbm"},
        "weights": W,
        "library_versions": {"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__},
        "tuned_params": TUNED,
        "best_iters": BEST_IT,
        "test_metrics": met,
        "note": "Weighted log-space blend of XGBoost+LightGBM+CatBoost; weights fit by NNLS on 5-fold host-grouped OOF predictions.",
    }
    art["conformal"] = conformal
    json.dump(art, open(f"{OUT}/models/artifacts.json", "w"))

    shutil.copy(f"{BASE}/data/processed/landmarks_clean.csv", f"{OUT}/models/landmarks_clean.csv")
    shutil.copy(f"{BASE}/machine_learning/export_for_deployment/parity_sample.csv", f"{OUT}/models/parity_sample.csv")
    shutil.copy(f"{BASE}/deployment/models/xgb_model.json", f"{OUT}/models/xgb_model.json")
    shutil.copytree(f"{BASE}/deployment/models/dummy", f"{OUT}/models/dummy", dirs_exist_ok=True)
    shutil.copy(f"{BASE}/deployment/models/__init__.py", f"{OUT}/models/__init__.py")
    shutil.copy(f"{BASE}/deployment/models/make_dummy_model.py", f"{OUT}/models/make_dummy_model.py")
    add_report({"deployment_blend_test": met})
    shutil.copy(f"{CK}/report.json", f"{OUT}/build_report.json")
    log("finalize DONE")

else:
    raise SystemExit(f"unknown stage {stage}")
