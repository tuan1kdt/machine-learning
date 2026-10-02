"""Kiểm tra rò rỉ theo address: so sánh chia ngẫu nhiên (random split) với chia theo nhóm address
(group split — một address chỉ nằm ở train HOẶC test). Dùng cấu hình tốt nhất từ experiments.py.

Chạy sau experiments.py:  python experiments_group.py
"""
import json, os, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RS, OUT = 42, "exp_out"
best = json.load(open(f"{OUT}/best_params.json"))
res = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_csv("BitcoinHeistData.csv")
TOP = df["label"].value_counts().drop("white").head(5).index.tolist()
df["target"] = np.where(df["label"] == "white", "white",
                np.where(df["label"].isin(TOP), df["label"], "otherRansom"))
CLASSES = sorted(df["target"].unique()); K = len(CLASSES); WHITE = CLASSES.index("white")
y_all = pd.Series(pd.Categorical(df["target"], categories=CLASSES).codes.astype("int64"), index=df.index)
date = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["day"] - 1, unit="D")
df["month"], df["weekday"] = date.dt.month, date.dt.dayofweek
inc_int = df["income"].round().astype("int64")
df["income_tz"] = sum((inc_int % 10**k == 0).astype("int8") for k in range(1, 13))
df["income_per_count"] = df["income"] / df["count"]
df["income_per_neighbor"] = df["income"] / df["neighbors"].clip(lower=1)
df["weight_per_count"] = df["weight"] / df["count"]
df["looped_ratio"] = df["looped"] / df["count"]
NUM = ["length", "weight", "count", "looped", "neighbors", "income",
       "income_tz", "income_per_count", "income_per_neighbor", "weight_per_count", "looped_ratio"]
CAT = ["year", "month", "weekday"]
X = df[NUM + CAT]; groups = df["address"]


def macro_f1_fast(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)))


def tune_w(P, y):
    w = np.ones(K); grid = np.exp(np.linspace(-5, 5, 41)); bestv = macro_f1_fast(y, P.argmax(1))
    for _ in range(3):
        for k in range(K):
            for g in grid:
                w2 = w.copy(); w2[k] = g; s = macro_f1_fast(y, (P * w2).argmax(1))
                if s > bestv + 1e-9: bestv, w = s, w2
    return w


def run(split_name, tr_idx, va_idx, te_idx):
    Xtr, ytr = X.loc[tr_idx], y_all.loc[tr_idx]
    w_idx = ytr.index[ytr == WHITE]
    keep = np.concatenate([np.random.RandomState(RS).choice(w_idx, 100_000, replace=False), ytr.index[ytr != WHITE]])
    for name, est in [("HGB", HistGradientBoostingClassifier(random_state=RS, early_stopping=True, validation_fraction=0.1,
                                                            n_iter_no_change=30, max_iter=2000, **best["HGB"])),
                      ("XGBoost-GPU", XGBClassifier(device="cuda", tree_method="hist", random_state=RS,
                                                    eval_metric="mlogloss", **best["XGBoost-GPU"]))]:
        pipe = Pipeline([("pre", ColumnTransformer([
            ("num", Pipeline([("l", FunctionTransformer(np.log1p)), ("s", StandardScaler())]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])), ("clf", est)])
        pipe.fit(X.loc[keep], y_all.loc[keep].to_numpy())
        w = tune_w(pipe.predict_proba(X.loc[va_idx]), y_all.loc[va_idx].to_numpy())
        p = (pipe.predict_proba(X.loc[te_idx]) * w).argmax(1)
        yt = y_all.loc[te_idx].to_numpy()
        r = {"split": split_name, "model": name, "macro_f1": macro_f1_fast(yt, p),
             "weighted_f1": float(f1_score(yt, p, average="weighted")),
             "binary_f1_ransom": float(f1_score(yt != WHITE, p != WHITE)),
             "per_class_f1": dict(zip(CLASSES, f1_score(yt, p, average=None, labels=range(K)).round(4).tolist()))}
        res.append(r); log(json.dumps(r, ensure_ascii=False))
        json.dump(res, open(f"{OUT}/group_results.json", "w"), indent=1, ensure_ascii=False)


# 1) Random split (giống notebook / experiments.py)
trf, te = train_test_split(df.index, test_size=0.2, stratify=y_all, random_state=RS)
tr, va = train_test_split(trf, test_size=0.2, stratify=y_all.loc[trf], random_state=RS)
run("random", tr, va, te)

# 2) Group split theo address
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
trf_g, te_g = next(gss.split(X, y_all, groups))
trf_g, te_g = df.index[trf_g], df.index[te_g]
tr_g, va_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS).split(trf_g, groups=groups.loc[trf_g]))
tr_g, va_g = trf_g[tr_g], trf_g[va_g]
log("group split sizes", len(tr_g), len(va_g), len(te_g),
    "overlap", len(set(groups.loc[tr_g]) & set(groups.loc[te_g])))
run("group_by_address", tr_g, va_g, te_g)
log("DONE")
