"""Thử nghiệm cải thiện kết quả Bài 1 (BitcoinHeist).

Mọi lựa chọn (tỉ lệ undersampling, đặc trưng, siêu tham số, trọng số lớp) được quyết định
trên tập VALIDATION tách từ tập train. Tập TEST (giống notebook) chỉ dùng ở bước cuối.

Chạy:  python experiments.py            (kết quả ghi vào exp_out/)
"""
import json, os, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RS = 42
OUT = "exp_out"
os.makedirs(OUT, exist_ok=True)
_log = open(f"{OUT}/exp.log", "a", encoding="utf-8")
RESULTS = []


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.write(s + "\n"); _log.flush()


def record(**kw):
    RESULTS.append(kw)
    json.dump(RESULTS, open(f"{OUT}/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("RESULT", json.dumps(kw, ensure_ascii=False))


# ---------------------------------------------------------------- dữ liệu
df = pd.read_csv("BitcoinHeistData.csv")
SMOKE = os.environ.get("SMOKE") == "1"  # chạy thử nhanh trên mẫu nhỏ
if SMOKE:
    df = df.sample(300_000, random_state=RS).reset_index(drop=True)
TOP = df["label"].value_counts().drop("white").head(5).index.tolist()
df["target"] = np.where(df["label"] == "white", "white",
                np.where(df["label"].isin(TOP), df["label"], "otherRansom"))
CLASSES = sorted(df["target"].unique())
WHITE = CLASSES.index("white")
K = len(CLASSES)
y_all = pd.Series(pd.Categorical(df["target"], categories=CLASSES).codes.astype("int64"), index=df.index)

date = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["day"] - 1, unit="D")
df["month"] = date.dt.month
df["weekday"] = date.dt.dayofweek

# Đặc trưng mới
inc_int = df["income"].round().astype("int64")
df["income_tz"] = sum((inc_int % 10**k == 0).astype("int8") for k in range(1, 13))  # số chữ số 0 ở cuối (độ "tròn")
df["income_per_count"] = df["income"] / df["count"]
df["income_per_neighbor"] = df["income"] / df["neighbors"].clip(lower=1)
df["weight_per_count"] = df["weight"] / df["count"]
df["looped_ratio"] = df["looped"] / df["count"]

NUM_BASE = ["length", "weight", "count", "looped", "neighbors", "income"]
NUM_FE = NUM_BASE + ["income_tz", "income_per_count", "income_per_neighbor", "weight_per_count", "looped_ratio"]
CAT = ["year", "month", "weekday"]
ALL = NUM_FE + CAT

X = df[ALL]
# Cùng phép chia với notebook (random_state, stratify, test_size giống hệt)
X_trf, X_te, y_trf, y_te = train_test_split(X, y_all, test_size=0.2, stratify=y_all, random_state=RS)
X_tr, X_va, y_tr, y_va = train_test_split(X_trf, y_trf, test_size=0.2, stratify=y_trf, random_state=RS)
# Validation chia đôi: nửa A để chỉnh trọng số lớp, nửa B để chấm điểm (tránh chấm trên chính dữ liệu đã chỉnh)
X_vA, X_vB, y_vA, y_vB = train_test_split(X_va, y_va, test_size=0.5, stratify=y_va, random_state=RS)
log("split", len(X_tr), len(X_va), len(X_te))

# ---------------------------------------------------------------- kiểm tra rò rỉ theo address
addr = df["address"]
tr_addr = set(addr.loc[X_trf.index])
in_tr = addr.loc[X_te.index].isin(tr_addr).to_numpy()
ransom_te = (y_te != WHITE).to_numpy()
record(stage="leakage",
       test_rows_addr_in_train=float(in_tr.mean()),
       ransom_test_rows_addr_in_train=float(in_tr[ransom_te].mean()),
       white_test_rows_addr_in_train=float(in_tr[~ransom_te].mean()))


# ---------------------------------------------------------------- tiện ích
def make_pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def undersample(Xs, ys, n_white, seed=RS):
    if n_white is None:
        return Xs, ys
    w = ys.index[ys == WHITE]
    keep = np.concatenate([np.random.RandomState(seed).choice(w, n_white, replace=False), ys.index[ys != WHITE]])
    return Xs.loc[keep], ys.loc[keep]


def macro_f1_fast(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)))


def tune_class_weights(P, y, passes=3):
    """Nhân xác suất mỗi lớp với w_k, tìm w (coordinate ascent) tối đa macro F1."""
    w = np.ones(K)
    grid = np.exp(np.linspace(-5, 5, 41))
    best = macro_f1_fast(y, (P * w).argmax(1))
    for _ in range(passes):
        for k in range(K):
            for g in grid:
                w2 = w.copy(); w2[k] = g
                s = macro_f1_fast(y, (P * w2).argmax(1))
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best


def scores(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return {"macro_f1": macro_f1_fast(y, p),
            "weighted_f1": float(f1_score(y, p, average="weighted")),
            "accuracy": float((y == p).mean()),
            "binary_f1_ransom": float(f1_score(y != WHITE, p != WHITE))}


def fit_proba(est, num, n_white, X_fit=None, y_fit=None, X_evals=()):
    X_fit = X_tr if X_fit is None else X_fit
    y_fit = y_tr if y_fit is None else y_fit
    Xs, ys = undersample(X_fit, y_fit, n_white)
    pipe = Pipeline([("pre", make_pre(num, CAT)), ("clf", clone(est))])
    t0 = time.perf_counter(); pipe.fit(Xs[num + CAT], ys.to_numpy()); t_fit = time.perf_counter() - t0
    out = []
    for Xe in X_evals:
        t0 = time.perf_counter(); P = pipe.predict_proba(Xe[num + CAT]); out.append((P, time.perf_counter() - t0))
    return pipe, t_fit, out


def eval_val(name, est, num, n_white, stage, **extra):
    """Train trên X_tr; chỉnh trọng số lớp trên val-A; chấm trên val-B."""
    _, t_fit, [(PA, _), (PB, t_pred)] = fit_proba(est, num, n_white, X_evals=(X_vA, X_vB))
    raw = scores(y_vB, PB.argmax(1))
    w, _ = tune_class_weights(PA, y_vA.to_numpy())
    adj = scores(y_vB, (PB * w).argmax(1))
    record(stage=stage, model=name, n_white=n_white, n_feat=len(num), train_s=round(t_fit, 2),
           pred_s=round(t_pred, 2), raw=raw, adjusted=adj, **extra)
    return adj["macro_f1"]


HGB = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=RS)

# ---------------------------------------------------------------- A. tỉ lệ undersampling
n_white_tr = int((y_tr == WHITE).sum())
best_a, best_nw = -1, None
for nw in ([5_000, 20_000, None] if SMOKE else [100_000, 200_000, 500_000, 1_000_000, None]):
    s = eval_val("HGB", HGB, NUM_BASE, nw, "A_undersample")
    if s > best_a:
        best_a, best_nw = s, nw
log("A best n_white =", best_nw, best_a)

# ---------------------------------------------------------------- B. đặc trưng mới
s_fe = eval_val("HGB", HGB, NUM_FE, best_nw, "B_features")
NUM = NUM_FE if s_fe > best_a else NUM_BASE
log("B features chosen:", "FE" if NUM is NUM_FE else "BASE")

# ---------------------------------------------------------------- C. tinh chỉnh siêu tham số (random search)
rng = random.Random(RS)
SPACES = {
    "HGB": (lambda p: HistGradientBoostingClassifier(random_state=RS, early_stopping=True, validation_fraction=0.1,
                                                     n_iter_no_change=30, max_iter=2000, **p),
            {"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_leaf_nodes": [31, 63, 127, 255],
             "min_samples_leaf": [20, 50, 100, 300], "l2_regularization": [0.0, 1.0, 10.0],
             "max_features": [0.5, 0.8, 1.0]}, 12),
    "LightGBM": (lambda p: LGBMClassifier(n_estimators=2000, random_state=RS, n_jobs=24, verbose=-1, **p),
                 {"learning_rate": [0.05, 0.1], "num_leaves": [31, 63, 127, 255],
                  "min_child_samples": [20, 50, 100], "subsample": [0.7, 1.0], "subsample_freq": [1],
                  "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]}, 8),
    "XGBoost-GPU": (lambda p: XGBClassifier(device="cuda", tree_method="hist", random_state=RS,
                                            eval_metric="mlogloss", **p),
                    {"n_estimators": [300, 600, 1000], "learning_rate": [0.03, 0.05, 0.1],
                     "max_depth": [6, 8, 10, 12], "min_child_weight": [1, 5, 10], "subsample": [0.7, 0.9],
                     "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [1.0, 5.0]}, 10),
    "RandomForest": (lambda p: RandomForestClassifier(n_jobs=24, random_state=RS, **p),
                     {"n_estimators": [300], "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.5],
                      "max_depth": [None, 30]}, 4),
}
# LightGBM n_estimators lớn cần dừng sớm → dùng cố định theo learning_rate để đơn giản
LGBM_ITERS = {0.05: 600, 0.1: 300}

best_cfg = {}
for name, (build, space, n_trials) in SPACES.items():
    best_s, best_p = -1, None
    for t in range(2 if SMOKE else n_trials):
        p = {k: rng.choice(v) for k, v in space.items()}
        est = build(p)
        if name == "LightGBM":
            est.set_params(n_estimators=LGBM_ITERS[p["learning_rate"]])
        try:
            s = eval_val(name, est, NUM, best_nw, "C_tuning", params=p)
        except Exception as e:  # cấu hình hỏng không làm dừng cả thí nghiệm
            log("FAIL", name, p, repr(e)); continue
        if s > best_s:
            best_s, best_p = s, p
    best_cfg[name] = best_p
    log("C best", name, best_s, best_p)
json.dump(best_cfg, open(f"{OUT}/best_params.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- D. đánh giá cuối trên TEST + ensemble
final_P_va, final_P_te = {}, {}
for name, (build, _, _) in SPACES.items():
    p = best_cfg[name]
    est = build(p)
    if name == "LightGBM":
        est.set_params(n_estimators=LGBM_ITERS[p["learning_rate"]])
    _, t_fit, [(Pv, _), (Pt, t_pred)] = fit_proba(est, NUM, best_nw, X_evals=(X_va, X_te))
    w, _ = tune_class_weights(Pv, y_va.to_numpy())
    final_P_va[name], final_P_te[name] = Pv, Pt
    record(stage="D_test", model=name, params=p, train_s=round(t_fit, 2), pred_s=round(t_pred, 2),
           class_weights=w.round(4).tolist(), raw=scores(y_te, Pt.argmax(1)), adjusted=scores(y_te, (Pt * w).argmax(1)))

# Ensemble: trung bình xác suất của các mô hình boosting
for combo in [("HGB", "LightGBM", "XGBoost-GPU"), ("HGB", "LightGBM", "XGBoost-GPU", "RandomForest")]:
    Pv = np.mean([final_P_va[m] for m in combo], axis=0)
    Pt = np.mean([final_P_te[m] for m in combo], axis=0)
    w, _ = tune_class_weights(Pv, y_va.to_numpy())
    record(stage="D_test", model="Ensemble(" + "+".join(combo) + ")", class_weights=w.round(4).tolist(),
           raw=scores(y_te, Pt.argmax(1)), adjusted=scores(y_te, (Pt * w).argmax(1)))

np.save(f"{OUT}/classes.npy", np.array(CLASSES))
log("DONE")
