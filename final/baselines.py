"""Baseline dạng bảng (XGBoost) để đối chứng với RNN/CNN.

    python -m final.baselines

- random / XGB_current : chỉ dòng hiện tại, chia ngẫu nhiên → đối chứng biểu diễn A (lặp lại Bài 1).
- group  / XGB_current : chỉ dòng hiện tại, chia theo address.
- group  / XGB_count   : + số lần address đã xuất hiện, khoảng cách ngày tới lần trước.
- group  / XGB_hist    : + thống kê (mean, max) của 15 bản ghi trước đó → đối thủ "bảng" của biểu diễn B.
Cùng undersampling, cùng hiệu chỉnh trọng số lớp trên validation như train.py.
"""
import json
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

from final import data
from final.train import OUT, tune_class_weights

PARAMS = json.load(open(data.ROOT / "exp_out" / "best_params.json"))["XGBoost-GPU"]


def history_features(prep: data.Prepared, rows: np.ndarray) -> np.ndarray:
    """mean & max của đặc trưng số trên các bản ghi TRƯỚC dòng hiện tại (tối đa L-1 bản ghi), 0 nếu không có."""
    out = []
    for i in range(0, len(rows), 200_000):
        h = prep.hist[rows[i:i + 200_000]]
        n = (h >= 0).sum(1)
        prev = (np.arange(h.shape[1]) < (n - 1)[:, None])                  # bỏ bước cuối = dòng hiện tại
        X = prep.num[np.clip(h, 0, None)] * prev[..., None]
        cnt = np.maximum(prev.sum(1, keepdims=True), 1)
        mx = np.where(prev[..., None], X, -np.inf).max(1)
        out.append(np.concatenate([X.sum(1) / cnt, np.where(np.isfinite(mx), mx, 0)], 1))
    return np.concatenate(out).astype(np.float32)


def features(prep: data.Prepared, rows: np.ndarray, kind: str) -> np.ndarray:
    parts = [prep.num[rows], prep.side[rows]]
    if kind in ("count", "hist"):
        g = prep.gap[rows]
        parts.append(np.stack([np.log1p(prep.n_prev[rows]), (g < 0), np.log1p(np.clip(g, 0, None))], 1))
    if kind == "hist":
        parts.append(history_features(prep, rows))
    return np.concatenate(parts, 1).astype(np.float32)


def run(prep: data.Prepared, split_kind: str, kind: str) -> dict:
    name, K = f"XGB_{kind}", len(prep.classes)
    fit, va, te = prep.idx["fit"], prep.idx["val"], prep.idx["test"]
    clf = XGBClassifier(device="cuda", tree_method="hist", random_state=data.RS, eval_metric="mlogloss", **PARAMS)
    t0 = time.perf_counter(); clf.fit(features(prep, fit, kind), prep.y[fit]); t_train = time.perf_counter() - t0
    w = tune_class_weights(clf.predict_proba(features(prep, va, kind)), prep.y[va], K)
    t0 = time.perf_counter(); pred = (clf.predict_proba(features(prep, te, kind)) * w).argmax(1)
    t_test = time.perf_counter() - t0
    y = prep.y[te]
    rep = classification_report(y, pred, labels=range(K), target_names=prep.classes, output_dict=True, zero_division=0)
    res = dict(name=name, rep="tabular", split=split_kind, model="xgboost", params=PARAMS, train_time_s=t_train,
               test_time_s=t_test, accuracy=rep["accuracy"], weighted_f1=rep["weighted avg"]["f1-score"],
               weighted_precision=rep["weighted avg"]["precision"], weighted_recall=rep["weighted avg"]["recall"],
               macro_precision=rep["macro avg"]["precision"], macro_recall=rep["macro avg"]["recall"],
               macro_f1=rep["macro avg"]["f1-score"],
               binary_f1_ransom=float(f1_score(y != prep.white, pred != prep.white)),
               per_class={c: rep[c] for c in prep.classes}, class_weights=w.tolist(),
               confusion=confusion_matrix(y, pred, labels=range(K)).tolist())
    out_dir = OUT / f"baseline_{split_kind}" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / "result.json", "w"), indent=1, ensure_ascii=False)
    np.save(out_dir / "pred_test.npy", pred.astype(np.int8))
    print(f"{split_kind:6s} {name:12s} macro F1 {res['macro_f1']:.4f} | binary F1 {res['binary_f1_ransom']:.4f} | "
          f"train {t_train:.1f}s", flush=True)
    return res


if __name__ == "__main__":
    run(data.prepare("random"), "random", "current")
    prep = data.prepare("group", history_len=16)
    for kind in ["current", "count", "hist"]:
        run(prep, "group", kind)
