"""Huấn luyện & đánh giá các cấu hình trong configs.py.

    python -m final.train --rep A                     # biểu diễn A, chia ngẫu nhiên (như Bài 1)
    python -m final.train --rep B                     # biểu diễn B, chia theo address
    python -m final.train --rep A --split group --only A_gru,A_cnn_bn_do

Quy trình cho mỗi cấu hình:
1. Train trên tập train đã undersample (100k white + toàn bộ ransomware), Adam + cross-entropy.
2. Sau mỗi epoch đo loss / accuracy / macro F1 trên train và val_us (validation cùng tỉ lệ white);
   early stopping theo macro F1 trên val_us, giữ trọng số tốt nhất.
3. Dự đoán xác suất trên toàn bộ validation → tìm trọng số lớp w (như Bài 1) → test: argmax(p·w).
Tập test chỉ được dùng ở bước 3, sau khi mô hình và w đã cố định.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from final import configs, data, models

OUT = data.ROOT / "outputs" / "final"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1_fast(y, p, K):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)))


def tune_class_weights(P, y, K, passes=3):
    """Tìm w sao cho argmax(P·w) có macro F1 cao nhất trên validation (giống Bài 1)."""
    w = np.ones(K); grid = np.exp(np.linspace(-5, 5, 41)); best = macro_f1_fast(y, P.argmax(1), K)
    for _ in range(passes):
        for k in range(K):
            for g in grid:
                w2 = w.copy(); w2[k] = g
                s = macro_f1_fast(y, (P * w2).argmax(1), K)
                if s > best + 1e-9:
                    best, w = s, w2
    return w


def truncate_history(hist: np.ndarray, L: int) -> np.ndarray:
    """Giữ L bước cuối (gần nhất) của mỗi lịch sử đệm phải."""
    n = (hist >= 0).sum(1)
    start = np.clip(n - L, 0, None)
    cols = start[:, None] + np.arange(L)
    out = np.take_along_axis(hist, np.clip(cols, 0, hist.shape[1] - 1), 1)
    return np.where(cols < n[:, None], out, -1).astype(np.int32)


def count_features(prep: data.Prepared) -> np.ndarray:
    """log1p(số lần xuất hiện trước), cờ lần đầu, log1p(khoảng cách ngày) — chuẩn hoá theo tập fit."""
    g = prep.gap
    X = np.stack([np.log1p(prep.n_prev), (g < 0), np.log1p(np.clip(g, 0, None))], 1).astype(np.float32)
    fit = prep.idx["fit"]
    return (X - X[fit].mean(0)) / X[fit].std(0)


class Batcher:
    """Giữ toàn bộ dữ liệu trên GPU, cắt batch theo chỉ số dòng."""

    def __init__(self, prep: data.Prepared, rep: str, history_len: int | None, side_count: bool = False):
        self.rep = rep
        self.y = torch.as_tensor(prep.y, device=DEV)
        side = prep.side
        if side_count:                       # thêm số lần xuất hiện trước đó + khoảng cách ngày (giống XGB_count)
            side = np.concatenate([side, count_features(prep)], 1)
        self.side = torch.as_tensor(side, device=DEV)
        if rep == "A":
            self.x = torch.as_tensor(prep.num, device=DEV)
            self.in_dim, self.seq_len = prep.num.shape[1], prep.num.shape[1]
        else:
            self.x = torch.as_tensor(prep.steps, device=DEV)
            self.hist = torch.as_tensor(truncate_history(prep.hist, history_len), device=DEV)
            self.in_dim, self.seq_len = prep.steps.shape[1], history_len

    def __call__(self, rows):
        rows = torch.as_tensor(rows, device=DEV)
        if self.rep == "A":
            return self.x[rows], self.side[rows], None, self.y[rows]
        h = self.hist[rows].long()
        mask = h >= 0
        x = self.x[h.clamp(min=0)] * mask.unsqueeze(-1)
        return x, self.side[rows], mask, self.y[rows]


@torch.no_grad()
def predict_proba(model, batcher, rows, bs=16384):
    model.eval()
    out = [torch.softmax(model(*batcher(rows[i:i + bs])[:3]), 1).cpu() for i in range(0, len(rows), bs)]
    return torch.cat(out).numpy()


def evaluate(model, batcher, rows, K, crit):
    P = predict_proba(model, batcher, rows)
    y = batcher.y[torch.as_tensor(rows, device=DEV)].cpu().numpy()
    loss = crit(torch.log(torch.as_tensor(P).clamp(min=1e-12)), torch.as_tensor(y)).item()
    p = P.argmax(1)
    return loss, float((p == y).mean()), macro_f1_fast(y, p, K)


def run(cfg: dict, prep: data.Prepared, split_kind: str) -> dict:
    tc, K = cfg["train"], len(prep.classes)
    out_dir = OUT / f"{cfg['rep']}_{split_kind}" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tc["seed"]); np.random.seed(tc["seed"])

    batcher = Batcher(prep, cfg["rep"], tc.get("history_len"), tc.get("side_count", False))
    model = models.build(cfg, batcher.in_dim, batcher.side.shape[1], K, batcher.seq_len).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    crit = nn.CrossEntropyLoss()
    fit, rng = prep.idx["fit"], np.random.RandomState(tc["seed"])
    print(f"\n=== {cfg['name']} ({split_kind}) | {models.n_params(model):,} tham số | fit {len(fit):,} dòng", flush=True)

    hist, best, best_state, bad, t_train = [], -1.0, None, 0, 0.0
    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        perm, losses, preds, ys = rng.permutation(fit), [], [], []
        for i in range(0, len(perm), tc["batch_size"]):
            x, side, mask, y = batcher(perm[i:i + tc["batch_size"]])
            logits = model(x, side, mask)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item() * len(y)); preds.append(logits.argmax(1)); ys.append(y)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0; t_train += dt
        p_tr, y_tr = torch.cat(preds).cpu().numpy(), torch.cat(ys).cpu().numpy()
        v_loss, v_acc, v_f1 = evaluate(model, batcher, prep.idx["val_us"], K, crit)
        rec = dict(epoch=epoch, time_s=dt, train_loss=sum(losses) / len(perm), train_acc=float((p_tr == y_tr).mean()),
                   train_macro_f1=macro_f1_fast(y_tr, p_tr, K), val_loss=v_loss, val_acc=v_acc, val_macro_f1=v_f1)
        hist.append(rec)
        print(f"ep{epoch:02d} {dt:5.1f}s | train loss {rec['train_loss']:.4f} f1 {rec['train_macro_f1']:.4f} | "
              f"val loss {v_loss:.4f} f1 {v_f1:.4f}", flush=True)
        if v_f1 > best + 1e-4:
            best, bad = v_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    best_epoch = max(hist, key=lambda r: r["val_macro_f1"])["epoch"]

    # Hiệu chỉnh trọng số lớp trên validation, rồi mới đánh giá test
    va, te = prep.idx["val"], prep.idx["test"]
    w = tune_class_weights(predict_proba(model, batcher, va), prep.y[va], K)
    t0 = time.perf_counter()
    P_te = predict_proba(model, batcher, te)
    pred = (P_te * w).argmax(1)
    t_test = time.perf_counter() - t0
    y_te = prep.y[te]
    rep = classification_report(y_te, pred, labels=range(K), target_names=prep.classes, output_dict=True,
                                zero_division=0)
    res = dict(name=cfg["name"], rep=cfg["rep"], split=split_kind, model=cfg["model"], params=cfg["params"],
               train=cfg["train"], n_params=models.n_params(model), epochs=len(hist), best_epoch=best_epoch,
               train_time_s=t_train, time_per_epoch_s=t_train / len(hist), test_time_s=t_test,
               val_us_macro_f1=best, accuracy=rep["accuracy"],
               weighted_precision=rep["weighted avg"]["precision"], weighted_recall=rep["weighted avg"]["recall"],
               weighted_f1=rep["weighted avg"]["f1-score"], macro_precision=rep["macro avg"]["precision"],
               macro_recall=rep["macro avg"]["recall"], macro_f1=rep["macro avg"]["f1-score"],
               binary_f1_ransom=float(f1_score(y_te != prep.white, pred != prep.white)),
               per_class={c: rep[c] for c in prep.classes}, class_weights=w.tolist(),
               confusion=confusion_matrix(y_te, pred, labels=range(K)).tolist())
    print(f"--> test macro F1 {res['macro_f1']:.4f} | binary F1 {res['binary_f1_ransom']:.4f} | "
          f"acc {res['accuracy']:.4f} | best epoch {best_epoch}", flush=True)

    json.dump(hist, open(out_dir / "history.json", "w"), indent=1)
    json.dump(res, open(out_dir / "result.json", "w"), indent=1, ensure_ascii=False)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "class_weights": w, "classes": prep.classes,
                "in_dim": batcher.in_dim, "seq_len": batcher.seq_len}, out_dir / "model.pt")
    np.save(out_dir / "pred_test.npy", pred.astype(np.int8))
    np.save(out_dir / "proba_test.npy", P_te.astype(np.float16))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["A", "B"], required=True)
    ap.add_argument("--split", choices=["random", "group"], default=None, help="mặc định: A→random, B→group")
    ap.add_argument("--only", default="", help="danh sách tên cấu hình, cách nhau bởi dấu phẩy hoặc +")
    ap.add_argument("--seeds", default="42", help="vd 42,43,44: lặp mỗi cấu hình với nhiều seed khởi tạo")
    a = ap.parse_args()
    split_kind = a.split or ("random" if a.rep == "A" else "group")
    cfgs = [c for c in (configs.A if a.rep == "A" else configs.B) if not a.only or c["name"] in re.split(r"[,+]", a.only)]
    cfgs = [{**c, "name": c["name"] + ("" if s == 42 else f"_s{s}"), "train": {**c["train"], "seed": s}}
            for s in map(int, re.split(r"[,+]", a.seeds)) for c in cfgs]       # seed khác → khởi tạo & thứ tự batch khác

    t0 = time.perf_counter()
    L = max((c["train"].get("history_len", 0) for c in cfgs), default=0)
    prep = data.prepare(split_kind, history_len=L or None)
    print(f"Chuẩn bị dữ liệu {time.perf_counter() - t0:.0f}s | device {DEV} | "
          + " | ".join(f"{k} {len(v):,}" for k, v in prep.idx.items()), flush=True)
    for c in cfgs:
        # Nhiều tiến trình có thể chạy song song: mỗi cấu hình chỉ do tiến trình đầu tiên tạo được file claim chạy
        d = OUT / f"{a.rep}_{split_kind}" / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(d / "claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        run(c, prep, split_kind)


if __name__ == "__main__":
    main()
