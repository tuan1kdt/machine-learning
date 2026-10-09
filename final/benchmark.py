"""Đo tốc độ huấn luyện công bằng: chạy riêng (GPU không bị chia sẻ), mỗi kiến trúc 1 epoch khởi động + 5 epoch đo.

    python -m final.benchmark
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn

from final import configs, data, models
from final.train import DEV, OUT, Batcher

NAMES = {"A": ["A_lstm", "A_gru", "A_lstm_2l", "A_gru_2l", "A_bigru", "A_cnn_simple", "A_cnn_bn_do", "A_cnn_3l", "A_mlp"],
         "B": ["B_lstm", "B_gru", "B_bigru", "B_cnn_bn_do", "B_cnn_3l", "B_mlp"]}


def epoch_time(cfg, prep, n_epochs=5):
    tc = cfg["train"]
    torch.manual_seed(0)
    b = Batcher(prep, cfg["rep"], tc.get("history_len"))
    m = models.build(cfg, b.in_dim, b.side.shape[1], len(prep.classes), b.seq_len).to(DEV)
    opt, crit, fit = torch.optim.Adam(m.parameters(), 1e-3), nn.CrossEntropyLoss(), prep.idx["fit"]
    times = []
    for e in range(n_epochs + 1):
        perm = np.random.permutation(fit)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(perm), tc["batch_size"]):
            x, s, mask, y = b(perm[i:i + tc["batch_size"]])
            loss = crit(m(x, s, mask), y)
            opt.zero_grad(); loss.backward(); opt.step()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times[1:])), models.n_params(m)


if __name__ == "__main__":
    res = {}
    for rep, split_kind in [("A", "random"), ("B", "group")]:
        prep = data.prepare(split_kind, history_len=16 if rep == "B" else None)
        for n in NAMES[rep]:
            t, p = epoch_time(configs.ALL[n], prep)
            res[n] = {"time_per_epoch_s": t, "n_params": p}
            print(f"{n:14s} {t:6.2f} s/epoch  {p:>8,} tham số", flush=True)
    json.dump(res, open(OUT / "benchmark.json", "w"), indent=1)
