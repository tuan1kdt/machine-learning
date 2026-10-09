"""Các cấu hình thực nghiệm. Mỗi cấu hình chỉ ghi phần khác so với cấu hình gốc của nhóm."""

TRAIN = dict(lr=1e-3, batch_size=512, max_epochs=100, patience=10, weight_decay=0.0, seed=42)

RNN = dict(cell="lstm", d=16, hidden=64, layers=1, bidirectional=False, dropout=0.2, pool="last")
CNN = dict(d=16, channels=(32, 64), kernel=3, bn=False, dropout=0.0, pool="max", flatten="flatten")
MLP = dict(hidden=128, layers=2, dropout=0.2)


def cfg(name, rep, model, base, params=None, **train):
    return dict(name=name, rep=rep, model=model, params={**base, **(params or {})}, train={**TRAIN, **train})


def rnn(name, rep, params=None, **train):
    return cfg(name, rep, "rnn", RNN if rep == "A" else {**RNN, "d": 32}, params, **train)


def cnn(name, rep, params=None, **train):
    return cfg(name, rep, "cnn", CNN if rep == "A" else {**CNN, "d": 32, "flatten": "gap"}, params, **train)


# ---------- Biểu diễn A: chuỗi 11 đặc trưng ----------
A = [
    rnn("A_lstm", "A"),
    rnn("A_gru", "A", {"cell": "gru"}),
    rnn("A_lstm_2l", "A", {"layers": 2}),
    rnn("A_gru_2l", "A", {"cell": "gru", "layers": 2}),
    rnn("A_gru_h128", "A", {"cell": "gru", "hidden": 128}),
    rnn("A_bigru", "A", {"cell": "gru", "bidirectional": True}),
    rnn("A_gru_mean", "A", {"cell": "gru", "pool": "mean"}),
    rnn("A_gru_attn", "A", {"cell": "gru", "pool": "attn"}),
    rnn("A_gru_nodrop", "A", {"cell": "gru", "dropout": 0.0}),
    rnn("A_gru_lr3e-4", "A", {"cell": "gru"}, lr=3e-4),
    rnn("A_gru_bs2048", "A", {"cell": "gru"}, batch_size=2048),

    cnn("A_cnn_simple", "A"),
    cnn("A_cnn_bn", "A", {"bn": True}),
    cnn("A_cnn_do", "A", {"dropout": 0.3}),
    cnn("A_cnn_bn_do", "A", {"bn": True, "dropout": 0.3}),
    cnn("A_cnn_gap", "A", {"bn": True, "dropout": 0.3, "flatten": "gap"}),
    cnn("A_cnn_3l", "A", {"bn": True, "dropout": 0.3, "channels": (32, 64, 128)}),
    cnn("A_cnn_wide", "A", {"bn": True, "dropout": 0.3, "channels": (64, 128)}),
    cnn("A_cnn_k5", "A", {"bn": True, "dropout": 0.3, "kernel": 5}),
    cnn("A_cnn_avgpool", "A", {"bn": True, "dropout": 0.3, "pool": "avg"}),
    cnn("A_cnn_lr3e-4", "A", {"bn": True, "dropout": 0.3}, lr=3e-4),
    cnn("A_cnn_bs2048", "A", {"bn": True, "dropout": 0.3}, batch_size=2048),

    cfg("A_mlp", "A", "mlp", MLP),
]

# ---------- Biểu diễn B: lịch sử của address (L bước gần nhất) ----------
B = [
    rnn("B_lstm", "B"),
    rnn("B_gru", "B", {"cell": "gru"}),
    rnn("B_gru_2l", "B", {"cell": "gru", "layers": 2}),
    rnn("B_bigru", "B", {"cell": "gru", "bidirectional": True}),
    rnn("B_gru_mean", "B", {"cell": "gru", "pool": "mean"}),
    rnn("B_gru_attn", "B", {"cell": "gru", "pool": "attn"}),
    rnn("B_gru_L1", "B", {"cell": "gru"}, history_len=1),      # đối chứng: chỉ dòng hiện tại (+ cờ lần đầu, Δngày)
    rnn("B_gru_L4", "B", {"cell": "gru"}, history_len=4),
    rnn("B_gru_L32", "B", {"cell": "gru"}, history_len=32),

    cnn("B_cnn_simple", "B"),
    cnn("B_cnn_bn_do", "B", {"bn": True, "dropout": 0.3}),
    cnn("B_cnn_gmp", "B", {"bn": True, "dropout": 0.3, "flatten": "gmp"}),
    cnn("B_cnn_3l", "B", {"bn": True, "dropout": 0.3, "channels": (32, 64, 128)}),
    cnn("B_cnn_k5", "B", {"bn": True, "dropout": 0.3, "kernel": 5}),

    cfg("B_mlp", "B", "mlp", MLP),                              # chỉ dòng hiện tại, không có lịch sử

    # + đặc trưng đếm (số lần xuất hiện trước, khoảng cách ngày) ở đầu vào phụ → so sánh công bằng với XGB_count
    rnn("B_gru_cnt", "B", {"cell": "gru"}, side_count=True),
    cnn("B_cnn_cnt", "B", {"bn": True, "dropout": 0.3}, side_count=True),
    cfg("B_mlp_cnt", "B", "mlp", MLP, side_count=True),
]
for c in B:
    c["train"].setdefault("history_len", 16)

ALL = {c["name"]: c for c in A + B}
