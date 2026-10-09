"""Gom kết quả trong outputs/final/ thành bảng (CSV + LaTeX) và hình cho notebook + báo cáo.

    python -m final.report
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final import data
from final.train import OUT

FIG = OUT / "figures"
TAB = OUT / "tables"
SEED_RE = re.compile(r"_s\d+$")
BAI1 = data.ROOT / "outputs" / "summary.csv"          # kết quả mô hình nền của Bài 1 (cùng cách chia ngẫu nhiên)
SHORT = {"montrealCryptXXX": "CryptXXX", "montrealCryptoLocker": "CryptoLocker", "otherRansom": "otherRansom",
         "paduaCryptoWall": "CryptoWall", "princetonCerber": "Cerber", "princetonLocky": "Locky", "white": "white"}


# ---------------------------------------------------------------- tải kết quả
def load_results() -> pd.DataFrame:
    df = pd.DataFrame([json.load(open(p)) for p in sorted(OUT.glob("*/*/result.json"))])
    df["group"] = np.where(df["rep"] == "tabular", "baseline_" + df["split"], df["rep"] + "_" + df["split"])
    df["config"] = df["name"].str.replace(SEED_RE, "", regex=True)
    return df


def load_history(group: str, name: str) -> pd.DataFrame:
    return pd.DataFrame(json.load(open(OUT / group / name / "history.json")))


def describe(r) -> str:
    """Mô tả ngắn kiến trúc chính của một cấu hình."""
    p, t = r["params"], r.get("train") or {}
    if r["model"] == "rnn":
        s = f"{'Bi' if p['bidirectional'] else ''}{p['cell'].upper()} {p['layers']}x{p['hidden']}, {p['pool']}"
    elif r["model"] == "cnn":
        s = (f"Conv {'-'.join(map(str, p['channels']))} k{p['kernel']} {p['pool']}pool"
             f"{' +BN' if p['bn'] else ''}, {p['flatten']}")
    elif r["model"] == "mlp":
        s = f"MLP {p['layers']}x{p['hidden']}"
    else:
        return {"XGB_current": "XGBoost, dòng hiện tại", "XGB_count": "XGBoost + đếm lịch sử",
                "XGB_hist": "XGBoost + đếm + mean/max lịch sử"}.get(r["name"], "XGBoost")
    if r["model"] != "rnn" or p["dropout"] != 0.2:
        s += f", do {p['dropout']}"
    if t.get("history_len") and t["history_len"] != 16:
        s += f", L={t['history_len']}"
    if t.get("side_count"):
        s += ", +đếm"
    return s


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mỗi cấu hình 1 dòng; nếu có nhiều seed → trung bình (và độ lệch chuẩn của macro F1)."""
    agg = df.groupby(["group", "config"]).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), n_seeds=("macro_f1", "size"),
        binary_f1=("binary_f1_ransom", "mean"), accuracy=("accuracy", "mean"), weighted_f1=("weighted_f1", "mean"),
        macro_precision=("macro_precision", "mean"), macro_recall=("macro_recall", "mean"),
        train_time_s=("train_time_s", "mean")).reset_index()
    first = df[df["name"] == df["config"]].set_index(["group", "config"])
    get = lambda col: [first[col].get(k, np.nan) if col in first else np.nan for k in zip(agg["group"], agg["config"])]
    for col in ["model", "n_params", "epochs", "best_epoch", "time_per_epoch_s"]:
        agg[col] = get(col)
    tr = get("train")
    agg["lr"] = [t.get("lr") if isinstance(t, dict) else np.nan for t in tr]
    agg["batch_size"] = [t.get("batch_size") if isinstance(t, dict) else np.nan for t in tr]
    agg["arch"] = [describe(first.loc[k]) for k in zip(agg["group"], agg["config"])]
    return agg.sort_values(["group", "macro_f1"], ascending=[True, False]).reset_index(drop=True)


def bai1_baselines() -> pd.DataFrame:
    b = pd.read_csv(BAI1)
    return b.rename(columns={"binary_f1_ransom": "binary_f1"})[["model", "macro_f1", "binary_f1", "accuracy",
                                                                   "weighted_f1", "train_time_s"]]


# ---------------------------------------------------------------- bảng LaTeX
def tex_escape(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_f1(m, s):
    return f"{m:.4f}" + (f" $\\pm$ {s:.4f}" if pd.notna(s) else "")


def latex_config_table(t: pd.DataFrame, notes: dict, fname: str, caption: str, label: str):
    lines = [r"\begin{table}[H]\centering\scriptsize", f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}p{3.6cm}rrrrrrr>{\raggedright\arraybackslash}X@{}}",
             r"\toprule",
             r"Cấu hình & Kiến trúc chính & LR & Batch & Ep. & Tham số & s/ep. & Macro F1 & F1 nhị phân & Nhận xét \\",
             r"\midrule"]
    best = t["macro_f1"].max()
    for _, r in t.iterrows():
        f1 = fmt_f1(r["macro_f1"], r["macro_f1_std"])
        if r["macro_f1"] == best:
            f1 = r"\textbf{" + f1 + "}"
        ep = f"{int(r['best_epoch'])}/{int(r['epochs'])}" if pd.notna(r["epochs"]) else "--"
        lines.append(" & ".join([
            r"\code{" + tex_escape(r["config"]) + "}", tex_escape(r["arch"]),
            f"{r['lr']:.0e}".replace("e-0", "e-") if pd.notna(r["lr"]) else "--",
            f"{int(r['batch_size'])}" if pd.notna(r["batch_size"]) else "--", ep,
            f"{int(r['n_params']):,}".replace(",", ".") if pd.notna(r["n_params"]) else "--",
            f"{r['time_per_epoch_s']:.2f}" if pd.notna(r["time_per_epoch_s"]) else "--",
            f1, f"{r['binary_f1']:.4f}", notes.get((r["group"], r["config"]), notes.get(r["config"], ""))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (TAB / fname).write_text("\n".join(lines), encoding="utf-8")


def latex_per_class(df: pd.DataFrame, names: list[tuple[str, str, str]], fname: str, caption: str, label: str):
    """names: (group, name, nhãn hiển thị)."""
    rows = {lbl: df[(df["group"] == g) & (df["name"] == n)].iloc[0]["per_class"] for g, n, lbl in names}
    classes = list(SHORT)
    lines = [r"\begin{table}[H]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * len(classes) + "r}", r"\toprule",
             "Mô hình & " + " & ".join(SHORT[c] for c in classes) + r" & Macro \\", r"\midrule"]
    for lbl, pc in rows.items():
        f = [pc[c]["f1-score"] for c in classes]
        lines.append(tex_escape(lbl) + " & " + " & ".join(f"{v:.3f}" for v in f) + f" & {np.mean(f):.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / fname).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- hình
def plot_curves(items: list[tuple[str, str, str]], fname: str, title: str):
    """items: (group, name, nhãn). Trái: loss train (liền) / val (đứt); phải: macro F1."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (g, n, lbl) in enumerate(items):
        h, c = load_history(g, n), f"C{i}"
        axes[0].plot(h["epoch"], h["train_loss"], c, label=f"{lbl} — train")
        axes[0].plot(h["epoch"], h["val_loss"], c, ls="--", label=f"{lbl} — val")
        axes[1].plot(h["epoch"], h["train_macro_f1"], c)
        axes[1].plot(h["epoch"], h["val_macro_f1"], c, ls="--")
        b = h.loc[h["val_macro_f1"].idxmax()]
        axes[1].plot(b["epoch"], b["val_macro_f1"], "o", color=c, ms=6)
    axes[0].set(title="Loss (cross-entropy) — liền: train, đứt: validation", xlabel="epoch")
    axes[1].set(title="Macro F1 — chấm tròn: epoch được chọn (early stopping)", xlabel="epoch")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)


def plot_confusion(r, fname: str, title: str):
    cm = np.array(r["confusion"])
    cmn = cm / cm.sum(1, keepdims=True)
    lbl = [SHORT[c] for c in r["per_class"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(cm)), lbl, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm)), lbl, fontsize=8)
    ax.set(xlabel="Dự đoán", ylabel="Thực tế", title=title)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)


def plot_bars(entries: list[tuple[str, float, float, str]], fname: str, title: str):
    """entries: (nhãn, macro F1, std, nhóm màu)."""
    colors = {"RNN": "#4C72B0", "CNN": "#DD8452", "MLP": "#8172B3", "Cây/khác": "#55A868"}
    entries = sorted(entries, key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(entries) + 1))
    y = np.arange(len(entries))
    ax.barh(y, [e[1] for e in entries], xerr=[0 if pd.isna(e[2]) else e[2] for e in entries],
            color=[colors[e[3]] for e in entries])
    for i, e in enumerate(entries):
        ax.text(e[1] + 0.005, i, f"{e[1]:.3f}", va="center", fontsize=7)
    ax.set_yticks(y, [e[0] for e in entries], fontsize=8)
    ax.set(xlabel="Macro F1 trên tập test", title=title, xlim=(0, max(e[1] for e in entries) + 0.08))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], labels=list(colors), fontsize=7,
              loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)


def family(model: str) -> str:
    return {"rnn": "RNN", "cnn": "CNN", "mlp": "MLP"}.get(model, "Cây/khác")


def best_of(summ, group, model) -> str:
    t = summ[(summ["group"] == group) & (summ["model"] == model)]
    return t.iloc[0]["config"]


def key_models(summ) -> list[tuple[str, str, str]]:
    """Các mô hình tiêu biểu để so sánh F1 từng lớp / ma trận nhầm lẫn."""
    return [("baseline_random", "XGB_current", "XGBoost (A, ngẫu nhiên)"),
            ("A_random", best_of(summ, "A_random", "rnn"), "RNN tốt nhất (A)"),
            ("A_random", best_of(summ, "A_random", "cnn"), "CNN tốt nhất (A)"),
            ("A_random", "A_mlp", "MLP (A)"),
            ("baseline_group", "XGB_current", "XGBoost, không lịch sử"),
            ("B_group", "B_gru_L1", "GRU, L=1 (chỉ cờ lần đầu + Δngày)"),
            ("B_group", best_of(summ, "B_group", "rnn"), "RNN tốt nhất (B)"),
            ("B_group", best_of(summ, "B_group", "cnn"), "CNN tốt nhất (B)"),
            ("baseline_group", "XGB_hist", "XGBoost + lịch sử")]


def make_figures(df, summ):
    exists = lambda g, n: (OUT / g / n / "history.json").exists()
    A, B = "A_random", "B_group"
    plot_curves([(A, n, n) for n in ["A_lstm", "A_gru", "A_lstm_2l", "A_bigru"] if exists(A, n)],
                "curves_A_rnn.png", "Biểu diễn A — RNN: LSTM vs GRU, 1 vs 2 tầng, hai chiều")
    plot_curves([(A, n, n) for n in ["A_cnn_simple", "A_cnn_bn", "A_cnn_do", "A_cnn_bn_do"] if exists(A, n)],
                "curves_A_cnn.png", "Biểu diễn A — CNN đơn giản vs + BatchNorm / Dropout")
    plot_curves([(A, n, n) for n in ["A_gru", "A_gru_nodrop", "A_gru_h128", "A_cnn_3l"] if exists(A, n)],
                "curves_A_reg.png", "Biểu diễn A — dropout và độ phức tạp mô hình")
    plot_curves([(A, n, n) for n in ["A_gru", "A_gru_lr3e-4", "A_gru_bs2048"] if exists(A, n)],
                "curves_A_lr.png", "Biểu diễn A — learning rate và batch size (GRU)")
    plot_curves([(B, n, n) for n in ["B_lstm", "B_gru", "B_cnn_bn_do", "B_mlp"] if exists(B, n)],
                "curves_B.png", "Biểu diễn B — RNN vs CNN vs MLP (MLP chỉ dùng bước hiện tại)")
    plot_curves([(B, n, n) for n in ["B_gru_L1", "B_gru_L4", "B_gru", "B_gru_L32"] if exists(B, n)],
                "curves_B_len.png", "Biểu diễn B — độ dài lịch sử L = 1, 4, 16, 32 (GRU)")

    one = df[df["name"] == df["config"]]
    for g, n, lbl in key_models(summ):
        r = one[(one["group"] == g) & (one["name"] == n)]
        if len(r):
            plot_confusion(r.iloc[0], f"cm_{g}_{n}.png", f"{lbl}: {n}")

    sA = summ[summ["group"] == A]
    entries = [(r["config"], r["macro_f1"], r["macro_f1_std"], family(r["model"])) for _, r in sA.iterrows()]
    entries += [(f"{r['model']} (Bài 1)", r["macro_f1"], np.nan, "Cây/khác") for _, r in bai1_baselines().iterrows()]
    plot_bars(entries, "bars_A.png", "Biểu diễn A (chia ngẫu nhiên): macro F1 trên test")
    sB = summ[summ["group"].isin([B, "baseline_group", "A_group"])]
    entries = [(r["config"] + (" [A, chia địa chỉ]" if r["group"] == "A_group" else ""), r["macro_f1"],
                r["macro_f1_std"], family(r["model"])) for _, r in sB.iterrows()]
    plot_bars(entries, "bars_B.png", "Chia theo địa chỉ: biểu diễn B, đối chứng và XGBoost")


# Nhận xét ngắn cho từng cấu hình (cột "Nhận xét" của các bảng). Khoá: tên cấu hình hoặc (nhóm, tên cấu hình).
NOTES = {
    "A_lstm": "cấu hình gốc RNN", "A_gru": "ít tham số hơn LSTM 19\\%, thấp hơn nhẹ",
    "A_lstm_2l": "thêm tầng: chênh lệch trong mức dao động seed", "A_gru_2l": "thêm tầng: cải thiện rất nhỏ",
    "A_gru_h128": "gấp 3 tham số, không tốt hơn", "A_bigru": "hội tụ sớm hơn, cao nhất nhóm A (trong dao động seed)",
    "A_gru_mean": "mean pooling $\\approx$ hidden cuối", "A_gru_attn": "attention $\\approx$ hidden cuối",
    "A_gru_nodrop": "bỏ dropout: khoảng cách train--val lớn hơn", "A_gru_lr3e-4": "lr nhỏ: hội tụ chậm, underfit",
    "A_gru_bs2048": "batch lớn: ít bước cập nhật, chạm giới hạn 100 epoch",
    "A_cnn_simple": "cấu hình gốc CNN", "A_cnn_bn": "BN: hội tụ nhanh nhất, overfit nhẹ",
    "A_cnn_do": "dropout: train $\\approx$ val, hơi underfit", "A_cnn_bn_do": "điều chuẩn thừa với mô hình nhỏ",
    "A_cnn_gap": "GAP bỏ thông tin ``vị trí = đặc trưng nào''", "A_cnn_3l": "sâu hơn không giúp (chuỗi chỉ 11 bước)",
    "A_cnn_wide": "gấp đôi filters, không tốt hơn", "A_cnn_k5": "kernel 5 $\\approx$ kernel 3",
    "A_cnn_avgpool": "AvgPool $\\approx$ MaxPool", "A_cnn_lr3e-4": "lr nhỏ: underfit",
    "A_cnn_bs2048": "batch lớn: hội tụ chậm hơn", "A_mlp": "không coi là chuỗi --- ngang RNN/CNN",
    "B_cnn_k5": "kernel 5 bao phủ nhiều lần xuất hiện hơn", "B_cnn_3l": "sâu hơn có ích với chuỗi 16 bước",
    "B_gru_2l": "RNN tốt nhất trên B, dropout giữa tầng", "B_cnn_cnt": "+ đặc trưng đếm: không cải thiện",
    "B_cnn_bn_do": "cấu hình gốc CNN-B", "B_gru": "cấu hình gốc RNN-B",
    "B_gru_attn": "attention $\\approx$ hidden cuối", "B_cnn_gmp": "Global Max $\\approx$ GAP",
    "B_lstm": "$\\approx$ GRU", "B_bigru": "hai chiều không giúp (bước cuối quan trọng nhất)",
    "B_gru_cnt": "+ đếm không giúp GRU (đã tự đếm được)", "B_mlp_cnt": "MLP + đếm $\\approx$ GRU có lịch sử",
    "B_gru_mean": "mean làm loãng bước hiện tại", "B_gru_L4": "4 bước đã cho phần lớn lợi ích",
    "B_gru_L32": "dừng sớm (epoch 26), không tốt hơn L=16", "B_cnn_simple": "không BN/Dropout: overfit rất sớm",
    "B_mlp": "chỉ bước hiện tại (có cờ lần đầu, $\\Delta$ngày)", "B_gru_L1": "1 bước: không có chuỗi",
    "XGB_current": "không có thông tin lịch sử", "XGB_count": "+ 3 đặc trưng đếm",
    "XGB_hist": "tốt nhất toàn bộ thực nghiệm",
    ("A_group", "A_gru"): "A trên chia theo địa chỉ", ("A_group", "A_lstm"): "A trên chia theo địa chỉ",
    ("A_group", "A_mlp"): "A trên chia theo địa chỉ", ("A_group", "A_cnn_bn_do"): "A trên chia theo địa chỉ",
}


def make_tables(df, summ):
    sA, sB = summ[summ["group"] == "A_random"], summ[summ["group"] == "B_group"]
    latex_config_table(sA[sA["model"] == "rnn"], NOTES, "tab_A_rnn.tex",
                       "Biểu diễn A — các cấu hình RNN (chia ngẫu nhiên). Ep.: epoch tốt nhất/số epoch đã chạy; "
                       "s/ep.: giây mỗi epoch khi chạy song song (xem Bảng~\\ref{tab:bench} cho tốc độ đo riêng).",
                       "tab:A_rnn")
    latex_config_table(sA[sA["model"] != "rnn"], NOTES, "tab_A_cnn.tex",
                       "Biểu diễn A — các cấu hình CNN và MLP (chia ngẫu nhiên).", "tab:A_cnn")
    latex_config_table(sB, NOTES, "tab_B.tex", "Biểu diễn B — lịch sử địa chỉ (chia theo địa chỉ), $L=16$ nếu không ghi.",
                       "tab:B")
    other = summ[summ["group"].isin(["baseline_group", "A_group"])]
    latex_config_table(other, NOTES, "tab_group_other.tex",
                       "Đối chứng trên cùng cách chia theo địa chỉ: XGBoost với các mức thông tin lịch sử và biểu diễn A.",
                       "tab:group_other")
    latex_per_class(df, key_models(summ), "tab_per_class.tex",
                    "F1 từng lớp trên tập test của các mô hình tiêu biểu (seed 42).", "tab:per_class")
    if (OUT / "benchmark.json").exists():
        bench = json.load(open(OUT / "benchmark.json"))
        lines = [r"\begin{table}[H]\centering\small",
                 r"\caption{Tốc độ huấn luyện đo riêng (GPU không chia sẻ, trung bình 5 epoch sau 1 epoch khởi động, "
                 r"$\approx$126.500 dòng/epoch, batch 512).}\label{tab:bench}",
                 r"\begin{tabular}{llrr}", r"\toprule", r"Cấu hình & Kiến trúc & Tham số & Giây/epoch \\", r"\midrule"]
        for n, b in bench.items():
            r = df[df["name"] == n].iloc[0]
            n_par = f"{b['n_params']:,}".replace(",", ".")
            lines.append(rf"\code{{{tex_escape(n)}}} & {tex_escape(describe(r))} & {n_par} & {b['time_per_epoch_s']:.2f}" + r" \\")
            if n == "A_mlp":
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TAB / "tab_bench.tex").write_text("\n".join(lines), encoding="utf-8")


GAP_MODELS = [("A_random", n) for n in ["A_lstm", "A_gru", "A_gru_nodrop", "A_gru_h128", "A_gru_lr3e-4", "A_gru_bs2048",
                                         "A_cnn_simple", "A_cnn_bn", "A_cnn_do", "A_cnn_bn_do", "A_mlp"]] + \
             [("B_group", n) for n in ["B_gru", "B_gru_2l", "B_gru_L32", "B_cnn_simple", "B_cnn_bn_do", "B_cnn_k5", "B_mlp"]]


def gap_table() -> pd.DataFrame:
    """Chẩn đoán overfit/underfit từ lịch sử huấn luyện (seed 42): so sánh train với val_us tại epoch tốt nhất và epoch cuối."""
    rows = []
    for g, n in GAP_MODELS:
        if not (OUT / g / n / "history.json").exists():
            continue
        h = load_history(g, n)
        b, last = h.loc[h["val_macro_f1"].idxmax()], h.iloc[-1]
        rows.append(dict(group=g, config=n, best_epoch=int(b["epoch"]), epochs=int(last["epoch"]),
                         min_val_loss_epoch=int(h.loc[h["val_loss"].idxmin(), "epoch"]),
                         train_f1_best=b["train_macro_f1"], val_f1_best=b["val_macro_f1"],
                         train_loss_last=last["train_loss"], val_loss_last=last["val_loss"],
                         train_f1_last=last["train_macro_f1"], val_f1_last=last["val_macro_f1"]))
    return pd.DataFrame(rows)


def latex_gap(t: pd.DataFrame, fname: str):
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Chẩn đoán overfit/underfit (seed 42; macro F1 và loss trên tập train đã undersample và \code{val\_us}). "
             r"Ep.\ min loss: epoch có val loss nhỏ nhất; Ep.\ tốt nhất: epoch có val macro F1 lớn nhất (được giữ lại).}\label{tab:gap}",
             r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Epoch} & \multicolumn{3}{c}{Tại epoch tốt nhất} & \multicolumn{3}{c}{Tại epoch cuối} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Cấu hình & min loss & tốt nhất & cuối & F1 train & F1 val & chênh & loss train & loss val & F1 chênh \\",
             r"\midrule"]
    for i, r in t.iterrows():
        if i and r["group"] != t.iloc[i - 1]["group"]:
            lines.append(r"\midrule")
        lines.append(rf"\code{{{tex_escape(r['config'])}}} & {r['min_val_loss_epoch']} & {r['best_epoch']} & {r['epochs']} & "
                     f"{r['train_f1_best']:.3f} & {r['val_f1_best']:.3f} & {r['train_f1_best'] - r['val_f1_best']:+.3f} & "
                     f"{r['train_loss_last']:.3f} & {r['val_loss_last']:.3f} & {r['train_f1_last'] - r['val_f1_last']:+.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / fname).write_text("\n".join(lines), encoding="utf-8")


BY_LEN_MODELS = [("baseline_group", "XGB_current"), ("B_group", "B_mlp"), ("B_group", "B_gru_L1"), ("B_group", "B_gru"),
                 ("B_group", "B_cnn_k5"), ("B_group", "B_mlp_cnt"), ("baseline_group", "XGB_hist")]
LEN_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3--5"), (6, 15, "6--15"), (16, 10**9, "$>$15")]


def by_len_table(prep=None) -> pd.DataFrame:
    """F1 nhị phân trên test (chia theo địa chỉ) theo số lần địa chỉ đã xuất hiện tính đến ngày dự đoán (kể cả hôm nay)."""
    from sklearn.metrics import f1_score
    prep = prep or data.prepare("group", history_len=16)
    te = prep.idx["test"]
    k, true = prep.n_prev[te] + 1, prep.y[te] != prep.white
    rows = [dict(model="Số dòng test", **{lbl: int(((k >= a) & (k <= b)).sum()) for a, b, lbl in LEN_BINS}),
            dict(model="\\% ransomware", **{lbl: 100 * true[(k >= a) & (k <= b)].mean() for a, b, lbl in LEN_BINS})]
    for g, n in BY_LEN_MODELS:
        f = OUT / g / n / "pred_test.npy"
        if f.exists():
            pred = np.load(f) != prep.white
            rows.append(dict(model=n, **{lbl: f1_score(true[(k >= a) & (k <= b)], pred[(k >= a) & (k <= b)])
                                         for a, b, lbl in LEN_BINS}))
    return pd.DataFrame(rows).set_index("model")


def latex_by_len(t: pd.DataFrame, fname: str):
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{F1 nhị phân (ransomware/white) trên tập test theo số lần địa chỉ đã xuất hiện tính đến ngày dự đoán "
             r"(chia theo địa chỉ). Hai hàng đầu: số dòng và tỉ lệ ransomware trong từng nhóm.}\label{tab:by_len}",
             r"\begin{tabular}{l" + "r" * len(t.columns) + "}", r"\toprule",
             "Số lần xuất hiện & " + " & ".join(t.columns) + r" \\", r"\midrule"]
    for i, (n, r) in enumerate(t.iterrows()):
        if i == 0:
            vals = [f"{int(v):,}".replace(",", ".") for v in r]
        elif i == 1:
            vals = [f"{v:.2f}" for v in r]
        else:
            vals = [f"{v:.3f}" for v in r]
        lines.append((n if i < 2 else rf"\code{{{tex_escape(n)}}}") + " & " + " & ".join(vals) + r" \\")
        if i == 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / fname).write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    df = load_results()
    summ = summary_table(df)
    summ.to_csv(TAB / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ[["group", "config", "arch", "macro_f1", "macro_f1_std", "n_seeds", "binary_f1", "n_params",
                "best_epoch", "epochs", "time_per_epoch_s"]].to_string(index=False))
    make_figures(df, summ)
    make_tables(df, summ)
    gap = gap_table(); gap.to_csv(TAB / "gap.csv", index=False); latex_gap(gap, "tab_gap.tex")
    print(gap.round(3).to_string(index=False))
    bl = by_len_table(); bl.to_csv(TAB / "by_len.csv"); latex_by_len(bl, "tab_by_len.tex")
    print(bl.round(3).to_string())
    return df, summ


if __name__ == "__main__":
    main()
