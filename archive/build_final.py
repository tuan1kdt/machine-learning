"""Sinh notebook trình bày hoàn chỉnh (phương án sau cải tiến)."""
import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

# =====================================================================================
md(r"""
# Bài 1 — Phân loại ransomware trên bộ dữ liệu BitcoinHeist

**Bài toán:** cho các đặc trưng giao dịch của một địa chỉ Bitcoin trong một ngày, dự đoán địa chỉ đó là địa chỉ
bình thường (`white`) hay thuộc họ ransomware nào → **phân loại đa lớp (7 lớp)**.

**Dữ liệu:** *BitcoinHeistRansomwareAddressDataset* — UCI Machine Learning Repository (Akcora et al., 2019),
2.916.697 dòng, 2011–2018.

**Nội dung**
1. Đọc & khám phá dữ liệu
2. Định nghĩa nhãn
3. Xây dựng đặc trưng
4. Chia dữ liệu train / validation / test
5. Tiền xử lý & chuẩn hoá
6. Xử lý mất cân bằng dữ liệu (undersampling + hiệu chỉnh trọng số lớp)
7. Các mô hình
8. Huấn luyện & đánh giá trên tập test
9. So sánh các mô hình
10. Phân tích thêm
11. Lưu mô hình
12. Kết luận
""")

code(r"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)
sns.set_theme(style="whitegrid")

DATA_PATH = "BitcoinHeistData.csv"
OUT_DIR = "outputs"
RANDOM_STATE = 42
N_JOBS = -1
N_WHITE = 100_000          # số mẫu 'white' giữ lại khi train (mục 6)
os.makedirs(OUT_DIR, exist_ok=True)

def detect_xgb_device():
    # Dùng GPU nếu có, ngược lại chạy CPU
    try:
        XGBClassifier(device="cuda", n_estimators=2).fit(np.random.rand(50, 3), np.arange(50) % 2)
        return "cuda"
    except Exception:
        return "cpu"
XGB_DEVICE = detect_xgb_device()
print("XGBoost device:", XGB_DEVICE)
""")

# =====================================================================================
md(r"""
## 1. Đọc & khám phá dữ liệu

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `address` | chuỗi | Địa chỉ Bitcoin (định danh) |
| `year`, `day` | số nguyên | Năm (2011–2018) và ngày trong năm (1–365) |
| `length` | số | Độ dài chuỗi giao dịch dài nhất dẫn tới địa chỉ |
| `weight` | số | Lượng "đầu vào" được gộp về địa chỉ |
| `count` | số | Số chuỗi giao dịch dẫn tới địa chỉ |
| `looped` | số | Số giao dịch tách ra rồi gộp lại (dấu hiệu "trộn" tiền) |
| `neighbors` | số | Số giao dịch có địa chỉ là đầu ra |
| `income` | số | Số satoshi nhận được (1 BTC = 10⁸ satoshi) |
| `label` | phân loại | `white` hoặc tên họ ransomware (29 giá trị) |
""")

code(r"""
t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
print(f"Đọc xong trong {time.perf_counter() - t0:.1f}s — {df.shape[0]:,} dòng × {df.shape[1]} cột")
print("Giá trị thiếu:", df.isna().sum().sum(), "| Số địa chỉ khác nhau:", f"{df['address'].nunique():,}")
df.head()
""")

code(r"""
df.describe().T
""")

code(r"""
label_counts = df["label"].value_counts()
fig, ax = plt.subplots(figsize=(10, 7))
label_counts.sort_values().plot.barh(ax=ax, logx=True)
ax.set(title=f"Phân phối nhãn gốc — {label_counts.size} lớp (thang log)", xlabel="Số dòng (log)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/label_distribution.png", dpi=120); plt.show()
label_counts.head(8).to_frame("count").assign(pct=lambda d: 100 * d["count"] / len(df))
""")

code(r"""
ct = pd.crosstab(df["year"], df["label"] != "white").rename(columns={False: "white", True: "ransomware"})
ct["ransom_pct"] = 100 * ct["ransomware"] / ct.sum(axis=1)
ct
""")

md(r"""
**Nhận xét:**
- Mất cân bằng nghiêm trọng: `white` chiếm **98,6%**; nhiều họ ransomware chỉ có 1–100 mẫu.
- Số dòng `white` mỗi năm đúng 365.000 (1.000/ngày) → `white` được lấy mẫu cố định theo ngày.
- Tỉ lệ ransomware thay đổi rất mạnh theo năm (0,001% năm 2018 → 4,1% năm 2016) → `year` là đặc trưng quan trọng.
""")

code(r"""
NUM_BASE = ["length", "weight", "count", "looped", "neighbors", "income"]
fig, axes = plt.subplots(2, len(NUM_BASE), figsize=(20, 6))
sample = df[NUM_BASE].sample(200_000, random_state=RANDOM_STATE)
for i, c in enumerate(NUM_BASE):
    axes[0, i].hist(sample[c], bins=50); axes[0, i].set_title(f"{c} (gốc)")
    axes[1, i].hist(np.log1p(sample[c]), bins=50, color="tab:orange"); axes[1, i].set_title(f"log1p({c})")
plt.suptitle("Các đặc trưng số lệch rất mạnh → dùng log1p trước khi chuẩn hoá")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/numeric_distributions.png", dpi=120); plt.show()
""")

# =====================================================================================
md(r"""
## 2. Định nghĩa nhãn

Với 29 lớp, các lớp chỉ có vài mẫu không thể học và đánh giá có ý nghĩa. Gộp thành **7 lớp**:
`white`, 5 họ ransomware lớn nhất, và `otherRansom` (các họ còn lại).
""")

code(r"""
WHITE = "white"
TOP_FAMILIES = label_counts.drop(WHITE).head(5).index.tolist()
df["target"] = np.where(df["label"] == WHITE, WHITE,
                np.where(df["label"].isin(TOP_FAMILIES), df["label"], "otherRansom"))
CLASSES = sorted(df["target"].unique())          # thứ tự lớp cố định, dùng xuyên suốt
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
K = len(CLASSES)
df["target"].value_counts().to_frame("count").assign(pct=lambda d: 100 * d["count"] / len(df))
""")

# =====================================================================================
md(r"""
## 3. Xây dựng đặc trưng

| Nhóm | Đặc trưng | Kiểu |
|---|---|---|
| Gốc | `length, weight, count, looped, neighbors, income` | numerical |
| Thời gian | `year`, `month`, `weekday` (tạo từ `year` + `day`) | categorical |
| **Mới** | `income_tz` — số chữ số 0 ở cuối của `income` (độ "tròn" của số tiền) | numerical |
| **Mới** | `income_per_count`, `income_per_neighbor`, `weight_per_count`, `looped_ratio` | numerical |

`address` bị loại vì là định danh, không phải đặc trưng.
""")

code(r"""
date = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["day"] - 1, unit="D")
df["month"] = date.dt.month
df["weekday"] = date.dt.dayofweek

inc_int = df["income"].round().astype("int64")
df["income_tz"] = sum((inc_int % 10**k == 0).astype("int8") for k in range(1, 13))
df["income_per_count"] = df["income"] / df["count"]
df["income_per_neighbor"] = df["income"] / df["neighbors"].clip(lower=1)
df["weight_per_count"] = df["weight"] / df["count"]
df["looped_ratio"] = df["looped"] / df["count"]

NUM = NUM_BASE + ["income_tz", "income_per_count", "income_per_neighbor", "weight_per_count", "looped_ratio"]
CAT = ["year", "month", "weekday"]
X = df[NUM + CAT]
y = df["target"].map(CLS_IDX)                      # nhãn dạng số 0..K-1 theo thứ tự CLASSES
print("Số đặc trưng:", len(NUM), "numerical +", len(CAT), "categorical")
""")

code(r"""
tz = pd.crosstab(df["income_tz"].clip(upper=10), np.where(df["target"] == WHITE, "white", "ransomware"),
                 normalize="columns")
ax = tz.plot.bar(figsize=(10, 4), title="Phân phối độ 'tròn' của income (số chữ số 0 ở cuối, satoshi)")
ax.set(xlabel="income_tz (≥10 gộp chung)", ylabel="Tỉ lệ trong nhóm")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/income_roundness.png", dpi=120); plt.show()
""")

# =====================================================================================
md(r"""
## 4. Chia dữ liệu train / validation / test

```
Toàn bộ (2,92 triệu) ──80%──► train_full ──80%──► TRAIN       : huấn luyện mô hình
                     │                    └─20%──► VALIDATION  : chọn cấu hình, hiệu chỉnh trọng số lớp
                     └─20%──► TEST (583 nghìn)                 : chỉ dùng để báo cáo kết quả cuối
```
Mọi phép chia đều **stratified** theo nhãn. Validation và test giữ **phân phối thật** (98,6% white).
""")

code(r"""
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE)

sizes = pd.concat([s.map(dict(enumerate(CLASSES))).value_counts().rename(n)
                   for n, s in [("train", y_tr), ("validation", y_val), ("test", y_test)]], axis=1)
sizes.loc["TỔNG"] = sizes.sum()
sizes
""")

# =====================================================================================
md(r"""
## 5. Tiền xử lý & chuẩn hoá

- **Numerical:** `log1p` (giảm độ lệch) → `StandardScaler` (trung bình 0, độ lệch chuẩn 1).
- **Categorical:** `OneHotEncoder` (`year`, `month`, `weekday`).
- Đặt trong `Pipeline` + `ColumnTransformer` → scaler/encoder **chỉ fit trên dữ liệu train**, tránh rò rỉ dữ liệu.
""")

code(r"""
def make_preprocessor(num_cols=NUM, cat_cols=CAT):
    return ColumnTransformer([
        ("num", Pipeline([("log1p", FunctionTransformer(np.log1p)),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

make_preprocessor()
""")

# =====================================================================================
md(r"""
## 6. Xử lý mất cân bằng dữ liệu

Dùng **hai kỹ thuật bổ sung cho nhau**, tác động ở hai thời điểm khác nhau:

| | (A) Undersampling | (B) Hiệu chỉnh trọng số lớp |
|---|---|---|
| Khi nào | **Trước khi train** | **Sau khi train**, lúc dự đoán |
| Làm gì | Chỉ giữ `N_WHITE` = 100.000 mẫu `white` + toàn bộ ransomware trong tập train | Dự đoán `argmax(p_k · w_k)` thay vì `argmax(p_k)`; `w` được tìm để tối đa macro F1 trên **validation** |
| Mục đích | Mô hình tập trung học các lớp ransomware; train nhanh hơn | Sửa lệch do (A): mô hình được train trên dữ liệu ~25% ransomware nên đánh giá xác suất ransomware cao hơn thực tế (~1,4%) |

**Ví dụ (B):** mô hình trả về `p(white)=0,40`, `p(Locky)=0,60`. Với `w_white=1`, `w_Locky=0,08`:
0,40 > 0,048 → dự đoán `white` thay vì báo động nhầm `Locky`. Mô hình không đổi, chỉ đổi ngưỡng quyết định.

Trọng số `w` được tìm bằng *coordinate ascent*: lần lượt thử từng `w_k` trên lưới `e^-5 … e^5`, giữ giá trị làm
macro F1 tăng, lặp 3 vòng.
""")

code(r"""
def undersample(X_, y_, n_white=N_WHITE, seed=RANDOM_STATE):
    # Giữ n_white mẫu white (None = giữ tất cả) + toàn bộ ransomware
    if n_white is None:
        return X_, y_
    w_idx = y_.index[y_ == CLS_IDX[WHITE]]
    keep = np.concatenate([np.random.RandomState(seed).choice(w_idx, n_white, replace=False),
                           y_.index[y_ != CLS_IDX[WHITE]]])
    return X_.loc[keep], y_.loc[keep]

def macro_f1_fast(y_true, y_pred):
    # Macro F1 tính từ ma trận nhầm lẫn (nhanh hơn sklearn, dùng trong vòng tìm trọng số)
    cm = np.bincount(np.asarray(y_true) * K + np.asarray(y_pred), minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0)))

def tune_class_weights(P, y_true, passes=3):
    w = np.ones(K); grid = np.exp(np.linspace(-5, 5, 41))
    best = macro_f1_fast(y_true, P.argmax(1))
    for _ in range(passes):
        for k in range(K):
            for g in grid:
                w2 = w.copy(); w2[k] = g
                s = macro_f1_fast(y_true, (P * w2).argmax(1))
                if s > best + 1e-9:
                    best, w = s, w2
    return w

X_fit, y_fit = undersample(X_tr, y_tr)
print(f"Train sau undersampling: {len(X_fit):,} dòng — ransomware chiếm {100 * (y_fit != CLS_IDX[WHITE]).mean():.1f}%"
      f" (thực tế: {100 * (y_tr != CLS_IDX[WHITE]).mean():.1f}%)")
""")

md(r"""
### 6.1 Kiểm chứng trên validation

Dùng HistGradientBoosting mặc định. Validation được chia đôi: nửa A để tìm trọng số `w`, nửa B để chấm điểm
(tránh chấm trên chính dữ liệu dùng để tìm `w`).
""")

code(r"""
half = np.arange(len(X_val)) % 2 == 0
y_val_np = y_val.to_numpy()

def val_check(n_white, num_cols=NUM):
    Xf, yf = undersample(X_tr, y_tr, n_white)
    pipe = Pipeline([("pre", make_preprocessor(num_cols)),
                     ("clf", HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE))])
    pipe.fit(Xf[num_cols + CAT], yf)
    P = pipe.predict_proba(X_val[num_cols + CAT])
    w = tune_class_weights(P[half], y_val_np[half])
    return {"Không hiệu chỉnh (argmax p)": macro_f1_fast(y_val_np[~half], P[~half].argmax(1)),
            "Có hiệu chỉnh (argmax p·w)": macro_f1_fast(y_val_np[~half], (P[~half] * w).argmax(1))}

n_white_all = int((y_tr == CLS_IDX[WHITE]).sum())
ablation = pd.DataFrame({nw or n_white_all: val_check(nw) for nw in [100_000, 200_000, 500_000, 1_000_000, None]
                         if nw is None or nw < n_white_all}).T
ablation.index.name = "Số mẫu white khi train"
display(ablation)

ax = ablation.plot(marker="o", logx=True, figsize=(9, 4), title="Macro F1 trên validation")
ax.set_xlabel("Số mẫu white trong tập train (log)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/imbalance_ablation.png", dpi=120); plt.show()
""")

code(r"""
feat_check = pd.DataFrame({"6 đặc trưng gốc": val_check(N_WHITE, NUM_BASE),
                           "11 đặc trưng (gốc + mới)": val_check(N_WHITE, NUM)}).T
feat_check
""")

md(r"""
**Nhận xét:**
- Hiệu chỉnh trọng số lớp (B) cải thiện macro F1 ở **mọi** mức undersampling.
- Nếu **không** hiệu chỉnh, giảm số mẫu white xuống 100k lại làm kết quả kém đi (mô hình nghiêng về ransomware);
  khi **có** hiệu chỉnh, 100k cho kết quả tốt nhất → (A) và (B) phải đi cùng nhau.
- Giữ quá nhiều white: loss bị lớp white chi phối, early stopping dừng trước khi học được các lớp ransomware.
- 5 đặc trưng mới giúp tăng thêm macro F1.
""")

# =====================================================================================
md(r"""
## 7. Các mô hình

| Mô hình | Nguyên lý | Siêu tham số |
|---|---|---|
| Logistic Regression | Tuyến tính, hàm softmax cho đa lớp | `max_iter=2000` |
| KNN | Bỏ phiếu theo k láng giềng gần nhất (khoảng cách Euclid) | `k=15`, trọng số theo khoảng cách |
| Random Forest | Bagging nhiều cây quyết định, mỗi cây học trên mẫu bootstrap | 300 cây, `max_features="sqrt"` |
| HistGradientBoosting | Boosting: cây sau sửa lỗi cây trước; chia giá trị theo histogram | lr 0,03, 255 lá, L2=10, early stopping |
| LightGBM | Gradient boosting, cây phát triển theo lá (leaf-wise) | lr 0,05, 31 lá, 600 cây, subsample 0,7 |
| XGBoost | Gradient boosting có regularization, chạy trên **GPU** | lr 0,05, độ sâu 12, 300 cây, colsample 0,6 |

Siêu tham số của 4 mô hình cây được chọn bằng random search (34 cấu hình) theo macro F1 trên validation —
xem `experiments.py`.
""")

code(r"""
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=N_JOBS),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=1,
                                           n_jobs=N_JOBS, random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=255, min_samples_leaf=20,
                                                           l2_regularization=10.0, max_features=0.8,
                                                           max_iter=2000, early_stopping=True, validation_fraction=0.1,
                                                           n_iter_no_change=30, random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                               subsample=0.7, subsample_freq=1, colsample_bytree=1.0, reg_lambda=1.0,
                               n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=12, min_child_weight=1,
                             subsample=0.9, colsample_bytree=0.6, reg_lambda=5.0,
                             tree_method="hist", device=XGB_DEVICE, eval_metric="mlogloss", random_state=RANDOM_STATE),
}
""")

# =====================================================================================
md(r"""
## 8. Huấn luyện & đánh giá trên tập test

Với mỗi mô hình:
1. **Train** trên tập train đã undersample (`N_WHITE` = 100k) → đo *thời gian training*.
2. Dự đoán xác suất trên validation → tìm trọng số lớp `w`.
3. **Test:** `argmax(p · w)` trên tập test → đo *thời gian testing* (gồm tiền xử lý + dự đoán + nhân trọng số).
""")

code(r"""
def log_progress(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def summarize(name, y_true, y_pred, t_train, t_test):
    rep = classification_report(y_true, y_pred, labels=range(K), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return rep, {"model": name, "accuracy": rep["accuracy"],
                 "weighted_precision": rep["weighted avg"]["precision"],
                 "weighted_recall": rep["weighted avg"]["recall"],
                 "weighted_f1": rep["weighted avg"]["f1-score"],
                 "macro_f1": rep["macro avg"]["f1-score"],
                 "binary_f1_ransom": f1_score(y_true != CLS_IDX[WHITE], y_pred != CLS_IDX[WHITE]),
                 "train_time_s": t_train, "test_time_s": t_test}

pipes, weights, preds, P_val, P_test, reports, rows = {}, {}, {}, {}, {}, {}, []
for name, clf in MODELS.items():
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])
    log_progress(f"[{name}] train ...")
    t0 = time.perf_counter(); pipe.fit(X_fit[NUM + CAT], y_fit); t_train = time.perf_counter() - t0

    P_val[name] = pipe.predict_proba(X_val[NUM + CAT])
    weights[name] = tune_class_weights(P_val[name], y_val_np)

    t0 = time.perf_counter()
    P_test[name] = pipe.predict_proba(X_test[NUM + CAT])
    preds[name] = (P_test[name] * weights[name]).argmax(1)
    t_test = time.perf_counter() - t0
    log_progress(f"[{name}] train {t_train:.1f}s | test {t_test:.1f}s")

    pipes[name] = pipe
    reports[name], s = summarize(name, y_test, preds[name], t_train, t_test)
    rows.append(s)
    print(f"\n===== {name} =====  train {t_train:.1f}s | test {t_test:.1f}s")
    print(classification_report(y_test, preds[name], labels=range(K), target_names=CLASSES, digits=4, zero_division=0))
""")

md(r"""
### 8.1 Ensemble

Trung bình xác suất của 3 mô hình boosting (HistGradientBoosting, LightGBM, XGBoost), rồi hiệu chỉnh trọng số lớp
như trên. Thời gian của ensemble = tổng thời gian của 3 mô hình thành phần.
""")

code(r"""
ENS = ["HistGradientBoosting", "LightGBM", "XGBoost"]
P_val_ens = np.mean([P_val[m] for m in ENS], axis=0)
weights["Ensemble"] = tune_class_weights(P_val_ens, y_val_np)
preds["Ensemble"] = (np.mean([P_test[m] for m in ENS], axis=0) * weights["Ensemble"]).argmax(1)
t_tr = sum(r["train_time_s"] for r in rows if r["model"] in ENS)
t_te = sum(r["test_time_s"] for r in rows if r["model"] in ENS)
reports["Ensemble"], s = summarize("Ensemble", y_test, preds["Ensemble"], t_tr, t_te)
rows.append(s)
print(classification_report(y_test, preds["Ensemble"], labels=range(K), target_names=CLASSES, digits=4, zero_division=0))
""")

# =====================================================================================
md(r"""
## 9. So sánh các mô hình

### 9.1 Độ đo tổng hợp & thời gian

- **Weighted F1:** trung bình F1 các lớp theo số mẫu → bị lớp `white` (98,6%) chi phối.
- **Macro F1:** trung bình F1 các lớp *không* trọng số → phản ánh đúng khả năng nhận diện các họ ransomware.
- **Binary F1:** F1 của bài toán "có phải ransomware không" (bất kể họ nào).
- Mốc so sánh: mô hình *luôn đoán white*.
""")

code(r"""
summary = pd.DataFrame(rows).set_index("model")
_, dummy = summarize("Luôn đoán white", y_test, np.full(len(y_test), CLS_IDX[WHITE]), 0, 0)
summary_show = pd.concat([summary, pd.DataFrame([dummy]).set_index("model")])
summary.to_csv(f"{OUT_DIR}/summary.csv")
(summary_show.style
    .format({c: "{:.4f}" for c in summary.columns if not c.endswith("_s")} | {"train_time_s": "{:.2f}", "test_time_s": "{:.2f}"})
    .background_gradient(cmap="Greens", subset=["macro_f1", "binary_f1_ransom"])
    .background_gradient(cmap="Reds", subset=["train_time_s", "test_time_s"]))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
summary[["accuracy", "weighted_f1", "macro_f1", "binary_f1_ransom"]].plot.bar(ax=axes[0])
axes[0].axhline(dummy["macro_f1"], ls="--", c="gray", lw=1, label="macro F1 của 'luôn đoán white'")
axes[0].set(title="Độ đo trên tập test", ylim=(0, 1.05), xlabel="")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
axes[0].tick_params(axis="x", rotation=20)
summary.drop(index="Ensemble")[["train_time_s", "test_time_s"]].plot.bar(ax=axes[1], logy=True)
axes[1].set(title="Thời gian training / testing (giây, thang log)", xlabel="")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=120); plt.show()
""")

md("### 9.2 Precision / Recall / F1 của từng lớp")

code(r"""
per_class = pd.concat({m: pd.DataFrame({c: reports[m][c] for c in CLASSES}).T[["precision", "recall", "f1-score"]]
                       for m in reports}, names=["model", "class"])
per_class.to_csv(f"{OUT_DIR}/per_class.csv")
tbl = per_class.unstack("model").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
tbl = tbl[[(m, k) for m in reports for k in ["precision", "recall", "f1-score"]]]
tbl.style.format("{:.3f}").background_gradient(cmap="Greens", axis=None)
""")

code(r"""
f1_pivot = per_class["f1-score"].unstack("model")[list(reports)]
fig, ax = plt.subplots(figsize=(14, 5))
f1_pivot.plot.bar(ax=ax)
ax.set(title="F1-score từng lớp trên tập test", ylabel="F1", xlabel="")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/f1_per_class.png", dpi=120); plt.show()
""")

code(r"""
best_name = summary["macro_f1"].idxmax()
best_single = summary.drop(index="Ensemble")["macro_f1"].idxmax()
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, m in zip(axes, [best_single, "LogisticRegression"]):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[m], labels=range(K), display_labels=CLASSES,
                                            normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{m} — ma trận nhầm lẫn (chuẩn hoá theo hàng = recall)")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=120); plt.show()
print("Tốt nhất theo macro F1:", best_name, "| Mô hình đơn tốt nhất:", best_single)
""")

# =====================================================================================
md(r"""
## 10. Phân tích thêm

### 10.1 Phát hiện ransomware (bài toán nhị phân)
""")

code(r"""
to_bin = lambda a: np.where(np.asarray(a) == CLS_IDX[WHITE], "white", "ransomware")
print(f"{best_single} — white vs ransomware")
print(classification_report(to_bin(y_test), to_bin(preds[best_single]), digits=4))
""")

md("### 10.2 Vai trò của đặc trưng `year`")

code(r"""
cat_ny = ["month", "weekday"]
pipe_ny = Pipeline([("pre", make_preprocessor(NUM, cat_ny)), ("clf", clone(MODELS[best_single]))])
pipe_ny.fit(X_fit[NUM + cat_ny], y_fit)
w_ny = tune_class_weights(pipe_ny.predict_proba(X_val[NUM + cat_ny]), y_val_np)
p_ny = (pipe_ny.predict_proba(X_test[NUM + cat_ny]) * w_ny).argmax(1)
_, s_ny = summarize(f"{best_single} (bỏ year)", y_test, p_ny, 0, 0)
pd.DataFrame([summary.loc[best_single].rename(best_single), pd.Series(s_ny).drop("model").rename(s_ny["model"])]
            )[["accuracy", "weighted_f1", "macro_f1", "binary_f1_ransom"]]
""")

md(r"""
### 10.3 Kiểm tra rò rỉ dữ liệu theo `address`

Cùng một địa chỉ có thể xuất hiện nhiều ngày. Với phép chia ngẫu nhiên, một địa chỉ ransomware có thể có mặt ở cả
train và test → kết quả có thể lạc quan. Kiểm tra bằng cách chia theo **nhóm địa chỉ** (`GroupShuffleSplit`):
mỗi địa chỉ chỉ nằm ở train *hoặc* test.
""")

code(r"""
addr = df["address"]
in_train = addr.loc[X_test.index].isin(set(addr.loc[X_train_full.index]))
is_ransom = y_test != CLS_IDX[WHITE]
display(pd.Series({"Tất cả dòng test": in_train.mean(), "Dòng test ransomware": in_train[is_ransom].mean(),
                   "Dòng test white": in_train[~is_ransom].mean()},
                  name="Tỉ lệ có address xuất hiện trong train").to_frame())

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
trf_g, te_g = next(gss.split(X, y, groups=addr))
tr_g, va_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                  .split(trf_g, groups=addr.iloc[trf_g]))
tr_g, va_g, te_g = df.index[trf_g[tr_g]], df.index[trf_g[va_g]], df.index[te_g]
assert not set(addr.loc[tr_g]) & set(addr.loc[te_g])

Xg, yg = undersample(X.loc[tr_g], y.loc[tr_g])
pipe_g = Pipeline([("pre", make_preprocessor()), ("clf", clone(MODELS[best_single]))]).fit(Xg[NUM + CAT], yg)
w_g = tune_class_weights(pipe_g.predict_proba(X.loc[va_g, NUM + CAT]), y.loc[va_g].to_numpy())
p_g = (pipe_g.predict_proba(X.loc[te_g, NUM + CAT]) * w_g).argmax(1)
_, s_g = summarize(f"{best_single} — chia theo address", y.loc[te_g], p_g, 0, 0)
pd.DataFrame([summary.loc[best_single].rename(f"{best_single} — chia ngẫu nhiên"),
              pd.Series(s_g).drop("model").rename(s_g["model"])])[["accuracy", "weighted_f1", "macro_f1", "binary_f1_ransom"]]
""")

# =====================================================================================
md(r"""
## 11. Lưu mô hình (dùng cho Bài 3 — demo Streamlit)

Lưu **cả 6 mô hình** và ensemble vào `outputs/models/`. Mỗi file chứa pipeline (tiền xử lý + mô hình), trọng số lớp
đã hiệu chỉnh và thông tin đặc trưng. Cách dự đoán:
`CLASSES[(pipeline.predict_proba(X[num + cat]) * class_weights).argmax(1)]`.
""")

code(r"""
MODEL_DIR = f"{OUT_DIR}/models"
os.makedirs(MODEL_DIR, exist_ok=True)
META = {"classes": CLASSES, "num_features": NUM, "cat_features": CAT, "top_families": TOP_FAMILIES}

# XGBoost: chuyển sang CPU để chạy được trên máy không có GPU
pipes["XGBoost"].named_steps["clf"].set_params(device="cpu")

files = {}
for name, pipe in pipes.items():
    path = f"{MODEL_DIR}/{name}.joblib"
    joblib.dump({"pipeline": pipe, "class_weights": weights[name], "model_name": name,
                 "metrics_test": summary.loc[name].to_dict(), **META}, path, compress=3)
    files[name] = path

# Ensemble: lưu tên các mô hình thành phần + trọng số lớp riêng của ensemble
path = f"{MODEL_DIR}/Ensemble.joblib"
joblib.dump({"components": ENS, "class_weights": weights["Ensemble"], "model_name": "Ensemble",
             "metrics_test": summary.loc["Ensemble"].to_dict(), **META}, path, compress=3)
files["Ensemble"] = path

# Bản sao mô hình đơn tốt nhất (tương thích với các bước trước)
joblib.dump({"pipeline": pipes[best_single], "class_weights": weights[best_single], "model_name": best_single,
             **META}, f"{OUT_DIR}/best_model.joblib", compress=3)

pd.DataFrame({"file": files, "size_MB": {k: os.path.getsize(v) / 1e6 for k, v in files.items()}})
""")

code(r"""
# Kiểm tra: nạp lại từng file và so dự đoán với kết quả đã tính ở mục 8 (trên 50.000 dòng test đầu tiên)
X_chk = X_test.iloc[:50_000][NUM + CAT]
loaded = {name: joblib.load(files[name]) for name in pipes}
P_chk = {}
check = {}
for name, art in loaded.items():
    P_chk[name] = art["pipeline"].predict_proba(X_chk)
    check[name] = ((P_chk[name] * art["class_weights"]).argmax(1) == preds[name][:50_000]).mean()
ens = joblib.load(files["Ensemble"])
p_ens = (np.mean([P_chk[m] for m in ens["components"]], axis=0) * ens["class_weights"]).argmax(1)
check["Ensemble"] = (p_ens == preds["Ensemble"][:50_000]).mean()
pd.Series(check, name="Tỉ lệ dự đoán trùng khớp sau khi nạp lại").to_frame()
""")

md(r"""
## 12. Kết luận

**Kết quả trên tập test (583.340 dòng, phân phối thật):**

| Mô hình | Accuracy | Weighted F1 | Macro F1 | Binary F1 | Train (s) | Test (s) |
|---|---|---|---|---|---|---|
| Luôn đoán `white` (mốc) | 0,986 | 0,979 | 0,142 | 0,000 | – | – |
| Logistic Regression | 0,944 | 0,961 | 0,347 | 0,193 | 5,9 | 0,3 |
| KNN | 0,975 | 0,978 | 0,416 | 0,345 | 0,1 | 14,0 |
| Random Forest | 0,983 | 0,984 | 0,555 | 0,470 | 2,2 | 2,0 |
| HistGradientBoosting | 0,985 | 0,985 | 0,569 | 0,503 | 33,9 | 8,0 |
| LightGBM | 0,984 | 0,985 | 0,569 | 0,489 | 6,0 | 8,5 |
| **XGBoost (GPU)** | **0,985** | **0,985** | **0,575** | **0,507** | 13,1 | **0,7** |
| Ensemble (HGB + LightGBM + XGBoost) | 0,984 | 0,985 | **0,576** | 0,503 | 53,0 | 17,2 |

**Nhận xét:**
1. **Các mô hình boosting tốt nhất.** XGBoost là mô hình đơn tốt nhất (macro F1 0,575) và dự đoán nhanh nhất trong
   nhóm (0,7s nhờ GPU). Ensemble chỉ nhỉnh hơn rất ít (0,576) nhưng tốn gấp ~4 lần thời gian → chọn **XGBoost**.
2. **Mô hình tuyến tính và KNN yếu hơn rõ rệt**: ranh giới giữa các lớp phi tuyến, và các họ ransomware có đặc
   trưng chồng lấn nhau.
3. **Thời gian:** KNN gần như không tốn thời gian train (chỉ lưu dữ liệu) nhưng chậm nhất khi test (14s) vì phải
   tính khoảng cách tới toàn bộ tập train; Logistic Regression ngược lại. HistGradientBoosting train lâu nhất (34s)
   do cấu hình 255 lá, learning rate nhỏ.
4. **Accuracy không phù hợp** với dữ liệu mất cân bằng: mô hình "luôn đoán white" có accuracy cao nhất (0,986) nhưng
   không phát hiện được ransomware nào. Macro F1 mới phản ánh đúng chất lượng.
5. **Xử lý mất cân bằng là yếu tố quan trọng nhất:** trên validation, không hiệu chỉnh thì macro F1 tốt nhất chỉ
   0,49 (giữ 500k white); undersampling 100k white + hiệu chỉnh trọng số lớp đạt 0,544. So với phiên bản ban đầu (chỉ undersampling 200k, không hiệu chỉnh, không
   đặc trưng mới, không tinh chỉnh), macro F1 trên test tăng **0,456 → 0,575**.
6. **Theo từng lớp (XGBoost):** nhận diện tốt CryptXXX (F1 0,86), Locky (0,73), Cerber (0,63); khó với CryptoWall
   (0,42), CryptoLocker (0,28) và `otherRansom` (0,11 — gộp nhiều họ hiếm, ít mẫu).
7. **`year` rất quan trọng:** bỏ `year`, macro F1 giảm 0,575 → 0,444.
8. **Độ tin cậy:** 54,7% dòng ransomware trong test có địa chỉ đã xuất hiện trong train, nhưng khi chia theo nhóm
   địa chỉ, macro F1 chỉ giảm 0,575 → 0,565 → mô hình không dựa vào việc "nhớ" địa chỉ.

**Hạn chế & hướng phát triển:** ở mức nhị phân, precision ransomware ~47% và recall ~55% — vẫn còn báo nhầm. Bộ dữ
liệu chỉ có 6 đặc trưng đồ thị; để cải thiện thêm cần bổ sung thông tin (giá BTC theo ngày để quy đổi `income` ra
USD, đặc trưng tổng hợp nhiều ngày của cùng một địa chỉ, cấu trúc đồ thị giao dịch đầy đủ).
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
