"""Đọc dữ liệu, tạo nhãn & đặc trưng, chia dữ liệu và dựng 2 cách biểu diễn chuỗi.

- Biểu diễn A ("feature-as-sequence"): mỗi dòng → chuỗi 11 bước, bước i là đặc trưng số thứ i.
- Biểu diễn B ("address history"): mỗi dòng → chuỗi tối đa L bản ghi gần nhất của CÙNG address
  (tính đến ngày hiện tại, chỉ dùng quá khứ), sắp theo thời gian, bước cuối là dòng hiện tại.

Nhãn, đặc trưng, cách chia và undersampling giống hệt Bài 1 (bai1.ipynb) để so sánh công bằng.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "BitcoinHeistData.csv"
RS = 42
N_WHITE = 100_000                 # số mẫu white giữ lại trong tập train (như Bài 1)
WHITE = "white"
NUM = ["length", "weight", "count", "looped", "neighbors", "income",
       "income_tz", "income_per_count", "income_per_neighbor", "weight_per_count", "looped_ratio"]
YEARS = list(range(2011, 2019))   # one-hot cố định: year (8) + month (12) + weekday (7) = 27 chiều


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH)
    top = df["label"].value_counts().drop(WHITE).head(5).index
    df["target"] = np.where(df["label"] == WHITE, WHITE,
                   np.where(df["label"].isin(top), df["label"], "otherRansom"))
    classes = sorted(df["target"].unique())
    df["y"] = pd.Categorical(df["target"], categories=classes).codes.astype(np.int64)

    date = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["day"] - 1, unit="D")
    df["month"], df["weekday"] = date.dt.month, date.dt.dayofweek
    df["t"] = (date - pd.Timestamp("2011-01-01")).dt.days            # ngày tuyệt đối, dùng để sắp lịch sử

    inc_int = df["income"].round().astype("int64")
    df["income_tz"] = sum((inc_int % 10**k == 0).astype("int8") for k in range(1, 13))
    df["income_per_count"] = df["income"] / df["count"]
    df["income_per_neighbor"] = df["income"] / df["neighbors"].clip(lower=1)
    df["weight_per_count"] = df["weight"] / df["count"]
    df["looped_ratio"] = df["looped"] / df["count"]
    return df, classes


def split(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """train / validation / test (64/16/20). 'random' = phân tầng như Bài 1; 'group' = theo address."""
    if kind == "random":
        trf, te = train_test_split(df.index.to_numpy(), test_size=0.2, stratify=df["y"], random_state=RS)
        tr, va = train_test_split(trf, test_size=0.2, stratify=df["y"].to_numpy()[trf], random_state=RS)
        return tr, va, te
    groups = df["address"].to_numpy()
    trf, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS).split(df, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS).split(trf, groups=groups[trf]))
    return trf[tr], trf[va], te


def undersample(idx: np.ndarray, y: np.ndarray, white: int, n_white: int, seed: int = RS) -> np.ndarray:
    """Giữ n_white dòng white (chọn ngẫu nhiên) + toàn bộ ransomware — giống hàm undersample của Bài 1."""
    w_idx = idx[y[idx] == white]
    return np.concatenate([np.random.RandomState(seed).choice(w_idx, n_white, replace=False), idx[y[idx] != white]])


def address_history(df: pd.DataFrame, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """hist[i] = chỉ số (theo df) của tối đa L lần xuất hiện gần nhất của address dòng i, tính cả dòng i,
    theo thứ tự thời gian, đệm -1 ở cuối. gap[i] = số ngày kể từ lần xuất hiện trước (-1 nếu là lần đầu).
    n_prev[i] = số lần address đã xuất hiện trước dòng i (không giới hạn bởi L)."""
    code = pd.factorize(df["address"])[0]
    order = np.lexsort((df.index.to_numpy(), df["t"].to_numpy(), code))      # sắp theo address, rồi thời gian
    c, t = code[order], df["t"].to_numpy()[order]
    p = np.arange(len(order))
    first = np.r_[True, c[1:] != c[:-1]]
    start = np.maximum.accumulate(np.where(first, p, 0))                     # vị trí đầu nhóm address
    n = np.minimum(p - start + 1, L)                                         # độ dài lịch sử (≤ L)
    offs = np.arange(L)
    cand = p[:, None] - n[:, None] + 1 + offs
    hist_sorted = np.where(offs < n[:, None], order[np.clip(cand, 0, len(order) - 1)], -1).astype(np.int32)
    gap_sorted = np.where(first, -1, t - np.r_[t[0], t[:-1]])

    hist = np.empty_like(hist_sorted); hist[order] = hist_sorted
    gap = np.empty(len(df), dtype=np.int64); gap[order] = gap_sorted
    n_prev = np.empty(len(df), dtype=np.int64); n_prev[order] = p - start
    return hist, gap, n_prev


@dataclass
class Prepared:
    classes: list[str]
    y: np.ndarray            # (N,) nhãn 0..K-1
    num: np.ndarray          # (N, 11) đặc trưng số đã log1p + chuẩn hoá (fit trên tập train)
    side: np.ndarray         # (N, 27) one-hot year/month/weekday
    idx: dict                # fit (train đã undersample), val, val_us (val undersample cùng tỉ lệ), test
    steps: np.ndarray | None = None   # (N, 13) đặc trưng của 1 bước trong biểu diễn B
    hist: np.ndarray | None = None    # (N, L) chỉ số các bước lịch sử, -1 = đệm
    gap: np.ndarray | None = None     # (N,) số ngày từ lần xuất hiện trước, -1 = lần đầu
    n_prev: np.ndarray | None = None  # (N,) số lần address đã xuất hiện trước đó

    @property
    def white(self) -> int:
        return self.classes.index(WHITE)


def prepare(split_kind: str = "random", history_len: int | None = None) -> Prepared:
    df, classes = load_frame()
    y = df["y"].to_numpy()
    white = classes.index(WHITE)
    tr, va, te = split(df, split_kind)
    fit = undersample(tr, y, white, N_WHITE)
    # val_us: tập validation với tỉ lệ white giống tập fit → đường cong train/val so sánh được với nhau
    n_white_val = int(round(N_WHITE * (y[va] == white).sum() / (y[tr] == white).sum()))
    val_us = undersample(va, y, white, n_white_val)

    raw = np.log1p(df[NUM].to_numpy(np.float64))
    mu, sd = raw[fit].mean(0), raw[fit].std(0)                               # chỉ fit trên dữ liệu train
    num = ((raw - mu) / sd).astype(np.float32)

    side = np.concatenate([
        (df["year"].to_numpy()[:, None] == np.array(YEARS)).astype(np.float32),
        (df["month"].to_numpy()[:, None] == np.arange(1, 13)).astype(np.float32),
        (df["weekday"].to_numpy()[:, None] == np.arange(7)).astype(np.float32)], axis=1)

    prep = Prepared(classes, y, num, side, dict(fit=fit, val=va, val_us=val_us, test=te))
    if history_len:
        hist, gap, n_prev = address_history(df, history_len)
        is_first = (gap < 0).astype(np.float32)
        lg = np.log1p(np.clip(gap, 0, None)).astype(np.float32)
        lg = (lg - lg[fit].mean()) / lg[fit].std()
        prep.steps = np.concatenate([num, is_first[:, None], lg[:, None]], axis=1)
        prep.hist, prep.gap, prep.n_prev = hist, gap, n_prev
    return prep
