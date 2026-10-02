"""Chuẩn bị dữ liệu & mô hình cho demo (chạy 1 lần):  python prepare_data.py [--with-rf]

- Tái tạo đúng phép chia train/test của Bài 1 (stratified, random_state=42) → lấy mẫu từ TẬP TEST
  (dữ liệu mô hình chưa thấy) để demo: toàn bộ dòng ransomware + 20.000 dòng white.
- Thống kê nhãn trên toàn bộ 2,9 triệu dòng (cho trang Khám phá dữ liệu).
- Sao chép kết quả Bài 1 (outputs/*.csv) và các mô hình (outputs/models) vào proj1.
  RandomForest (~128 MB) chỉ được sao chép khi có cờ --with-rf.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
SRC = ROOT.parent
DATA = ROOT / "data"
RANDOM_STATE = 42
N_WHITE_DEMO = 20_000

df = pd.read_csv(SRC / "BitcoinHeistData.csv")
label_counts = df["label"].value_counts()
top = label_counts.drop("white").head(5).index
df["target"] = np.where(df["label"] == "white", "white", np.where(df["label"].isin(top), df["label"], "otherRansom"))

# Cùng phép chia như bai1.ipynb mục 4 (chỉ cần nhãn để stratify → chỉ số test giống hệt)
_, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE)
test = df.loc[test_idx]
demo = pd.concat([test[test["target"] != "white"],
                  test[test["target"] == "white"].sample(N_WHITE_DEMO, random_state=RANDOM_STATE)])
demo = demo.sample(frac=1, random_state=RANDOM_STATE)

DATA.mkdir(exist_ok=True)
(DATA / "results").mkdir(exist_ok=True)
demo.to_csv(DATA / "test_sample.csv.gz", index=False)
label_counts.rename_axis("label").reset_index(name="count").to_csv(DATA / "label_counts.csv", index=False)
df.groupby(["year", "target"]).size().rename("count").reset_index().to_csv(DATA / "year_target_counts.csv", index=False)
for f in ["summary.csv", "per_class.csv"]:
    shutil.copy(SRC / "outputs" / f, DATA / "results" / f)

(ROOT / "models").mkdir(exist_ok=True)
for p in (SRC / "outputs" / "models").glob("*.joblib"):
    if p.stem != "RandomForest" or "--with-rf" in sys.argv:
        shutil.copy(p, ROOT / "models" / p.name)

print(f"test_sample: {len(demo):,} dòng ({(demo['target'] != 'white').sum():,} ransomware)")
print("models:", sorted(p.name for p in (ROOT / "models").glob("*.joblib")))
