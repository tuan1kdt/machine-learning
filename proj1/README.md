# Bài 3 — Demo Streamlit: phát hiện ransomware (BitcoinHeist)

Demo cho mô hình phân loại 7 lớp (`white` + 5 họ ransomware + `otherRansom`) đã huấn luyện ở Bài 1 (`../bai1.ipynb`).
Trang **Chọn đặc trưng (Bài 2)** trình bày kết quả của `../bai2.ipynb`.

## Cài đặt & chạy (Python > 3.10)

```bash
pip install -r requirements.txt
brew install libomp            # chỉ macOS — XGBoost/LightGBM cần OpenMP
python prepare_data.py         # 1 lần: tạo data/ và sao chép mô hình vào models/ (thêm --with-rf để chép RandomForest ~128 MB)
streamlit run app.py
```

`prepare_data.py` cần `../BitcoinHeistData.csv` và `../outputs/` (kết quả của notebook Bài 1). Mô hình được tìm
trong `models/` trước, sau đó `../outputs/models/`.

## Các trang

| Trang | Nội dung |
|---|---|
| Giới thiệu | Bài toán, mô tả bộ dữ liệu, pipeline |
| Khám phá dữ liệu | Phân phối nhãn, theo năm; phân phối đặc trưng theo lớp; ma trận tương quan |
| So sánh mô hình | Accuracy / precision / recall / F1 (từng lớp, weighted, macro), thời gian train / test |
| Chọn đặc trưng (Bài 2) | Hồi quy `log_income` bằng Linear Regression: tương quan Pearson / Spearman với mục tiêu, ma trận tương quan (đặc trưng dư thừa); MAE theo Top-k và theo tập đặc trưng; thử nghiệm trực tiếp (Top-k / ngưỡng \|r\| / tự chọn, loại dư thừa) rồi train lại trên mẫu |
| Dự đoán | Nhập 1 địa chỉ (hoặc lấy ngẫu nhiên dòng thật từ tập test) · Dự đoán hàng loạt từ CSV + ma trận nhầm lẫn |
| Nhóm thực hiện | Thành viên — **sửa `MEMBERS` trong `Team.py`** |

## Dữ liệu demo (`data/`)

- `test_sample.csv.gz` — 28.283 dòng lấy từ **tập test** của Bài 1 (mô hình chưa thấy): toàn bộ 8.283 dòng
  ransomware + 20.000 dòng white.
- `label_counts.csv`, `year_target_counts.csv` — thống kê nhãn trên toàn bộ 2.916.697 dòng.
- `results/summary.csv`, `results/per_class.csv` — kết quả Bài 1 trên tập test.
- `bai2/` — kết quả Bài 2, **do notebook `../bai2.ipynb` tạo ra** (ô cuối mục 4, không phải `prepare_data.py`):
  `corr_target.csv`, `corr_matrix.csv` (tương quan trên tập train), `topk.csv`, `subsets.csv` (MAE trên tập test),
  `sample.csv.gz` (mẫu 100.000 dòng train + 25.000 dòng test cho tab *Thử nghiệm trực tiếp*).
