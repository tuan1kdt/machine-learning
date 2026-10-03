import pandas as pd
import streamlit as st

from common import card

# TODO: điền họ tên & MSSV của các thành viên trong nhóm
MEMBERS = [
    {"Họ tên": "Thành viên 1", "MSSV": "MSSV1", "Phụ trách": "Bài 1 — tiền xử lý & mô hình"},
    {"Họ tên": "Thành viên 2", "MSSV": "MSSV2", "Phụ trách": "Bài 2 — Feature Selection"},
    {"Họ tên": "Thành viên 3", "MSSV": "MSSV3", "Phụ trách": "Bài 3 — Demo Streamlit"},
]


def Team() -> None:
    st.header("👥 Nhóm thực hiện")
    st.dataframe(pd.DataFrame(MEMBERS), hide_index=True, width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        card("📚 Môn học", "Học máy (Machine Learning) — Dự án giữa kỳ 2026")
        card("🔗 Nguồn dữ liệu",
             '<a href="https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset" '
             'target="_blank">UCI — BitcoinHeist Ransomware Address Dataset</a><br>'
             "Akcora, C. G., Li, Y., Gel, Y. R., Kantarcioglu, M. (2019). <i>BitcoinHeist: Topological Data "
             "Analysis for Ransomware Detection on the Bitcoin Blockchain</i>.")
    with b:
        card("🗂️ Cấu trúc mã nguồn",
             "<code>app.py</code> — điều hướng<br>"
             "<code>features.py</code> — tạo đặc trưng & dự đoán (giống Bài 1)<br>"
             "<code>prepare_data.py</code> — chuẩn bị dữ liệu demo & sao chép mô hình<br>"
             "<code>Home / Statistics / Models / Predict / Team.py</code> — các trang<br>"
             "<code>FeatureSelection.py</code> — trang Bài 2: chọn đặc trưng dựa trên tương quan "
             "(dữ liệu <code>data/bai2/</code> do <code>../bai2.ipynb</code> tạo)<br>"
             "<code>LinearClassification.py</code> — trang Bài 2 bản phân lớp: Linear Regression làm bộ phân lớp "
             "tuyến tính (dữ liệu <code>data/bai2_cls/</code> do <code>../bai2_classification.ipynb</code> tạo)")
