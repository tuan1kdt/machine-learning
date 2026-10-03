import pandas as pd
import streamlit as st

from common import card, summary


def Home() -> None:
    st.title("🛡️ Phát hiện địa chỉ Bitcoin ransomware")
    st.caption("BitcoinHeist Ransomware Address Dataset · UCI Machine Learning Repository · Akcora et al., 2019")

    s = summary().set_index("model")
    best = s.drop(index="Ensemble", errors="ignore")["macro_f1"].idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng dữ liệu", "2.916.697")
    c2.metric("Số lớp", "7", help="white + 5 họ ransomware lớn nhất + otherRansom")
    c3.metric("Mô hình tốt nhất", best)
    c4.metric(f"Macro F1 ({best})", f"{s.loc[best, 'macro_f1']:.3f}", help="Trên tập test 583.340 dòng")

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        card("🎯 Bài toán",
             "Cho các đặc trưng giao dịch của một địa chỉ Bitcoin trong một ngày, dự đoán địa chỉ đó là "
             "<b>bình thường</b> (<code>white</code>) hay thuộc <b>họ ransomware</b> nào — "
             "phân loại đa lớp, 7 lớp.")
        card("📦 Bộ dữ liệu",
             "Các giao dịch Bitcoin từ 2011 đến 2018, xây dựng đồ thị giao dịch theo ngày. Mỗi dòng là một "
             "(địa chỉ, ngày). Mất cân bằng nặng: <code>white</code> chiếm <b>98,6%</b>; 28 họ ransomware còn lại "
             "chỉ có 1,4%.")
        st.dataframe(pd.DataFrame([
            ("address", "chuỗi", "Địa chỉ Bitcoin (định danh — không dùng làm đặc trưng)"),
            ("year, day", "số nguyên", "Năm 2011–2018 và ngày trong năm 1–365"),
            ("length", "numerical", "Độ dài chuỗi giao dịch dài nhất dẫn tới địa chỉ"),
            ("weight", "numerical", "Lượng đầu vào được gộp về địa chỉ"),
            ("count", "numerical", "Số chuỗi giao dịch dẫn tới địa chỉ"),
            ("looped", "numerical", "Số giao dịch tách ra rồi gộp lại (dấu hiệu trộn tiền)"),
            ("neighbors", "numerical", "Số giao dịch có địa chỉ là đầu ra"),
            ("income", "numerical", "Số satoshi nhận được (1 BTC = 10⁸ satoshi)"),
            ("label", "categorical", "white hoặc tên họ ransomware (29 giá trị)"),
        ], columns=["Cột", "Kiểu", "Ý nghĩa"]), hide_index=True, width="stretch")

    with right:
        card("⚙️ Pipeline (Bài 1)",
             "<ol style='margin:0;padding-left:1.2rem'>"
             "<li>Gộp 29 nhãn → 7 lớp</li>"
             "<li>Tạo đặc trưng: <code>month</code>, <code>weekday</code>, độ tròn của <code>income</code>, "
             "các tỉ lệ income/count, income/neighbor, weight/count, looped/count</li>"
             "<li>Chia train / validation / test (stratified 64/16/20)</li>"
             "<li>Chuẩn hoá: <code>log1p</code> + <code>StandardScaler</code> (số), "
             "<code>OneHotEncoder</code> (year, month, weekday)</li>"
             "<li>Undersampling: giữ 100k <code>white</code> khi train</li>"
             "<li>Hiệu chỉnh trọng số lớp trên validation: dự đoán = argmax(p·w)</li>"
             "<li>6 mô hình + Ensemble, so sánh accuracy / precision / recall / F1 và thời gian</li></ol>")
        card("🧭 Hướng dẫn sử dụng demo",
             "<b>Khám phá dữ liệu</b> — phân phối nhãn, đặc trưng, tương quan.<br>"
             "<b>So sánh mô hình</b> — kết quả Bài 1 trên tập test.<br>"
             "<b>Chọn đặc trưng (Bài 2)</b> — tương quan với <code>log_income</code>, MAE của Linear Regression "
             "theo tập đặc trưng; tự chọn đặc trưng và train lại trực tiếp.<br>"
             "<b>Bài 2 — Phân lớp tuyến tính</b> — cùng bài toán 7 lớp, Linear Regression làm bộ phân lớp "
             "(one-hot + argmax); tương quan point-biserial, MAE / accuracy / macro F1 theo tập đặc trưng.<br>"
             "<b>Dự đoán</b> — nhập thông tin 1 địa chỉ hoặc tải lên file CSV để phân loại; "
             "có thể lấy ngẫu nhiên dòng thật từ tập test.")
        st.markdown(" ".join(f'<span class="pill">{t}</span>' for t in
                             ["Python 3.14", "Streamlit", "scikit-learn", "Plotly", "XGBoost", "LightGBM"]),
                    unsafe_allow_html=True)
