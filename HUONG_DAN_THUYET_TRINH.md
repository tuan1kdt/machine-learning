# Hướng dẫn thuyết trình giữa kỳ — BitcoinHeist

Tài liệu học nhanh cho người mới bắt đầu: đọc theo thứ tự. Mọi con số ở đây lấy từ `bai1_executed.ipynb`,
`bai2_executed.ipynb` và thư mục `outputs/`.

---

## 0. Làm ngay (10 phút)

1. **Điền họ tên + MSSV** ở 2 chỗ:
   - slide bìa của deck (đang để `[Họ tên thành viên 1]`, `MSSV: [__________]`);
   - `proj1/Team.py`, biến `MEMBERS` (dòng 7–11).
2. **Mở sẵn demo** (chạy từ thư mục dự án), rồi bấm qua cả 6 trang một lượt để app nạp sẵn mô hình:

   ```bash
   .venv/bin/streamlit run proj1/app.py
   ```

3. **Tải deck về máy** (PPTX hoặc PDF) làm bản dự phòng khi mất mạng.

## 1. Kế hoạch 2 tiếng

| Thời gian | Việc | Mục |
|---|---|---|
| 0:00–0:10 | Làm các việc ở mục 0 | 0 |
| 0:10–0:30 | Học 10 khái niệm Machine Learning cơ bản | 2 |
| 0:30–0:40 | Học thuộc "câu chuyện 1 phút" | 3 |
| 0:40–1:10 | Đọc từng slide và ghi chú người nói, nói to theo | 4, 5 |
| 1:10–1:25 | Tập demo theo kịch bản | 6 |
| 1:25–1:45 | Tự trả lời to các câu hỏi thường gặp | 7 |
| 1:45–2:00 | Tổng duyệt 1 lần có bấm giờ, làm checklist nộp bài | 9 |

**Chỉ còn 1 tiếng?** Làm mục 0 và mục 3. Đọc ghi chú người nói của slide 8, 9, 10, 14, 18, 19. Tập demo, rồi đọc
8 câu hỏi đầu ở mục 7.

---

## 2. Machine Learning trong 20 phút: 10 khái niệm phải hiểu

**1. Machine Learning là gì?** Thay vì tự viết luật "nếu… thì…", ta đưa cho máy **nhiều ví dụ đã có đáp án**.
Thuật toán tự tìm ra quy luật, rồi dùng quy luật đó để đoán cho ví dụ mới. Trong dự án này, mỗi ví dụ là
**một địa chỉ Bitcoin trong một ngày**. Đáp án là địa chỉ đó **bình thường (white)** hay **thuộc họ ransomware nào**.

**2. Đặc trưng (feature) và nhãn (label).**
- Đặc trưng là thông tin đầu vào: `length`, `weight`, `count`, `looped`, `neighbors`, `income`, `year`…
- Nhãn là thứ cần đoán: cột `label`.

**3. Phân loại và hồi quy.**
- *Phân loại (classification)* là đoán một **nhóm**. Đây là Bài 1: white, Locky, Cerber…
- *Hồi quy (regression)* là đoán một **con số**. Đây là Bài 2: số tiền `income`.

**4. Train / validation / test, giống việc ôn thi.**
- *Train* là sách bài tập: mô hình học trên đây.
- *Validation* là đề thi thử: dùng để chọn cấu hình (siêu tham số, trọng số lớp).
- *Test* là đề thi thật: chỉ dùng **một lần** để chấm điểm cuối.

Dự án chia 64% / 16% / 20%, tập test có 583.340 dòng. Chia *stratified* nghĩa là mỗi phần giữ đúng tỉ lệ các lớp.

**5. Overfitting và rò rỉ dữ liệu.**
- *Overfitting* là "học vẹt": làm bài tập đúng hết nhưng gặp đề mới thì sai. Cách phát hiện là chấm trên tập
  test chưa từng thấy.
- *Rò rỉ dữ liệu (data leakage)* là lỡ cho mô hình "xem trước đề thi", nên điểm cao giả. Dự án tránh bằng 2 cách:
  - bộ chuẩn hoá chỉ học (fit) trên train, đặt chung trong `Pipeline`;
  - kiểm tra lại bằng cách chia theo địa chỉ, để mỗi địa chỉ chỉ nằm ở train **hoặc** test.

**6. Chuẩn hoá dữ liệu.**
- `log1p(x) = log(1 + x)` nén các giá trị cực lớn. Ví dụ `income` trải từ 0,3 đến gần 500.000 BTC, sau log chỉ
  còn từ 17 đến 31. Cộng 1 để x = 0 vẫn tính được.
- `StandardScaler` tính z = (x − trung bình) / độ lệch chuẩn. Sau bước này mọi cột có trung bình 0 và độ lệch
  chuẩn 1, tức cùng một "thước đo". Rất quan trọng với KNN (dựa trên khoảng cách) và Logistic Regression.
- *One-hot* biến `year = 2016` thành 8 cột 0/1. Nhờ vậy mô hình học riêng từng năm, không coi năm là con số tăng đều.

**7. Mất cân bằng dữ liệu.** 98,6% số dòng là white. Một mô hình "lười" luôn đoán white vẫn đúng 98,6%, nhưng không
bắt được ransomware nào. Vì vậy **accuracy đánh lừa**.

**8. Precision, recall, F1, ví dụ với lớp Locky.** Giả sử có 100 địa chỉ Locky thật. Mô hình hô "Locky" 120 lần,
trong đó 80 lần đúng:
- **Precision** = 80 / 120 = 67%: *khi mô hình báo Locky, bao nhiêu % là đúng?* Thấp nghĩa là báo động nhầm nhiều.
- **Recall** = 80 / 100 = 80%: *trong các Locky thật, bắt được bao nhiêu %?* Thấp nghĩa là bỏ sót nhiều.
- **F1** = 2 × P × R / (P + R) = 0,73. F1 chỉ cao khi **cả hai** cùng cao.

Ví dụ này rất gần kết quả thật của XGBoost với Locky: P 0,683, R 0,789, F1 0,732.

**9. Weighted F1 và macro F1.**
- *Weighted F1* là trung bình F1 các lớp **theo số mẫu**. White chiếm 98,6% nên weighted F1 ≈ F1 của white,
  luôn khoảng 0,98.
- *Macro F1* là trung bình cộng F1 của 7 lớp, **mỗi lớp quan trọng như nhau**. Nó phản ánh khả năng bắt
  ransomware, và là tiêu chí chính của nhóm.

**10. MAE và hệ số tương quan Pearson (cho Bài 2).**
- *MAE* là trung bình |giá trị thật − dự đoán|, càng nhỏ càng tốt. Ví dụ: thật [10, 20, 30], đoán [12, 18, 33],
  sai số là 2, 2, 3, nên MAE = 7/3 ≈ 2,33.
- *Pearson r* (từ −1 đến 1) đo mức độ hai đại lượng cùng tăng/giảm theo **đường thẳng**:
  - gần 1: cùng chiều mạnh;
  - gần −1: ngược chiều mạnh;
  - gần 0: không có quan hệ tuyến tính.

  Tương quan **không** phải là nhân quả.

---

## 3. Câu chuyện 1 phút (học thuộc)

Dùng để mở đầu, hoặc khi thầy/cô hỏi "tóm tắt lại xem nhóm đã làm gì".

> Nhóm em dùng bộ dữ liệu BitcoinHeist trên UCI, gần 2,92 triệu dòng; mỗi dòng là một địa chỉ Bitcoin trong một
> ngày. Mục tiêu là phát hiện địa chỉ thuộc ransomware. Khó khăn lớn nhất là 98,6% dữ liệu là địa chỉ bình thường.
>
> **Bài 1:** nhóm gộp 29 nhãn thành 7 lớp, chuẩn hoá bằng log và StandardScaler. Nhóm xử lý mất cân bằng bằng
> undersampling kết hợp hiệu chỉnh trọng số lớp, rồi so sánh 6 mô hình. Accuracy bị mất cân bằng đánh lừa, nên
> nhóm chọn mô hình theo macro F1. XGBoost tốt nhất: macro F1 0,575, weighted F1 0,985, dự đoán 583 nghìn dòng
> trong 0,75 giây.
>
> **Bài 2:** nhóm dự đoán số tiền nhận được bằng Linear Regression và chọn đặc trưng bằng tương quan Pearson. Chỉ 6
> trên 12 đặc trưng đã đạt 98% mức cải thiện MAE, và train nhanh gấp đôi.
>
> **Bài 3:** ứng dụng Streamlit để xem dữ liệu, so sánh mô hình, thử chọn đặc trưng và dự đoán trực tiếp.

---

## 4. Hiểu từng bài (để trả lời được câu hỏi)

### Bài 1: Phân loại (5 điểm) · `bai1.ipynb`

1. **Đọc & khám phá (mục 1).** 2.916.697 dòng, 10 cột, không thiếu giá trị. Có 3 phát hiện:
   - mất cân bằng nặng;
   - `year` rất quan trọng: từ 2012 đến 2017, năm nào cũng đúng 365.000 dòng white (1.000/ngày), còn tỉ lệ
     ransomware dao động từ 0,001% đến 4,1%;
   - các cột số lệch mạnh.
2. **Gộp nhãn (mục 2).** 29 nhãn thành 7 lớp: white, 5 họ lớn nhất (paduaCryptoWall, montrealCryptoLocker,
   princetonCerber, princetonLocky, montrealCryptXXX) và otherRansom. Lý do: nhiều họ chỉ có 1–100 mẫu, không
   đủ để học.
3. **Đặc trưng mới (mục 3).**
   - `month`, `weekday`, lấy từ `day`;
   - `income_tz`: số chữ số 0 ở cuối `income`, vì tiền chuộc hay là số tròn như đúng 1 BTC;
   - 4 tỉ lệ: `income_per_count`, `income_per_neighbor`, `weight_per_count`, `looped_ratio`.
4. **Chia dữ liệu (mục 4).** Train 64%, validation 16%, test 20%, chia stratified.
5. **Chuẩn hoá (mục 5).** Cột số dùng log1p rồi StandardScaler. `year`, `month`, `weekday` dùng OneHotEncoder.
   Tất cả nằm trong một `Pipeline` và chỉ fit trên train.
6. **Mất cân bằng (mục 6).**
   - (A) undersampling: giữ 100.000 white và toàn bộ ransomware;
   - (B) hiệu chỉnh trọng số: chọn lớp theo argmax(p · w), với w tìm trên validation.

   Macro F1 trên validation: không xử lý 0,464; chỉ A 0,456; chỉ B 0,481; **A + B 0,544**.
7. **Mô hình (mục 7).** Logistic Regression, KNN (k = 15), Random Forest (300 cây), HistGradientBoosting, LightGBM,
   XGBoost (GPU), cộng một Ensemble. Siêu tham số chọn bằng random search 34 cấu hình (`experiments.py`).
8. **Kết quả (mục 8–9).** Bảng ở slide 10–12. XGBoost là mô hình đơn tốt nhất.
9. **Kiểm chứng (mục 10).**
   - Bài toán nhị phân "có phải ransomware không": precision 47%, recall 55%.
   - Bỏ `year`: macro F1 còn 0,444.
   - Chia theo địa chỉ: macro F1 vẫn 0,565.
10. **Lưu mô hình (mục 11).** Các file `.joblib` được demo dùng lại.

### Bài 2: Feature Selection (3 điểm) · `bai2.ipynb`

- **Bài toán.** Dự đoán `log(1 + income)` từ 12 đặc trưng bằng Linear Regression, đánh giá bằng MAE.
  - Không dùng nhãn của Bài 1, vì 7 lớp không có thứ tự nên không hồi quy được.
  - Bỏ các đặc trưng tạo từ `income` để tránh rò rỉ đáp án.
- **(a) Tương quan (mục 3).**
  - Tính Pearson r giữa từng đặc trưng và mục tiêu trên tập train, rồi xếp hạng theo |r|. Mạnh nhất là year
    (−0,285) và neighbors (+0,198).
  - Ma trận tương quan giữa các đặc trưng tìm ra cặp dư thừa: day – month, r = 0,996.
  - Có so sánh thêm với Spearman, và đối chiếu bằng `r_regression`, `SelectKBest(f_regression)` của scikit-learn.
- **(b) Thử các tập đặc trưng (mục 4).** Thử Top-k theo |r|, ngưỡng |r| ≥ t, loại dư thừa, và tập đối chứng.
  - **Top-6 đạt 98% mức cải thiện với 50% số đặc trưng.**
  - Tập đối chứng (6 đặc trưng |r| thấp nhất) chỉ cải thiện 2,5%, chứng tỏ tương quan chọn đúng.
- **Vì sao k = 6?** Đó là k nhỏ nhất đạt ≥ 95% mức cải thiện; Top-5 mới được 93,5%.
- **% cải thiện** = (MAE mốc − MAE tập đặc trưng) / (MAE mốc − MAE 12 đặc trưng) × 100. "Mốc" là luôn đoán giá
  trị trung bình, MAE 1,4641.
- **MAE 1,33 nghĩa là gì?** Mục tiêu là log, nên dự đoán thường lệch khoảng e^1,33 ≈ 3,8 lần so với số tiền thật.
  Linear Regression chỉ giải thích ~15% biến động (R² ≈ 0,15); hãy chủ động nêu đây là hạn chế.

### Bài 3: Demo (2 điểm) · `proj1/`

- Dùng Python 3.14, Streamlit, scikit-learn, Plotly, XGBoost, LightGBM.
- Có 6 trang:

  | Trang | Nội dung |
  |---|---|
  | Giới thiệu | Tổng quan bài toán |
  | Khám phá dữ liệu | Tab Nhãn / Đặc trưng / Tương quan |
  | So sánh mô hình | Bảng độ đo và độ đo từng lớp |
  | Chọn đặc trưng (Bài 2) | Tab Tương quan / Kết quả MAE / Thử nghiệm trực tiếp |
  | Dự đoán | Tab Một địa chỉ / Hàng loạt (CSV) |
  | Nhóm thực hiện | Thành viên và nguồn dữ liệu |

- Mô hình nạp từ `proj1/models/`, không phải train lại. Random Forest không có trong demo vì file nặng 128 MB.
- Tab "Thử nghiệm trực tiếp" train lại Linear Regression trên mẫu 100.000 dòng train / 25.000 dòng test. Vì vậy MAE
  ở đó lệch nhẹ so với notebook (dùng toàn bộ dữ liệu); nếu bị hỏi thì giải thích như vậy.

---

## 5. Kịch bản theo slide (~15 phút + 2–3 phút demo)

Ghi chú người nói đầy đủ có trong từng slide của deck. Bảng dưới chỉ là **câu chốt**: nếu quên hết, hãy nói ít
nhất câu này.

| # | Slide | Thời gian | Câu chốt |
|---|---|---|---|
| 1 | Bìa | 20 s | Phát hiện địa chỉ Bitcoin ransomware trên bộ BitcoinHeist (UCI). |
| 2 | Nội dung | 15 s | Đi đúng 3 yêu cầu của đề: phân loại, chọn đặc trưng, demo. |
| 3 | Bộ dữ liệu | 40 s | 2,92 triệu dòng, 29 nhãn, 98,6% white: mất cân bằng nặng. |
| 4 | Các trường | 45 s | 6 đặc trưng đồ thị + thời gian + income; có cả numerical lẫn categorical. |
| 5 | Khám phá | 45 s | 3 phát hiện: mất cân bằng, year quan trọng, cột số lệch nên dùng log1p. |
| 6 | Quy trình | 50 s | Chuẩn hoá nằm trong Pipeline, chỉ fit trên train, nên không rò rỉ. |
| 7 | 6 mô hình | 60 s | 4 nhóm thuật toán: tuyến tính, khoảng cách, bagging, boosting. |
| 8 | Độ đo | 50 s | Luôn đoán white vẫn được accuracy 98,6%, nên chọn theo macro F1. |
| 9 | Mất cân bằng | 60 s | A và B phải đi cùng nhau: 0,464 lên 0,544. |
| 10 | Kết quả | 60 s | XGBoost: macro F1 0,575, weighted F1 0,985, test 0,75 s. |
| 11 | Từng lớp | 45 s | Tốt: CryptXXX 0,86, Locky 0,73. Khó: otherRansom 0,11. |
| 12 | Thời gian | 45 s | KNN train 0 s nhưng test 14 s; XGBoost cân bằng nhất. |
| 13 | Kiểm chứng | 45 s | Chia theo địa chỉ vẫn 0,565: mô hình không học thuộc địa chỉ. |
| 14 | Bài 2: bài toán | 60 s | Linear Regression + MAE cần mục tiêu liên tục, nên dự đoán log(income). |
| 15 | Xếp hạng r | 40 s | year −0,285, neighbors +0,198; các đặc trưng còn lại rất yếu. |
| 16 | Minh hoạ | 30 s | Tương quan mạnh nhìn thấy được trên biểu đồ; weekday thì phẳng. |
| 17 | Dư thừa | 45 s | day – month r = 0,996: bỏ day, MAE không đổi. |
| 18 | Top-k | 60 s | Top-6 đạt 98% mức cải thiện với 50% đặc trưng, train nhanh gấp 2. |
| 19 | Tổng hợp | 60 s | Đối chứng ≈ mốc, nên tương quan chọn đúng; hạn chế: R² 0,15. |
| 20 | Demo | 2–3 phút | Chuyển sang app, làm theo kịch bản ở mục 6. |
| 21 | Kết luận | 40 s | Kết quả chính, hạn chế, hướng phát triển. |
| 22 | Hỏi đáp | — | Cảm ơn và mời câu hỏi. |

- **Bị giới hạn 10 phút?** Mỗi slide 4, 12, 13, 16 chỉ nói 1 câu, và demo trong 2 phút.
- **Nhóm 3 người?** Theo phân công trong `Team.py`:
  - Thành viên 1 (Bài 1): slide 6–13;
  - Thành viên 2 (Bài 2): slide 14–19;
  - Thành viên 3 (Bài 3): slide 1–5 và 20–22.

---

## 6. Kịch bản demo (2–3 phút)

1. **Giới thiệu**: 10 giây, chỉ nói "đây là ứng dụng của nhóm, gồm 6 trang".
2. **Khám phá dữ liệu**: mở tab **Nhãn** (mất cân bằng), rồi tab **Tương quan**.
3. **So sánh mô hình**: chỉ vào bảng độ đo. Ở ô **Mô hình** chọn XGBoost, rồi đổi giữa
   **f1-score / precision / recall** để xem từng lớp.
4. **Chọn đặc trưng (Bài 2)**:
   - tab **Kết quả MAE**: chỉ vào đường cong Top-k;
   - tab **Thử nghiệm trực tiếp**: kéo thanh **k** từ 2 lên 6, bật **Loại đặc trưng dư thừa**, MAE cập nhật ngay.
5. **Dự đoán**:
   - tab **Một địa chỉ**: chọn lớp `princetonLocky`, bấm **🎲 Lấy mẫu ngẫu nhiên**, rồi **🔍 Dự đoán**, xem xác
     suất từng lớp;
   - tab **Hàng loạt (CSV)**: chọn **Mẫu từ tập test**, xem bảng độ đo và ma trận nhầm lẫn.
   - **Lưu ý, dễ bị hỏi:** mẫu này mặc định có **30% ransomware** (tập test thật chỉ có 1,4%) để các lớp hiếm hiện rõ.
     Vì vậy accuracy chỉ khoảng **0,86**, khác 0,985 trên slide. Nếu bị hỏi, kéo thanh **Tỉ lệ ransomware trong mẫu**
     về 0,05: accuracy tăng lên khoảng **0,97**. Ý chính: cùng một mô hình, độ đo phụ thuộc vào tỉ lệ các lớp trong dữ liệu.

Nếu mô hình đoán sai ở bước 5, đừng hoảng. Hãy nói: *"macro F1 0,575 nghĩa là mô hình vẫn còn nhầm, nhất là ở các
lớp hiếm; đây là hạn chế nhóm đã nêu."* Thầy cô đánh giá cao sự trung thực.

---

## 7. Câu hỏi thường gặp & trả lời mẫu

### Độ đo

1. **Tại sao không dùng accuracy để chọn mô hình?**
   Vì dữ liệu mất cân bằng. Mô hình luôn đoán white có accuracy 0,986, cao hơn mọi mô hình, nhưng không phát hiện
   được ransomware nào (macro F1 chỉ 0,142). Nhóm vẫn báo cáo accuracy theo yêu cầu đề, nhưng chọn mô hình theo
   macro F1.
2. **Precision và recall, cái nào quan trọng hơn?**
   Tuỳ mục tiêu. Hệ thống cảnh báo ransomware thường ưu tiên recall để không bỏ sót, nhưng precision thấp lại gây
   nhiều báo động nhầm. F1 cân bằng cả hai. Nếu cần, có thể chỉnh trọng số w để nghiêng về recall.
3. **Macro F1 khác weighted F1 thế nào?**
   Weighted F1 lấy trung bình theo số mẫu nên bị white chi phối (≈ 0,985 với mọi mô hình tốt). Macro F1 coi 7 lớp
   như nhau, nên phân biệt rõ mô hình tốt và kém.
4. **Weighted F1 0,985 cao vậy, sao còn nói mô hình chưa tốt?**
   Vì nó gần như chỉ đo lớp white. Riêng câu hỏi "có phải ransomware không", precision là 47% và recall 55%.

### Dữ liệu & tiền xử lý

5. **Tại sao phải chuẩn hoá? Mô hình cây có cần không?**
   Mô hình cây không cần, vì cây chỉ so sánh với ngưỡng. Logistic Regression và KNN thì cần: với KNN, cột có giá trị
   lớn sẽ lấn át khoảng cách. Nhóm dùng chung một pipeline cho mọi mô hình để so sánh công bằng.
6. **Tại sao dùng log1p?**
   Các cột số lệch rất mạnh (income từ 0,3 đến gần 500.000 BTC). Log nén giá trị cực lớn để chúng không lấn át,
   giúp mô hình tuyến tính và KNN học tốt hơn.
7. **Tại sao gộp 29 nhãn thành 7?**
   Nhiều họ chỉ có 1–100 mẫu, không thể học hay đánh giá có ý nghĩa. Giữ 5 họ lớn nhất, gộp phần còn lại thành
   otherRansom.
8. **Sao không chỉnh tham số trực tiếp trên tập test?**
   Như vậy là "xem đề trước": điểm sẽ ảo, không phản ánh dữ liệu mới. Mọi lựa chọn đều làm trên validation; test
   chỉ dùng một lần.
9. **Dùng `year` có phải gian lận không?**
   Không, vì năm giao dịch là thông tin có thật lúc dự đoán. Nhưng nó phản ánh cách thu thập dữ liệu: white lấy cố
   định 1.000/ngày, còn ransomware tập trung vào 2013–2016. Nhóm đã kiểm tra: bỏ year thì macro F1 còn 0,444. Khi áp
   dụng cho năm mới, mô hình có thể kém hơn; đây là hạn chế.

### Mất cân bằng

10. **Undersampling có làm mất thông tin không?**
    Có bỏ bớt white, nhưng 100 nghìn mẫu white vẫn đủ đa dạng. Thực nghiệm cho thấy: khi kết hợp hiệu chỉnh trọng
    số, 100k white cho kết quả tốt nhất (0,544), hơn cả giữ toàn bộ 1,84 triệu white (0,481).
11. **Sao không dùng SMOTE / oversampling?**
    Nhóm chưa thử. Ransomware có hơn 41 nghìn mẫu thật nên đủ để học. SMOTE tạo mẫu giả bằng nội suy, có thể sinh ra
    giá trị không thực tế với các cột dạng đếm như `count`, `looped`, và làm tập train lớn hơn nên chậm hơn. Đây là
    một hướng mở rộng.
12. **Hiệu chỉnh trọng số có phải là "nhìn trộm" test không?**
    Không. Trọng số w được tìm trên validation; mô hình không đổi, chỉ đổi cách ra quyết định. Test chỉ dùng để
    chấm điểm.

### Mô hình

13. **Random Forest khác Boosting thế nào?**
    - Random Forest: nhiều cây **độc lập**, mỗi cây học trên một mẫu ngẫu nhiên, rồi bỏ phiếu. Cách này giảm dao
      động (variance).
    - Boosting: các cây **nối tiếp**, cây sau học phần lỗi của các cây trước. Cách này giảm sai lệch (bias) nên
      thường chính xác hơn, và cần learning rate nhỏ, regularization, early stopping để chống overfitting.
14. **Vì sao Logistic Regression kém?**
    Nó chỉ vẽ được ranh giới tuyến tính, trong khi ranh giới giữa các họ ransomware phi tuyến và chồng lấn nhau. Cây
    quyết định chia dữ liệu thành nhiều vùng nhỏ nên bắt được quan hệ phi tuyến.
15. **Vì sao KNN train nhanh nhất nhưng test chậm nhất?**
    KNN không học gì khi train, chỉ lưu dữ liệu. Khi dự đoán, nó phải tính khoảng cách từ mỗi điểm trong 583 nghìn
    điểm test tới 126 nghìn điểm train.
16. **Có bị overfitting không?**
    Kết quả báo cáo trên 583 nghìn dòng test chưa từng thấy. Boosting có early stopping và regularization. Khi chia
    theo địa chỉ, macro F1 vẫn 0,565.
17. **Ensemble cao hơn, sao lại chọn XGBoost?**
    Ensemble chỉ hơn 0,002 macro F1 (0,576 so với 0,575), nhưng tốn gấp khoảng 4 lần thời gian train và 22 lần thời
    gian test (16,8 s so với 0,75 s).

### Bài 2

18. **Tại sao Bài 2 không dùng nhãn của Bài 1?**
    Đề yêu cầu Linear Regression + MAE, tức bài toán hồi quy, cần mục tiêu là số liên tục. 7 lớp không có thứ tự, nên
    đánh số 0–6 rồi hồi quy là vô nghĩa. Nhóm chọn `income`, một biến số có ý nghĩa thật, và lấy log vì nó lệch mạnh.
19. **Tương quan cao nhất chỉ 0,285, phương pháp có ý nghĩa không?**
    Có, vì thứ hạng vẫn đúng. Top-6 đạt 98% mức cải thiện, còn 6 đặc trưng có |r| thấp nhất chỉ 2,5%. Mục tiêu của
    Bài 2 là so sánh các tập đặc trưng, không phải đạt MAE thấp nhất.
20. **Vì sao dùng Pearson mà không phải Spearman?**
    Linear Regression là mô hình tuyến tính, còn Pearson đo đúng quan hệ tuyến tính. Thực nghiệm xác nhận: Top-6 theo
    Spearman cho MAE 1,3476 (86,8%), kém hơn Top-6 theo Pearson (1,3325).
21. **Dùng p-value để chọn đặc trưng được không?**
    Không hữu ích ở đây. Với 2,3 triệu dòng train, mọi tương quan dù rất nhỏ (looped r = 0,002) đều có p < 0,05. Phải
    dựa vào độ lớn |r| và kiểm chứng bằng MAE.
22. **Nhược điểm của chọn đặc trưng bằng tương quan?**
    - Chỉ đo quan hệ tuyến tính giữa từng cặp, nên không thấy tương tác: hai đặc trưng yếu riêng lẻ có thể mạnh khi
      kết hợp.
    - Nhạy với ngoại lai.
    - Không nói lên nhân quả.

    Có thể mở rộng bằng Mutual Information, RFE, Lasso (L1) hoặc độ quan trọng đặc trưng của mô hình cây.
23. **R² = 0,15 thấp có sao không?**
    Đó là giới hạn của mô hình tuyến tính với dữ liệu này: quan hệ với income yếu và phi tuyến. Nhóm nêu rõ đây là
    hạn chế. Kết luận về chọn đặc trưng vẫn đúng, vì mọi tập đặc trưng được so sánh trên cùng điều kiện.

### Demo & câu hỏi mở

24. **Ứng dụng Streamlit hoạt động thế nào?**
    Streamlit chạy script Python và tự sinh giao diện web. Mỗi khi người dùng đổi một nút hay thanh kéo, script chạy
    lại. Dữ liệu và mô hình được cache để không phải nạp lại. Mô hình là file `.joblib` chứa pipeline (tiền xử lý +
    mô hình) và trọng số lớp, lưu từ Bài 1.
25. **Có thêm thời gian, nhóm sẽ làm gì?**
    - Bổ sung giá BTC theo ngày để quy income ra USD.
    - Tổng hợp hành vi nhiều ngày của cùng một địa chỉ.
    - Dùng cấu trúc đồ thị giao dịch đầy đủ.
    - Ở Bài 2: thử Mutual Information, RFE và mô hình phi tuyến.
26. **Gặp câu không biết trả lời?**
    Nói thật: *"Nhóm em chưa thử điều đó; đây là một hướng hay để mở rộng."* Không bịa con số.

---

## 8. Bảng số liệu cần nhớ

| Nội dung | Số liệu |
|---|---|
| Dữ liệu | 2.916.697 dòng · 10 cột · 29 nhãn, gộp còn 7 lớp · 98,6% white |
| Chia dữ liệu | 64 / 16 / 20% · test 583.340 dòng |
| Mất cân bằng (macro F1, validation) | không xử lý 0,464 · chỉ A 0,456 · chỉ B 0,481 · A + B 0,544 |
| XGBoost (test) | accuracy 0,985 · weighted F1 0,985 · macro F1 0,575 · train 12,8 s · test 0,75 s |
| Mốc "luôn đoán white" | accuracy 0,986 · weighted F1 0,979 · macro F1 0,142 |
| Ransomware vs white (XGBoost) | precision 47% · recall 55% |
| Kiểm chứng | bỏ year: 0,444 · chia theo địa chỉ: 0,565 |
| Bài 2, MAE | mốc 1,4641 · 12 đặc trưng 1,3300 · Top-6 1,3325 |
| Bài 2, % cải thiện | Top-1 55% · Top-2 85% · Top-6 98% · đối chứng 2,5% · Spearman Top-6 86,8% |
| Tương quan | year −0,285 · neighbors +0,198 · day–month 0,996 |

---

## 9. Checklist nộp bài

- [ ] Họ tên + MSSV trên slide bìa và trong `proj1/Team.py`.
- [ ] Chạy lại app một lần sau khi sửa `Team.py`.
- [ ] Tải deck (PPTX hoặc PDF), tạo thư mục `slides/` trong thư mục dự án và đặt file vào đó.
- [ ] Tạo file zip tên `<MSSV1>_<MSSV2>_<MSSV3>.zip`, gồm slide, mã nguồn và dữ liệu. Thay `MSSV1_MSSV2_MSSV3`
      bằng MSSV thật, rồi chạy từ thư mục dự án:

  ```bash
  zip -r MSSV1_MSSV2_MSSV3.zip slides bai1.ipynb bai1_executed.ipynb bai2.ipynb bai2_executed.ipynb experiments.py experiments_group.py exp_out requirements.txt BitcoinHeistData.csv outputs proj1 -x "*/__pycache__/*" "outputs/models/*" "outputs/best_model.joblib" "*.DS_Store"
  ```

- [ ] File zip khoảng 170 MB: CSV nén còn ~116 MB, mô hình demo ~49 MB. Nếu hệ thống nộp bài giới hạn dung lượng,
      hỏi thầy/cô có được thay CSV bằng link UCI (https://archive.ics.uci.edu/dataset/526) hoặc link Google Drive
      không.
- [ ] Một sinh viên đại diện nộp.
