# 🤖 Phát Hiện Tin Giả Tiếng Việt bằng PhoBERT

> Dự án môn Trí Tuệ Nhân Tạo – Nhóm Giday  
> GitHub: https://github.com/vamirio1710/Trituenhantaonhomgiday  
> ⏰ Deadline nộp bài: **21h30 ngày 09/06/2026**

---

## 📌 Dự án này làm gì?

Dự án này xây dựng một hệ thống **tự động kiểm tra độ tin cậy của một câu khẳng định (claim)** dựa trên bằng chứng (evidence) từ văn bản tiếng Việt.

Ví dụ đơn giản:
- **Claim (câu cần kiểm tra):** *"Việt Nam có 100 triệu dân"*
- **Evidence (bằng chứng):** *"Theo Tổng cục Thống kê, dân số Việt Nam năm 2023 đạt khoảng 99,4 triệu người"*
- **Kết quả:** ✅ SUPPORTED (bằng chứng ủng hộ câu trên)

Hệ thống có thể phân loại thành **3 nhãn**:
| Nhãn | Ý nghĩa |
|------|---------|
| `SUPPORTED` | Bằng chứng ủng hộ câu khẳng định |
| `REFUTED` | Bằng chứng bác bỏ câu khẳng định |
| `NEI` | Không đủ thông tin để kết luận |

---

## 🧠 Tại sao dùng PhoBERT?

**PhoBERT** là mô hình AI được huấn luyện đặc biệt trên tiếng Việt (giống như BERT nhưng "biết" tiếng Việt hơn). So với các phương pháp truyền thống:

| Phương pháp | Hiểu ngữ nghĩa? | Hiểu tiếng Việt? |
|------------|----------------|-----------------|
| TF-IDF + Logistic Regression | ❌ Không | ⚠️ Một phần |
| mBERT (đa ngôn ngữ) | ✅ Có | ⚠️ Trung bình |
| **PhoBERT (mô hình chính)** | ✅ Có | ✅ Tốt nhất |

---

## 📂 Cấu trúc thư mục

```
Trituenhantaonhomgiday/
│
├── Mô_tả.ipynb                  # Mô tả tổng quan bài toán và dataset
├── Tiền_xử_lý.ipynb             # Code xử lý dữ liệu thô → dữ liệu sạch
├── Baseline_TFIDF.ipynb         # Mô hình baseline TF-IDF + Logistic Regression
├── PhoBERT_FineTuning.ipynb     # Fine-tuning mô hình PhoBERT chính
├── Evaluation.ipynb             # Đánh giá, biểu đồ, ma trận nhầm lẫn
├── app.py                       # Giao diện web Streamlit
└── README.md                    # File này
```

---

## 🗃️ Dataset

Dự án sử dụng **VnFactcheck** (còn gọi là VFND) – dataset chuẩn cho bài toán kiểm tra tính xác thực của thông tin tiếng Việt.

- Nguồn: [ACL 2024 Paper](https://aclanthology.org/2024.findings-acl.551.pdf)
- Gồm các cặp (Claim, Evidence) được gán nhãn thủ công
- 3 nhãn: SUPPORTED / REFUTED / NEI

---

## ⚙️ Cài đặt môi trường

Dự án chạy trên **Google Colab** (miễn phí, có GPU).

```bash
# Cài các thư viện cần thiết
pip install transformers torch underthesea scikit-learn streamlit pandas matplotlib seaborn
```

Hoặc nếu chạy local:
```bash
pip install -r requirements.txt
```

---

## 🚀 Cách chạy

### 1. Tiền xử lý dữ liệu
Mở và chạy toàn bộ `Tiền_xử_lý.ipynb`  
→ Output: file dữ liệu sạch lưu lên Google Drive chung

### 2. Chạy baseline (nhanh, không cần GPU)
Mở `Baseline_TFIDF.ipynb` và chạy  
→ Kết quả Accuracy, F1-score của TF-IDF + Logistic Regression

### 3. Fine-tuning PhoBERT (cần GPU)
Mở `PhoBERT_FineTuning.ipynb` trên Colab  
→ Chọn Runtime → Change runtime type → **GPU**  
→ Chạy toàn bộ notebook

### 4. Xem kết quả đánh giá
Mở `Evaluation.ipynb`  
→ Xem biểu đồ, confusion matrix, bảng so sánh các mô hình

### 5. Chạy giao diện demo
```bash
streamlit run app.py
```
→ Mở trình duyệt tại `http://localhost:8501`

---

## 👥 Phân công thành viên

### 👤 Hiền (TV1)
**Viết báo cáo:**
- Chương 1: Giới thiệu dự án (bối cảnh, mục tiêu, phạm vi, phương pháp)
- Mục 3.2: Các mô hình dùng để so sánh
- Chương 5: Kết luận và khuyến nghị
- Format toàn bộ file Word

**Code:**
- Implement baseline **TF-IDF + Logistic Regression**

---

### 👤 Nguyên (TV2)
**Viết báo cáo:**
- Mục 2.2.1: Thu thập dữ liệu (nguồn gốc VnFactcheck, thống kê, ví dụ 3 nhãn)
- Mục 2.2.2: Tiền xử lý dữ liệu (pipeline từng bước, minh họa trước/sau)

**Code:**
- Toàn bộ pipeline tiền xử lý: lọc HTML, tách từ bằng `underthesea`, xử lý lệch nhãn
- Viết dưới dạng **hàm** để dễ test và tái sử dụng

---

### 👤 Điện (TV3)
**Viết báo cáo:**
- Mục 2.1: Cơ sở lý thuyết (Transformer, BERT, PhoBERT, Self-attention)
- Mục 3.1: Xây dựng mô hình (kiến trúc pipeline, fine-tuning, hyperparameter)

**Code:**
- Load và **fine-tuning PhoBERT**
- Viết DataLoader, cấu hình huấn luyện, Early Stopping

---

### 👤 Lân (TV4)
**Viết báo cáo:**
- Mục 2.2.3: Trích chọn đặc trưng
- Mục 3.3: Toàn bộ phần đánh giá mô hình (độ đo, bảng so sánh, ablation study, phân tích lỗi)

**Code:**
- Vẽ biểu đồ kết quả, **Confusion Matrix**
- Chạy so sánh mBERT, XLM-R vs PhoBERT
- Ablation study (chỉ Claim / chỉ Evidence / ghép cả hai)
- Phân tích 20–30 mẫu bị đoán sai

---

### 👤 Minh (TV5)
**Viết báo cáo:**
- Chương 4: Demo và triển khai (Streamlit, Evidence Retrieval, kết quả demo)
- Hỗ trợ Lân viết bảng so sánh và chụp ảnh minh họa ở mục 3.3

**Code:**
- Xây dựng **giao diện web Streamlit**
- Tích hợp mô hình PhoBERT vào inference thực tế
- Deploy lên Streamlit Cloud (nếu kịp)

---

## 🗓️ Timeline

| Giai đoạn | Nội dung | Deadline | Người thực hiện |
|-----------|----------|----------|----------------|
| GĐ 1 | Đọc paper, thống nhất bài toán, cài môi trường | 2/6 | Cả nhóm |
| GĐ 2 | Tải dataset, tiền xử lý, lưu Drive | 2/6 | TV2 (Nguyên) |
| GĐ 3 | Load PhoBERT, fine-tune, đóng gói hàm | 6/6 | TV3 (Điện) |
| GĐ 4 | Baseline, so sánh mô hình, ablation study | 7/6 | TV4 (Lân) |
| GĐ 5 | Inference, giao diện Streamlit, demo | 8/6 | TV5 (Minh) |
| GĐ 6 | Viết báo cáo, format Word | **9/6 – 21h30** | Cả nhóm |

---

## 📋 Nội dung báo cáo

```
CHƯƠNG I   – Tổng quan đề tài
  1.1  Mô tả vấn đề
  1.2  Mục tiêu nghiên cứu
  1.3  Phạm vi nghiên cứu
  1.4  Phương pháp tiếp cận
  1.5  Khảo sát nghiên cứu liên quan

CHƯƠNG II  – Cơ sở lý thuyết và dữ liệu
  2.1  Cơ sở lý thuyết (Transformer, BERT, PhoBERT)
  2.2  Cơ sở dữ liệu
    2.2.1  Thu thập dữ liệu
    2.2.2  Tiền xử lý dữ liệu
    2.2.3  Trích chọn đặc trưng

CHƯƠNG III – Xây dựng và đánh giá mô hình
  3.1  Xây dựng mô hình PhoBERT
  3.2  Các mô hình so sánh (TF-IDF, mBERT, XLM-R)
  3.3  Đánh giá mô hình
    3.3.1  Các độ đo đánh giá
    3.3.2  Kết quả so sánh
    3.3.3  Ablation study
    3.3.4  Phân tích kết quả và lỗi

CHƯƠNG IV  – Demo
  4.1  Evidence Retrieval
  4.2  Triển khai trên Streamlit
  4.3  Kết quả demo

CHƯƠNG V   – Kết luận và khuyến nghị
  5.1  Kết luận
  5.2  Khuyến nghị
```

---

## 🔧 Các quyết định kỹ thuật

| Quyết định | Lý do |
|------------|-------|
| Dùng **PhoBERT** làm mô hình chính | Pre-trained trên tiếng Việt, hiệu quả nhất cho NLP tiếng Việt |
| Dùng **TF-IDF + Logistic Regression** làm baseline | Đơn giản, dễ giải thích, làm chuẩn so sánh |
| So sánh thêm **mBERT và XLM-R** | Đánh giá đa chiều, tăng tính thuyết phục |
| **Ablation Study** trên Claim và Evidence riêng biệt | Phân tích đóng góp từng thành phần |
| Demo bằng **Streamlit** | Nhanh, dễ dùng, phù hợp demo học thuật |
| Môi trường: **Google Colab** | Miễn phí, có GPU, dễ chia sẻ trong nhóm |

---

*README được tạo bởi nhóm Giday – Môn Trí Tuệ Nhân Tạo*
