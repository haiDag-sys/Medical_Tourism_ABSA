# Đánh Giá Trông Đợi Du Khách Trên Dịch Vụ Du Lịch Y Tế & Nha Khoa Tại Việt Nam
### Aspect-Based Sentiment Analysis (ABSA) with Hybrid AI & Mixture of Experts (MoE)

Hệ thống Trí tuệ Nhân tạo kết hợp (*Hybrid AI*) phục vụ phân tích cảm xúc đa khía cạnh (*ABSA*) và khám phá tri thức trông đợi của du khách trên các dịch vụ du lịch y tế và nha khoa tại Việt Nam.

---

## 🏛️ Kiến Trúc Thư Mục Dự Án (Clean Production Architecture)

```text
Medical_Tourism_ABSA/
├── app/                                # Máy chủ dịch vụ FastAPI (Kiến trúc Module)
│   ├── main.py                         # Khởi tạo API Endpoints, CORS và Web Routing
│   ├── model_registry.py               # Singleton nạp trước các mô hình AI vào RAM
│   ├── pipeline.py                     # Tiền xử lý NMT, chuẩn hóa từ viết tắt, Intent Filter
│   └── moe_router.py                   # Bộ định tuyến tách mệnh đề & 6 Aspect Experts (MoE)
│
├── training/                           # Mã nguồn nghiên cứu khoa học & huấn luyện AI
│   ├── train_sentiment_models.py       # Huấn luyện Thế hệ 1: TF-IDF + Linear SVM
│   ├── train_gen2_word2vec.py          # Huấn luyện Thế hệ 2: Word2Vec + Random Forest (Bagging)
│   ├── train_gen3_phobert.py           # Huấn luyện Thế hệ 3: PhoBERT-base-v2 + MLP SOTA
│   ├── benchmark_all_generations.py    # Thực nghiệm đối chuẩn trên 3,973 mẫu kiểm thử độc lập
│   └── cluster_kmeans_analysis.py      # Phân cụm K-Means (K=5) khám phá 5 nhóm trông đợi du khách
│
├── models/                             # Trọng số mô hình đã huấn luyện (.pkl, .model < 25MB)
│   ├── best_sentiment_model.pkl        # Trọng số Linear SVM đã hiệu chuẩn Platt Scaling
│   ├── tfidf_vectorizer.pkl            # Bộ trích xuất ma trận từ vựng 10,000 đặc trưng
│   ├── word2vec_medical.model          # Mô hình Word2Vec Skip-Gram 200 chiều miền y tế
│   ├── gen2_word2vec_sentiment_model.pkl # Bộ phân loại Random Forest (100 cây)
│   ├── gen2_tfidf_weights.pkl          # Trọng số IDF phục vụ TF-IDF Weighted Pooling
│   └── gen3_phobert_classifier.pkl     # Mạng nơ-ron đa tầng MLP (GELU, Dropout 0.3)
│
├── data/                               # Bộ ngữ liệu chuẩn hóa & Mẫu kiểm thử
│   ├── Combined_Medical_Tourism_TrainModel.csv # 23,838 mẫu cân bằng nhãn huấn luyện
│   ├── Combined_Medical_Tourism_Clustered.csv  # 39,359 mẫu phục vụ phân cụm K-Means
│   ├── Combined_Medical_Tourism_EDA.csv        # 41,985 mẫu thô thu thập từ đa nguồn
│   ├── Test_20_DanhGia_KhachHang.csv           # File mẫu test nhanh tính năng Web (CSV)
│   └── Test_30_DanhGia_SongNgu_Anh_Viet.xlsx   # File mẫu test song ngữ Anh - Việt (Excel)
│
├── static/                             # Giao diện Web Dashboard (HTML5, CSS3, JS)
│   ├── index.html                      # Bảng điều khiển phân tích thời gian thực
│   ├── style.css                       # Thiết kế phong cách Y tế - Lâm sàng hiện đại
│   └── app.js                          # Xử lý tương tác AJAX và trực quan hóa dữ liệu
│
├── plots/                              # Biểu đồ nghiên cứu khoa học phục vụ báo cáo
│   ├── He_Thong_Kien_Truc_Tong_The.png # Sơ đồ kiến trúc toàn diện của hệ thống
│   ├── pipeline_hybrid_architecture.png # Luồng tiền xử lý đa tầng
│   ├── benchmark_3_generations_metrics.png # Biểu đồ đối chuẩn 3 thế hệ NLP
│   ├── confusion_matrix_3_generations.png  # Ma trận nhầm lẫn chi tiết
│   └── kmeans_clusters_2d.png          # Không gian 2D biểu diễn 5 cụm tâm lý du khách
│
├── .gitignore                          # Cấu hình loại bỏ cache và môi trường ảo
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md                           # Hướng dẫn chi tiết dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### 1. Cài đặt môi trường Python
Khuyến nghị sử dụng Python 3.10 hoặc 3.11:
```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
venv\Scripts\activate   # Trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Khởi chạy Máy chủ Dịch vụ & Web Dashboard
Chạy lệnh uvicorn từ thư mục gốc của dự án:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Truy cập hệ thống trên trình duyệt:
*   🌐 **Giao diện Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
*   📚 **Tài liệu API Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔬 Kết Quả Thực Nghiệm Đối Chuẩn (Benchmark)
Thực nghiệm trên **3,973 mẫu kiểm thử độc lập** (phân tầng tỷ lệ 5:1 từ 23,838 mẫu chuẩn hóa):

| Thế hệ / Mô hình | Độ chính xác (Accuracy) | Macro-F1 | Độ trễ suy luận (Latency) |
| :--- | :---: | :---: | :---: |
| **Thế hệ 1 – Linear SVM (TF-IDF)** | 82.15% | 80.40% | ~1.2 ms |
| **Thế hệ 2 – Word2Vec + Random Forest** | 81.65% | 79.10% | ~3.8 ms |
| **Thế hệ 3 – PhoBERT + MLP SOTA** | 85.02% | 83.45% | ~18.5 ms |
| **Mô hình Kết hợp (Ensemble Soft Voting)** | **88.95%** | **86.12%** | ~22.0 ms |
| **Kiến trúc MoE (Aspect-level Accuracy)** | **89.20%** | **87.50%** | ~25.4 ms |

---

## 👨‍💻 Tác Giả & Bản Quyền
*   **Sinh viên thực hiện:** Nguyễn Hải Đăng - MSSV: B2308351
*   **Đơn vị:** Ngành Khoa học Máy tính – Trường Công nghệ Thông tin & Truyền thông – Đại học Cần Thơ.
