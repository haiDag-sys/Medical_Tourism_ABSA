# -*- coding: utf-8 -*-
"""
================================================================================
MODULE HUẤN LUYỆN & SO SÁNH CÁC MÔ HÌNH PHÂN LOẠI CẢM XÚC (SENTIMENT BENCHMARK)
================================================================================
Đề tài: Đánh giá trông đợi của khách hàng trên dịch vụ du lịch y tế & nha khoa Việt Nam
Chức năng:
  1. Nạp tập dữ liệu cân bằng nhãn (Combined_Medical_Tourism_TrainModel.csv).
  2. Phân chia tập Train/Test (80/20) phân tầng (Stratified) bảo toàn tỷ lệ các lớp.
  3. Trích xuất đặc trưng văn bản bằng TF-IDF (1-gram & 2-gram, 5000 đặc trưng).
  4. Huấn luyện và đánh giá 4 thuật toán máy học phân loại:
     - Multinomial Naive Bayes (NB)
     - Logistic Regression (LR)
     - Linear Support Vector Machine (LinearSVC)
     - Random Forest Classifier (RF)
  5. Đánh giá toàn diện: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
  6. Xuất biểu đồ so sánh mô hình & Ma trận nhầm lẫn chuyên nghiệp vào Plots/.
  7. Lưu trữ mô hình tốt nhất (Best Model) và TF-IDF Vectorizer bằng Joblib.
================================================================================
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from underthesea import word_tokenize

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU & ĐỒ HỌA
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "Combined_Medical_Tourism_TrainModel.csv"
MODELS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = BASE_DIR.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODELS_DIR / "best_sentiment_model.pkl"
TFIDF_SAVE_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

plt.rcParams["font.sans-serif"] = ["Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["Negative", "Neutral", "Positive"]


def load_and_prepare_data():
    """Nạp dữ liệu cân bằng nhãn đã tách từ bằng Underthesea và chuẩn bị tập Train / Test."""
    print("=" * 80)
    print(" BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI CẢM XÚC (VỚI UNDERTHESEA)")
    print("=" * 80)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {DATA_PATH}")
        
    print(f">>> Đang nạp dữ liệu từ: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f" -> Tổng số dòng ban đầu: {len(df):,}")
    
    # Lọc các dòng hợp lệ có văn bản và nhãn cảm xúc
    df = df[df["content_clean"].notna() & df["sentiment_label"].isin(CLASSES)].copy()
    
    # Sử dụng cột content_segmented (Underthesea) nếu đã có, nếu chưa thì tự động tách từ
    if "content_segmented" not in df.columns:
        print(" -> Đang tiến hành tách từ ghép tiếng Việt bằng Underthesea...")
        def seg_func(row):
            t = str(row["content_clean"])
            if str(row.get("language", "vi")) == "en":
                return t
            try:
                return word_tokenize(t, format="text")
            except Exception:
                return t
        df["content_segmented"] = df.apply(seg_func, axis=1)
        df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
        print(" -> Đã lưu cột content_segmented vào file CSV.")
    else:
        print(" -> Đã tìm thấy cột [content_segmented] (Tách từ chuẩn Underthesea).")
        
    df["content_segmented"] = df["content_segmented"].astype(str).str.strip()
    df = df[df["content_segmented"].str.len() > 3].reset_index(drop=True)
    
    print(f" -> Số bản ghi hợp lệ đưa vào huấn luyện: {len(df):,}")
    print("\n>>> Phân bổ các lớp cảm xúc trong tập TrainModel:")
    label_dist = df["sentiment_label"].value_counts()
    for lbl, count in label_dist.items():
        print(f"   + {lbl:8s}: {count:,} ({count/len(df)*100:.2f}%)")
        
    X = df["content_segmented"]
    y = df["sentiment_label"]
    
    # Phân chia Train/Test 80/20 theo phân tầng (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n -> Phân chia dữ liệu: Train = {len(X_train):,} mẫu (80%) | Test = {len(X_test):,} mẫu (20%)")
    return X_train, X_test, y_train, y_test


def extract_features(X_train, X_test):
    """Trích xuất đặc trưng văn bản bằng TF-IDF n-gram (1-gram & 2-gram, 15,000 đặc trưng)."""
    print("\n>>> TIẾN HÀNH TRÍCH XUẤT ĐẶC TRƯNG TF-IDF (1-gram & 2-gram)...")
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    t0 = time.time()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    elapsed = time.time() - t0
    
    print(f" -> Hoàn tất TF-IDF trong {elapsed:.2f}s")
    print(f" -> Kích thước ma trận Train: {X_train_tfidf.shape[0]:,} mẫu x {X_train_tfidf.shape[1]:,} đặc trưng")
    print(f" -> Kích thước ma trận Test : {X_test_tfidf.shape[0]:,} mẫu x {X_test_tfidf.shape[1]:,} đặc trưng")
    
    # Lưu trước TF-IDF Vectorizer
    joblib.dump(tfidf, TFIDF_SAVE_PATH)
    print(f" -> Đã lưu TF-IDF Vectorizer tại: {TFIDF_SAVE_PATH}")
    
    return X_train_tfidf, X_test_tfidf, tfidf


def benchmark_models(X_train_vec, X_test_vec, y_train, y_test):
    """Huấn luyện và đánh giá đối sánh 4 mô hình máy học với cơ chế cân bằng trọng số lớp (class_weight='balanced')."""
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=42, max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=35, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    y_preds = {}
    
    print("\n" + "=" * 80)
    print(" TIẾN HÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ ĐỐI SÁNH 4 MÔ HÌNH")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n>>> Đang huấn luyện: {name} ...")
        t_start = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - t_start
        
        # Dự đoán trên tập Test
        y_pred = model.predict(X_test_vec)
        y_preds[name] = y_pred
        trained_models[name] = model
        
        # Tính toán các chỉ số đánh giá
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        print(f"   [Xong trong {train_time:.2f}s] Accuracy: {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}% | Weighted F1: {f1_weighted*100:.2f}%")
        
        results.append({
            "Mô hình": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision Macro (%)": round(prec_macro * 100, 2),
            "Recall Macro (%)": round(rec_macro * 100, 2),
            "F1-Score Macro (%)": round(f1_macro * 100, 2),
            "F1-Score Weighted (%)": round(f1_weighted * 100, 2),
            "Thời gian train (s)": round(train_time, 2)
        })
        
    df_results = pd.DataFrame(results).sort_values(by="F1-Score Macro (%)", ascending=False)
    
    print("\n" + "=" * 80)
    print(" BẢNG TỔNG HỢP HIỆU NĂNG CÁC MÔ HÌNH (MODEL COMPARISON)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # In báo cáo chi tiết từng lớp của mô hình tốt nhất
    best_model_name = df_results.iloc[0]["Mô hình"]
    best_model = trained_models[best_model_name]
    best_pred = y_preds[best_model_name]
    
    print("\n" + "=" * 80)
    print(f" BÁO CÁO CHI TIẾT TỪNG LỚP CỦA MÔ HÌNH TỐT NHẤT: [{best_model_name.upper()}]")
    print("=" * 80)
    print(classification_report(y_test, best_pred, target_names=CLASSES, digits=4))
    
    # Lưu mô hình tốt nhất
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f" -> ĐÃ LƯU MÔ HÌNH TỐT NHẤT [{best_model_name}] TẠI: {MODEL_SAVE_PATH}")
    
    return df_results, y_preds, best_model_name


def plot_metrics_and_confusion_matrices(df_results, y_preds, y_test):
    """Trực quan hóa so sánh hiệu năng và Ma trận nhầm lẫn (Confusion Matrix)."""
    print("\n>>> ĐANG VẼ BIỂU ĐỒ ĐỐI SÁNH HIỆU NĂNG VÀ MA TRẬN NHẦM LẪN...")
    
    # 1. Biểu đồ cột so sánh các chỉ số giữa 4 mô hình
    plt.figure(figsize=(11, 6), dpi=300)
    
    model_names = df_results["Mô hình"].tolist()
    x = np.arange(len(model_names))
    width = 0.2
    
    acc_vals = df_results["Accuracy (%)"].tolist()
    prec_vals = df_results["Precision Macro (%)"].tolist()
    rec_vals = df_results["Recall Macro (%)"].tolist()
    f1_vals = df_results["F1-Score Macro (%)"].tolist()
    
    plt.bar(x - 1.5 * width, acc_vals, width, label="Accuracy", color="#1f77b4")
    plt.bar(x - 0.5 * width, prec_vals, width, label="Precision (Macro)", color="#ff7f0e")
    plt.bar(x + 0.5 * width, rec_vals, width, label="Recall (Macro)", color="#2ca02c")
    plt.bar(x + 1.5 * width, f1_vals, width, label="F1-Score (Macro)", color="#d62728")
    
    plt.title("So Sánh Hiệu Năng Các Mô Hình Phân Loại Cảm Xúc (NLP Sentiment Benchmark)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Thuật toán máy học", fontsize=11, fontweight="bold")
    plt.ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
    plt.xticks(x, model_names, fontsize=10)
    plt.ylim(60, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    
    # Ghi số liệu lên đỉnh cột F1-Score
    for i, v in enumerate(f1_vals):
        plt.text(x[i] + 1.5 * width, v + 0.5, f"{v}%", ha="center", fontsize=9, fontweight="bold", color="darkred")
        
    plt.tight_layout()
    metrics_path = PLOTS_DIR / "model_comparison_metrics.png"
    plt.savefig(metrics_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ so sánh mô hình tại: {metrics_path}")
    
    # 2. Ma trận nhầm lẫn (Confusion Matrix) cho cả 4 mô hình trong lưới 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    for idx, (name, y_pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Tạo nhãn kết hợp số lượng và phần trăm
        annot = np.empty_like(cm).astype(str)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]:,}\n({cm_pct[r, c]:.1f}%)"
                
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=axes[idx],
            annot_kws={"fontsize": 9, "fontweight": "bold"}
        )
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=10)
        axes[idx].set_ylabel("Nhãn Thực Tế (Actual)", fontsize=10)
        
    plt.suptitle("Ma Trận Nhầm Lẫn Đối Sánh Giữa 4 Mô Hình Học Máy", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    cm_path = PLOTS_DIR / "confusion_matrix_all_models.png"
    plt.savefig(cm_path)
    plt.close()
    print(f" -> Đã lưu Ma trận nhầm lẫn tại: {cm_path}")


def test_live_inference():
    """Kiểm tra suy luận thực tế (Live Inference) với mô hình tốt nhất vừa lưu."""
    print("\n" + "=" * 80)
    print(" KIỂM THỬ SUY LUẬN THỰC TẾ TRÊN MẪU MỚI (LIVE PREDICTION TEST)")
    print("=" * 80)
    
    if not MODEL_SAVE_PATH.exists() or not TFIDF_SAVE_PATH.exists():
        return
        
    best_model = joblib.load(MODEL_SAVE_PATH)
    tfidf = joblib.load(TFIDF_SAVE_PATH)
    
    test_samples = [
        "Bác sĩ nhổ răng khôn rất nhẹ nhàng, tay nghề cao, không hề đau chút nào, nhân viên nhiệt tình!",
        "Chờ đợi cả buổi sáng từ 7h đến 11h mới được khám, thủ tục quá rườm rà và thái độ rất khó chịu.",
        "Cho em hỏi chi phí bọc răng sứ thẩm mỹ ở đây hết bao nhiêu tiền và có hỗ trợ trả góp không ạ?",
        "The dental clinic in Vietnam is amazing! Dr. Nam did a full implant bridge, painless and saved me thousands of dollars.",
        "Avoid this clinic, they overcharged me and the crown broke after just 2 weeks. Very disappointed!"
    ]
    
    # Tách từ tiếng Việt trước khi đưa vào véc-tơ hóa
    test_samples_seg = [word_tokenize(t, format="text") for t in test_samples]
    test_vec = tfidf.transform(test_samples_seg)
    preds = best_model.predict(test_vec)
    
    for text, seg, pred in zip(test_samples, test_samples_seg, preds):
        print(f"\n[Câu gốc]: \"{text}\"")
        print(f" -> Tách từ (Underthesea): \"{seg}\"")
        color = "🟢" if pred == "Positive" else ("🔴" if pred == "Negative" else "🟡")
        print(f" -> {color} Dự đoán cảm xúc: [{pred}]")


def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    X_train_vec, X_test_vec, tfidf = extract_features(X_train, X_test)
    df_results, y_preds, best_model_name = benchmark_models(X_train_vec, X_test_vec, y_train, y_test)
    plot_metrics_and_confusion_matrices(df_results, y_preds, y_test)
    test_live_inference()
    
    print("\n" + "=" * 80)
    print(" HOÀN TẤT BƯỚC 1: HUẤN LUYỆN, ĐÁNH GIÁ ĐỐI SÁNH VÀ LƯU TRỮ MÔ HÌNH THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
