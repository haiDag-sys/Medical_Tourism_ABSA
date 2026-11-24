# -*- coding: utf-8 -*-
"""
DỰ ÁN: ĐÁNH GIÁ TRÔNG ĐỢI CỦA KHÁCH HÀNG TRÊN DỊCH VỤ DU LỊCH Y TẾ & NHA KHOA TẠI VIỆT NAM
MODULE: HUẤN LUYỆN MÔ HÌNH THẾ HỆ 2 (DENSE STATIC WORD EMBEDDINGS - WORD2VEC)
--------------------------------------------------------------------------------
Phương pháp:
1. Huấn luyện mô hình Word2Vec (Skip-Gram, vector_size=200) trên toàn bộ kho ngữ liệu
   đã tách từ bằng Underthesea.
2. Trích xuất đặc trưng câu (Document Representation):
   - Phương pháp A: Average Word Embeddings (Mean Pooling).
   - Phương pháp B: TF-IDF Weighted Word Embeddings (Weighted Average Pooling).
3. Huấn luyện và đối sánh các bộ phân loại: Logistic Regression, Support Vector Machine,
   và Random Forest trên không gian véc-tơ đậm đặc (Dense Vector Space).
4. Lưu mô hình Word2Vec và bộ phân loại tốt nhất vào thư mục Models/.
================================================================================
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from gensim.models import Word2Vec
from underthesea import word_tokenize

# Cấu hình UTF-8 cho console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "Combined_Medical_Tourism_TrainModel.csv"
MODELS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = BASE_DIR.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

W2V_MODEL_PATH = MODELS_DIR / "word2vec_medical.model"
GEN2_CLASSIFIER_PATH = MODELS_DIR / "gen2_word2vec_sentiment_model.pkl"
GEN2_TFIDF_WEIGHTS_PATH = MODELS_DIR / "gen2_tfidf_weights.pkl"

CLASSES = ["Negative", "Neutral", "Positive"]
VECTOR_SIZE = 200


def load_data():
    """Nạp dữ liệu đã tách từ chuẩn Underthesea."""
    print("=" * 80)
    print(" BẮT ĐẦU QUY TRÌNH THẾ HỆ 2: WORD2VEC DENSE EMBEDDINGS + CLASSIFIER")
    print("=" * 80)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")
        
    print(f">>> Đang nạp dữ liệu từ: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["content_segmented"].notna() & df["sentiment_label"].isin(CLASSES)].copy()
    df["content_segmented"] = df["content_segmented"].astype(str).str.strip()
    df = df[df["content_segmented"].str.len() > 3].reset_index(drop=True)
    
    print(f" -> Tổng số mẫu hợp lệ: {len(df):,}")
    return df


def train_word2vec(sentences_tokenized):
    """Huấn luyện mô hình Word2Vec (Skip-Gram) trên kho ngữ liệu miền y tế - du lịch."""
    print("\n>>> Đang huấn luyện mô hình Word2Vec (Skip-Gram, vector_size=200, window=5) ...")
    start_time = time.time()
    
    # sg=1: Skip-Gram (phù hợp trích xuất ngữ nghĩa sâu sắc cho từ hiếm và thuật ngữ y tế)
    w2v_model = Word2Vec(
        sentences=sentences_tokenized,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=15,
        seed=42
    )
    train_time = time.time() - start_time
    print(f" -> Hoàn tất huấn luyện Word2Vec trong {train_time:.2f}s. Kích thước từ vựng: {len(w2v_model.wv):,} từ.")
    w2v_model.save(str(W2V_MODEL_PATH))
    print(f" -> Đã lưu Word2Vec model tại: {W2V_MODEL_PATH}")
    
    # Thử nghiệm tính tương đồng ngữ nghĩa của một số từ khóa đặc trưng
    test_words = ["bác_sĩ", "chi_phí", "sạch_sẽ", "đau", "nhiệt_tình"]
    print("\n>>> Kiểm tra độ tương đồng ngữ nghĩa (Semantic Proximity Test):")
    for tw in test_words:
        if tw in w2v_model.wv:
            sims = w2v_model.wv.most_similar(tw, topn=3)
            sim_str = ", ".join([f"{w} ({s:.2f})" for w, s in sims])
            print(f"   * Từ '{tw}': gần nghĩa nhất với -> [{sim_str}]")
            
    return w2v_model


def get_document_vectors(sentences_tokenized, w2v_model, tfidf_weights=None):
    """
    Biến đổi từng câu văn bản thành một vector cố định d-chiều:
    Nếu có tfidf_weights -> Weighted Average Pooling.
    Nếu không có -> Mean Pooling đơn thuần.
    """
    vectors = []
    vocab = set(w2v_model.wv.index_to_key)
    
    for sent in sentences_tokenized:
        words = [w for w in sent if w in vocab]
        if not words:
            vectors.append(np.zeros(VECTOR_SIZE, dtype=np.float32))
            continue
            
        if tfidf_weights is not None:
            weights = [tfidf_weights.get(w, 1.0) for w in words]
            w_sum = sum(weights)
            if w_sum > 0:
                weighted_vec = sum(w2v_model.wv[w] * wt for w, wt in zip(words, weights)) / w_sum
                vectors.append(weighted_vec)
            else:
                vectors.append(np.mean([w2v_model.wv[w] for w in words], axis=0))
        else:
            vectors.append(np.mean([w2v_model.wv[w] for w in words], axis=0))
            
    return np.array(vectors, dtype=np.float32)


def benchmark_gen2_models(X_train_vec, X_test_vec, y_train, y_test):
    """Huấn luyện và đánh giá đối sánh các bộ phân loại trên không gian Word2Vec."""
    print("\n" + "=" * 80)
    print(" ĐỐI SÁNH CÁC BỘ PHÂN LOẠI TRÊN KHÔNG GIAN DENSE VÉC-TƠ (WORD2VEC)")
    print("=" * 80)
    
    models = {
        "Logistic Regression (W2V)": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Linear SVM (W2V)": LinearSVC(C=1.0, max_iter=2000, random_state=42),
        "Random Forest (W2V)": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    }
    
    results = []
    y_preds = {}
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    for name, clf in models.items():
        print(f"\n>>> Đang huấn luyện: [{name}] ...")
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        fit_time = time.time() - t0
        
        t1 = time.time()
        y_pred = clf.predict(X_test_vec)
        inf_time = (time.time() - t1) / len(X_test_vec) * 1000  # ms/mẫu
        
        y_preds[name] = y_pred
        acc = accuracy_score(y_test, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        
        print(f" -> Accuracy: {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}% | Suy luận: {inf_time:.3f} ms/mẫu")
        
        results.append({
            "Mô Hình": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Macro F1 (%)": round(f1_macro * 100, 2),
            "Weighted F1 (%)": round(f1_w * 100, 2),
            "Precision (Macro %)": round(prec_macro * 100, 2),
            "Recall (Macro %)": round(rec_macro * 100, 2),
            "Fit Time (s)": round(fit_time, 2),
            "Latency (ms/mẫu)": round(inf_time, 3)
        })
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_model = clf
            
    df_res = pd.DataFrame(results).sort_values(by="Macro F1 (%)", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 80)
    print(" BẢNG XẾP HẠNG THẾ HỆ 2 (WORD2VEC EMBEDDINGS):")
    print(df_res.to_string(index=False))
    print("=" * 80)
    print(f"\n🏆 Mô hình tốt nhất Thế hệ 2: [{best_model_name}] với Macro F1 = {best_f1*100:.2f}%")
    
    # Lưu mô hình tốt nhất
    joblib.dump(best_model, GEN2_CLASSIFIER_PATH)
    print(f" -> Đã lưu bộ phân loại tốt nhất tại: {GEN2_CLASSIFIER_PATH}")
    
    # In báo cáo phân loại chi tiết
    print("\n>>> Báo Cáo Phân Loại Chi Tiết (Classification Report) của mô hình dẫn đầu:")
    print(classification_report(y_test, y_preds[best_model_name], target_names=CLASSES, digits=4))
    
    return df_res, y_preds, best_model_name


def plot_gen2_results(df_results, y_preds, y_test):
    """Vẽ biểu đồ đối sánh và ma trận nhầm lẫn cho Thế hệ 2."""
    plt.figure(figsize=(10, 5))
    df_plot = pd.melt(
        df_results, 
        id_vars=["Mô Hình"], 
        value_vars=["Accuracy (%)", "Macro F1 (%)"],
        var_name="Chỉ Số", 
        value_name="Điểm Số (%)"
    )
    ax = sns.barplot(data=df_plot, x="Mô Hình", y="Điểm Số (%)", hue="Chỉ Số", palette=["#2b5c8f", "#d95f02"])
    plt.title("Hiệu Năng Các Bộ Phân Loại Thế Hệ 2 (Word2Vec Embeddings)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(50, 100)
    plt.xlabel("Mô Hình Phân Loại", fontsize=11)
    plt.ylabel("Điểm Số (%)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, fontweight="bold", xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plot_file = PLOTS_DIR / "gen2_word2vec_metrics.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f" -> Đã lưu biểu đồ Thế hệ 2 tại: {plot_file}")


def main():
    df = load_data()
    
    # Chuẩn bị danh sách từ cho Word2Vec
    sentences = [str(text).split() for text in df["content_segmented"]]
    
    # Huấn luyện mô hình Word2Vec
    w2v_model = train_word2vec(sentences)
    
    # Tính trọng số TF-IDF để làm TF-IDF Weighted Pooling
    print("\n>>> Đang tính trọng số TF-IDF cho từng từ ghép ...")
    tfidf = TfidfVectorizer(max_features=15000, sublinear_tf=True)
    tfidf.fit(df["content_segmented"])
    # Trọng số IDF cho từng từ
    idf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    joblib.dump(idf_weights, GEN2_TFIDF_WEIGHTS_PATH)
    
    # Chia tập train/test (cùng tỉ lệ 80/20 và random_state=42 để đối sánh công bằng)
    X = sentences
    y = df["sentiment_label"].values
    X_train_sent, X_test_sent, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n>>> Đang trích xuất Document Vectors (TF-IDF Weighted Average Pooling) ...")
    X_train_vec = get_document_vectors(X_train_sent, w2v_model, idf_weights)
    X_test_vec = get_document_vectors(X_test_sent, w2v_model, idf_weights)
    print(f" -> Kích thước ma trận Train: {X_train_vec.shape}")
    print(f" -> Kích thước ma trận Test:  {X_test_vec.shape}")
    
    df_results, y_preds, best_model_name = benchmark_gen2_models(X_train_vec, X_test_vec, y_train, y_test)
    plot_gen2_results(df_results, y_preds, y_test)
    
    print("\n" + "=" * 80)
    print(" HOÀN TẤT HUẤN LUYỆN THẾ HỆ 2 (WORD2VEC) THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
