# -*- coding: utf-8 -*-
"""
DỰ ÁN: ĐÁNH GIÁ TRÔNG ĐỢI CỦA KHÁCH HÀNG TRÊN DỊCH VỤ DU LỊCH Y TẾ & NHA KHOA TẠI VIỆT NAM
MODULE: ĐỐI SÁNH TOÀN DIỆN 3 THẾ HỆ XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP BENCHMARK REPORT)
--------------------------------------------------------------------------------
Đối tượng đối sánh:
1. Thế hệ 1 (Statistical / Bag-of-Words): TF-IDF N-grams (1-2) + LinearSVC
2. Thế hệ 2 (Dense Static Embeddings):    Word2Vec (Skip-Gram) + TF-IDF Pooling + Random Forest
3. Thế hệ 3 (Contextual Transformers):    PhoBERT-base-v2 [CLS] + Logistic/Linear Classifier

Tất cả mô hình được đánh giá trên cùng một tập Test chuẩn (Stratified 20%, random_state=42)
nhằm đảm bảo tính khách quan và chuẩn mực theo phương pháp luận nghiên cứu khoa học.
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
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
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
PLOTS_DIR = BASE_DIR.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# File mô hình các thế hệ
GEN1_MODEL_PATH = MODELS_DIR / "best_sentiment_model.pkl"
GEN1_TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

GEN2_W2V_PATH = MODELS_DIR / "word2vec_medical.model"
GEN2_MODEL_PATH = MODELS_DIR / "gen2_word2vec_sentiment_model.pkl"
GEN2_IDF_PATH = MODELS_DIR / "gen2_tfidf_weights.pkl"

GEN3_MODEL_PATH = MODELS_DIR / "gen3_phobert_classifier.pkl"
GEN3_CACHE_PATH = BASE_DIR.parent / "data" / "phobert_embeddings_cache.npz"

CLASSES = ["Negative", "Neutral", "Positive"]
PHOBERT_NAME = "vinai/phobert-base-v2"


def load_raw_test_data():
    """Tải và chia tập Test chuẩn 20% giống hệt như các script huấn luyện."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["content_segmented"].notna() & df["sentiment_label"].isin(CLASSES)].copy()
    df["content_segmented"] = df["content_segmented"].astype(str).str.strip()
    df = df[df["content_segmented"].str.len() > 3].reset_index(drop=True)
    
    indices = np.arange(len(df))
    y = df["sentiment_label"].values
    
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )
    
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_test, y_test, test_idx


def evaluate_gen1(df_test, y_test):
    """Đánh giá Thế hệ 1: TF-IDF + LinearSVC."""
    print("\n>>> [1/3] Đang đánh giá Thế hệ 1: TF-IDF N-grams + LinearSVC ...")
    if not GEN1_MODEL_PATH.exists() or not GEN1_TFIDF_PATH.exists():
        raise FileNotFoundError("Chưa tìm thấy mô hình Thế hệ 1 tại Models/")
        
    model = joblib.load(GEN1_MODEL_PATH)
    tfidf = joblib.load(GEN1_TFIDF_PATH)
    
    t0 = time.time()
    X_test_vec = tfidf.transform(df_test["content_segmented"])
    t_feat = time.time() - t0
    
    t1 = time.time()
    y_pred = model.predict(X_test_vec)
    t_pred = time.time() - t1
    
    latency = (t_feat + t_pred) / len(y_test) * 1000
    acc = accuracy_score(y_test, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f" -> Accuracy: {acc*100:.2f}% | Macro F1: {f1_m*100:.2f}% | Latency: {latency:.3f} ms/mẫu")
    return {
        "Thế Hệ": "Thế hệ 1 (Bag-of-Words)",
        "Đại Diện Kỹ Thuật": "TF-IDF N-grams (1-2) + LinearSVC",
        "Accuracy (%)": round(acc * 100, 2),
        "Macro F1 (%)": round(f1_m * 100, 2),
        "Weighted F1 (%)": round(f1_w * 100, 2),
        "Precision (%)": round(p_m * 100, 2),
        "Recall (%)": round(r_m * 100, 2),
        "Latency (ms)": round(latency, 3),
        "y_pred": y_pred
    }


def evaluate_gen2(df_test, y_test):
    """Đánh giá Thế hệ 2: Word2Vec + TF-IDF Weighted Pooling + Classifier."""
    print("\n>>> [2/3] Đang đánh giá Thế hệ 2: Word2Vec Skip-Gram + Pooling + Classifier ...")
    if not GEN2_W2V_PATH.exists() or not GEN2_MODEL_PATH.exists():
        raise FileNotFoundError("Chưa tìm thấy mô hình Thế hệ 2 tại Models/")
        
    w2v = Word2Vec.load(str(GEN2_W2V_PATH))
    model = joblib.load(GEN2_MODEL_PATH)
    idf_weights = joblib.load(GEN2_IDF_PATH) if GEN2_IDF_PATH.exists() else None
    
    vocab = set(w2v.wv.index_to_key)
    vector_size = w2v.vector_size
    
    t0 = time.time()
    vectors = []
    for sent in df_test["content_segmented"]:
        words = [w for w in str(sent).split() if w in vocab]
        if not words:
            vectors.append(np.zeros(vector_size, dtype=np.float32))
            continue
        if idf_weights:
            weights = [idf_weights.get(w, 1.0) for w in words]
            w_sum = sum(weights)
            if w_sum > 0:
                weighted_vec = sum(w2v.wv[w] * wt for w, wt in zip(words, weights)) / w_sum
                vectors.append(weighted_vec)
            else:
                vectors.append(np.mean([w2v.wv[w] for w in words], axis=0))
        else:
            vectors.append(np.mean([w2v.wv[w] for w in words], axis=0))
            
    X_test_vec = np.array(vectors, dtype=np.float32)
    t_feat = time.time() - t0
    
    t1 = time.time()
    y_pred = model.predict(X_test_vec)
    t_pred = time.time() - t1
    
    latency = (t_feat + t_pred) / len(y_test) * 1000
    acc = accuracy_score(y_test, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f" -> Accuracy: {acc*100:.2f}% | Macro F1: {f1_m*100:.2f}% | Latency: {latency:.3f} ms/mẫu")
    return {
        "Thế Hệ": "Thế hệ 2 (Static Embeddings)",
        "Đại Diện Kỹ Thuật": "Word2Vec (Skip-Gram) + Random Forest",
        "Accuracy (%)": round(acc * 100, 2),
        "Macro F1 (%)": round(f1_m * 100, 2),
        "Weighted F1 (%)": round(f1_w * 100, 2),
        "Precision (%)": round(p_m * 100, 2),
        "Recall (%)": round(r_m * 100, 2),
        "Latency (ms)": round(latency, 3),
        "y_pred": y_pred
    }


def evaluate_gen3(test_idx, y_test):
    """Đánh giá Thế hệ 3: PhoBERT Contextual Embeddings + Classifier."""
    print("\n>>> [3/3] Đang đánh giá Thế hệ 3: Pre-trained PhoBERT-base-v2 Contextual Transformers ...")
    if not GEN3_MODEL_PATH.exists():
        raise FileNotFoundError("Chưa tìm thấy mô hình Thế hệ 3 tại Models/")
        
    model = joblib.load(GEN3_MODEL_PATH)
    
    if GEN3_CACHE_PATH.exists():
        cache_data = np.load(GEN3_CACHE_PATH)
        X_all_emb = cache_data["embeddings"]
        X_test_emb = X_all_emb[test_idx]
    else:
        raise FileNotFoundError("Vui lòng chạy Train_Gen3_PhoBERT.py để trích xuất véc-tơ PhoBERT trước.")
        
    t1 = time.time()
    y_pred = model.predict(X_test_emb)
    t_pred = time.time() - t1
    
    latency = t_pred / len(y_test) * 1000
    acc = accuracy_score(y_test, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f" -> Accuracy: {acc*100:.2f}% | Macro F1: {f1_m*100:.2f}% | Latency: {latency:.3f} ms/mẫu")
    return {
        "Thế Hệ": "Thế hệ 3 (Contextual Transformers)",
        "Đại Diện Kỹ Thuật": "PhoBERT-base-v2 [CLS] + Classifier",
        "Accuracy (%)": round(acc * 100, 2),
        "Macro F1 (%)": round(f1_m * 100, 2),
        "Weighted F1 (%)": round(f1_w * 100, 2),
        "Precision (%)": round(p_m * 100, 2),
        "Recall (%)": round(r_m * 100, 2),
        "Latency (ms)": round(latency, 3),
        "y_pred": y_pred
    }


def plot_benchmark_charts(res_list, y_test):
    """Vẽ biểu đồ đối sánh toàn diện và Confusion Matrix của 3 thế hệ."""
    df_bench = pd.DataFrame(res_list)
    
    # 1. Biểu đồ cột so sánh Accuracy, Macro F1, Precision, Recall
    plt.figure(figsize=(12, 6))
    df_plot = pd.melt(
        df_bench,
        id_vars=["Thế Hệ"],
        value_vars=["Accuracy (%)", "Macro F1 (%)", "Precision (%)", "Recall (%)"],
        var_name="Chỉ Số Đánh Giá",
        value_name="Điểm Số (%)"
    )
    
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax = sns.barplot(data=df_plot, x="Thế Hệ", y="Điểm Số (%)", hue="Chỉ Số Đánh Giá", palette=palette)
    plt.title("ĐỐI SÁNH HIỆU NĂNG 3 THẾ HỆ BIỂU DIỄN NLP TRÊN DỮ LIỆU DU LỊCH Y TẾ", fontsize=13, fontweight="bold", pad=15)
    plt.ylim(65, 100)
    plt.xlabel("Thế Hệ Công Nghệ NLP", fontsize=11, fontweight="bold")
    plt.ylabel("Điểm Số (%)", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight="bold", xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    chart_path = PLOTS_DIR / "benchmark_3_generations_metrics.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n -> Đã lưu biểu đồ đối sánh tại: {chart_path}")
    
    # 2. Ma trận nhầm lẫn đối sánh cạnh nhau
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm_palettes = ["Blues", "Oranges", "Greens"]
    
    for idx, res in enumerate(res_list):
        cm = confusion_matrix(y_test, res["y_pred"], labels=CLASSES)
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        annot_matrix = np.empty_like(cm).astype(str)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
                
        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap=cm_palettes[idx],
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=axes[idx],
            annot_kws={"fontsize": 9, "fontweight": "bold"}
        )
        axes[idx].set_title(f"{res['Thế Hệ']}\nAccuracy: {res['Accuracy (%)']}% | F1: {res['Macro F1 (%)']}%", fontsize=11, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=10)
        axes[idx].set_ylabel("Nhãn Thực Tế (Actual)", fontsize=10)
        
    plt.suptitle("Ma Trận Nhầm Lẫn Đối Sánh Giữa 3 Thế Hệ (TF-IDF vs Word2Vec vs PhoBERT)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_all_path = PLOTS_DIR / "confusion_matrix_3_generations.png"
    plt.savefig(cm_all_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Đã lưu ma trận nhầm lẫn đối sánh tại: {cm_all_path}")


def qualitative_test():
    """Thử nghiệm chất lượng thực tế trên các câu thách thức (phủ định, đa nghĩa, từ lóng y tế)."""
    print("\n" + "=" * 80)
    print(" ĐỐI SÁNH ĐỊNH TÍNH (QUALITATIVE TEST): THỬ THÁCH NGỮ CẢNH PHỨC TẠP")
    print("=" * 80)
    
    test_cases = [
        "Bác sĩ nhổ răng khôn không hề đau chút nào, thao tác siêu nhanh và tận tình.",
        "Dịch vụ ở đây chẳng có điểm gì để chê, nhân viên đón tiếp chu đáo từ A đến Z.",
        "Tưởng giá rẻ mà đến nơi kê đơn phụ phí linh tinh gấp đôi, bực mình vô cùng.",
        "Cho mình hỏi chi phí cấy ghép Implant Straumann tại bệnh viện có được bảo hiểm y tế chi trả không?",
        "Bác sĩ khám nhanh quá, chưa kịp hỏi gì đã cho đơn thuốc đi về."
    ]
    
    # Nạp cả 3 mô hình
    gen1_clf = joblib.load(GEN1_MODEL_PATH)
    gen1_tfidf = joblib.load(GEN1_TFIDF_PATH)
    
    gen2_clf = joblib.load(GEN2_MODEL_PATH)
    gen2_w2v = Word2Vec.load(str(GEN2_W2V_PATH))
    gen2_idf = joblib.load(GEN2_IDF_PATH) if GEN2_IDF_PATH.exists() else None
    
    gen3_clf = joblib.load(GEN3_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(PHOBERT_NAME)
    phobert = AutoModel.from_pretrained(PHOBERT_NAME)
    phobert.eval()
    
    for sentence in test_cases:
        seg = word_tokenize(sentence, format="text")
        
        # Gen 1
        g1_vec = gen1_tfidf.transform([seg])
        g1_pred = gen1_clf.predict(g1_vec)[0]
        
        # Gen 2
        words = [w for w in seg.split() if w in gen2_w2v.wv]
        if words and gen2_idf:
            weights = [gen2_idf.get(w, 1.0) for w in words]
            w_sum = sum(weights)
            g2_vec = sum(gen2_w2v.wv[w] * wt for w, wt in zip(words, weights)) / w_sum
        elif words:
            g2_vec = np.mean([gen2_w2v.wv[w] for w in words], axis=0)
        else:
            g2_vec = np.zeros(gen2_w2v.vector_size)
        g2_pred = gen2_clf.predict([g2_vec])[0]
        
        # Gen 3
        enc = tok([seg], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            g3_emb = phobert(enc["input_ids"], enc["attention_mask"]).last_hidden_state[:, 0, :].numpy()
        g3_pred = gen3_clf.predict(g3_emb)[0]
        
        print(f"\n[Câu]: \"{sentence}\"")
        print(f"  + Thế hệ 1 (TF-IDF):    [{g1_pred}]")
        print(f"  + Thế hệ 2 (Word2Vec):  [{g2_pred}]")
        print(f"  + Thế hệ 3 (PhoBERT):   [{g3_pred}]")


def main():
    print("=" * 80)
    print(" BẮT ĐẦU QUY TRÌNH ĐỐI SÁNH TOÀN DIỆN 3 THẾ HỆ XỬ LÝ NGÔN NGỮ TỰ NHIÊN")
    print("=" * 80)
    
    df_test, y_test, test_idx = load_raw_test_data()
    print(f" -> Kích thước tập Test độc lập: {len(df_test):,} mẫu")
    
    res1 = evaluate_gen1(df_test, y_test)
    res2 = evaluate_gen2(df_test, y_test)
    res3 = evaluate_gen3(test_idx, y_test)
    
    res_list = [res1, res2, res3]
    df_summary = pd.DataFrame([{k: v for k, v in r.items() if k != "y_pred"} for r in res_list])
    
    print("\n" + "=" * 80)
    print(" BẢNG TỔNG HỢP SO SÁNH 3 THẾ HỆ:")
    print(df_summary.to_string(index=False))
    print("=" * 80)
    
    plot_benchmark_charts(res_list, y_test)
    qualitative_test()
    
    print("\n" + "=" * 80)
    print(" HOÀN TẤT ĐỐI SÁNH TOÀN DIỆN 3 THẾ HỆ THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
