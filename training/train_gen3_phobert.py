# -*- coding: utf-8 -*-
"""
DỰ ÁN: ĐÁNH GIÁ TRÔNG ĐỢI CỦA KHÁCH HÀNG TRÊN DỊCH VỤ DU LỊCH Y TẾ & NHA KHOA TẠI VIỆT NAM
MODULE: HUẤN LUYỆN MÔ HÌNH THẾ HỆ 3 (CONTEXTUAL PRE-TRAINED TRANSFORMERS - PHOBERT)
--------------------------------------------------------------------------------
Phương pháp:
1. Sử dụng mô hình PhoBERT-base-v2 (VinAI Research) - mô hình ngôn ngữ ngữ cảnh
   tiền huấn luyện hàng đầu cho tiếng Việt (dựa trên kiến trúc RoBERTa).
2. Mã hóa từng câu đánh giá đã tách từ Underthesea thành véc-tơ ngữ cảnh 768 chiều
   thông qua token đại diện [CLS] (Contextual Feature Extraction).
3. Lưu bộ nhớ đệm (Cache) véc-tơ PhoBERT vào file Data/phobert_embeddings.npz để tái sử dụng tức thì.
4. Huấn luyện và đối sánh các bộ phân loại tối ưu (Logistic Regression, Support Vector Machine,
   Neural Classifier) trên không gian đặc trưng ngữ cảnh của PhoBERT.
5. Lưu mô hình phân loại tốt nhất vào Models/gen3_phobert_classifier.pkl.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize

# Cấu hình UTF-8 cho console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "Combined_Medical_Tourism_TrainModel.csv"
EMBEDDINGS_CACHE_PATH = BASE_DIR.parent / "data" / "phobert_embeddings_cache.npz"
MODELS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = BASE_DIR.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GEN3_CLASSIFIER_PATH = MODELS_DIR / "gen3_phobert_classifier.pkl"
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 128
BATCH_SIZE = 64
CLASSES = ["Negative", "Neutral", "Positive"]


def load_data(limit=None):
    """Nạp dữ liệu đã tách từ chuẩn Underthesea."""
    print("=" * 80)
    print(" BẮT ĐẦU QUY TRÌNH THẾ HỆ 3: CONTEXTUAL TRANSFORMERS (PHOBERT-BASE-V2)")
    print("=" * 80)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")
        
    print(f">>> Đang nạp dữ liệu từ: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["content_segmented"].notna() & df["sentiment_label"].isin(CLASSES)].copy()
    df["content_segmented"] = df["content_segmented"].astype(str).str.strip()
    df = df[df["content_segmented"].str.len() > 3].reset_index(drop=True)
    
    if limit and limit < len(df):
        print(f" -> [Chế độ thử nghiệm nhanh]: Lấy mẫu ngẫu nhiên {limit:,} bản ghi cân bằng nhãn...")
        df = df.groupby("sentiment_label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), limit // len(CLASSES)), random_state=42)
        ).reset_index(drop=True)
    
    print(f" -> Tổng số mẫu hợp lệ: {len(df):,}")
    return df


def extract_phobert_embeddings(texts, batch_size=BATCH_SIZE):
    """
    Trích xuất đặc trưng ngữ cảnh (Contextual Embeddings) từ PhoBERT thông qua [CLS] token.
    Sử dụng PyTorch tối ưu đa luồng CPU và xử lý theo batch.
    """
    print(f"\n>>> Khởi tạo PhoBERT Tokenizer và Model: [{PHOBERT_MODEL_NAME}] ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Thiết bị tính toán: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
    model.to(device)
    model.eval()
    
    all_embeddings = []
    total_samples = len(texts)
    print(f">>> Bắt đầu trích xuất [CLS] Embeddings ({total_samples:,} mẫu, batch_size={batch_size}) ...")
    t0 = time.time()
    
    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Lấy đặc trưng của token đầu tiên [CLS] (shape: [batch_size, 768])
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
        if (i // batch_size) % 25 == 0 or (i + batch_size) >= total_samples:
            elapsed = time.time() - t0
            progress = min(i + batch_size, total_samples)
            pct = progress / total_samples * 100
            print(f"   * Tiến trình: {progress:,}/{total_samples:,} mẫu ({pct:.1f}%) - Thời gian: {elapsed:.1f}s")
            
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    total_time = time.time() - t0
    print(f" -> Hoàn tất trích xuất đặc trưng trong {total_time:.1f}s. Kích thước ma trận: {embeddings_matrix.shape}")
    return embeddings_matrix


def get_or_create_embeddings(df):
    """Nạp véc-tơ từ cache nếu đã có, hoặc tạo mới và lưu lại."""
    if EMBEDDINGS_CACHE_PATH.exists():
        print(f"\n>>> Tìm thấy bộ nhớ đệm đặc trưng PhoBERT tại: {EMBEDDINGS_CACHE_PATH}")
        print(" -> Đang tải véc-tơ ngữ cảnh từ bộ nhớ đệm (Cache) ...")
        t0 = time.time()
        data = np.load(EMBEDDINGS_CACHE_PATH)
        X_embeddings = data["embeddings"]
        y_labels = data["labels"]
        print(f" -> Tải thành công trong {time.time()-t0:.2f}s! Kích thước: {X_embeddings.shape}")
        return X_embeddings, y_labels
    else:
        texts = df["content_segmented"].tolist()
        y_labels = df["sentiment_label"].values
        X_embeddings = extract_phobert_embeddings(texts)
        print(f">>> Đang lưu véc-tơ đặc trưng vào bộ nhớ đệm: {EMBEDDINGS_CACHE_PATH} ...")
        np.savez_compressed(EMBEDDINGS_CACHE_PATH, embeddings=X_embeddings, labels=y_labels)
        print(" -> Đã lưu bộ nhớ đệm thành công.")
        return X_embeddings, y_labels


def benchmark_gen3_models(X_train_emb, X_test_emb, y_train, y_test):
    """Huấn luyện và đối sánh các bộ phân loại tối ưu trên đặc trưng ngữ cảnh PhoBERT."""
    print("\n" + "=" * 80)
    print(" ĐỐI SÁNH CÁC BỘ PHÂN LOẠI TRÊN KHÔNG GIAN NGỮ CẢNH TRANSFORMERS (PHOBERT)")
    print("=" * 80)
    
    models = {
        "Logistic Regression (PhoBERT)": LogisticRegression(max_iter=1000, C=1.5, random_state=42),
        "Linear SVM (PhoBERT)": LinearSVC(C=0.8, max_iter=2000, random_state=42),
        "MLP Classifier (PhoBERT)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42, early_stopping=True)
    }
    
    results = []
    y_preds = {}
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    for name, clf in models.items():
        print(f"\n>>> Đang huấn luyện: [{name}] ...")
        t0 = time.time()
        clf.fit(X_train_emb, y_train)
        fit_time = time.time() - t0
        
        t1 = time.time()
        y_pred = clf.predict(X_test_emb)
        inf_time = (time.time() - t1) / len(X_test_emb) * 1000  # ms/mẫu
        
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
    print(" BẢNG XẾP HẠNG THẾ HỆ 3 (PHOBERT CONTEXTUAL EMBEDDINGS):")
    print(df_res.to_string(index=False))
    print("=" * 80)
    print(f"\n🏆 Mô hình tốt nhất Thế hệ 3: [{best_model_name}] với Macro F1 = {best_f1*100:.2f}%")
    
    # Lưu mô hình tốt nhất
    joblib.dump(best_model, GEN3_CLASSIFIER_PATH)
    print(f" -> Đã lưu bộ phân loại tốt nhất tại: {GEN3_CLASSIFIER_PATH}")
    
    # In báo cáo phân loại chi tiết
    print("\n>>> Báo Cáo Phân Loại Chi Tiết (Classification Report) của mô hình dẫn đầu:")
    print(classification_report(y_test, y_preds[best_model_name], target_names=CLASSES, digits=4))
    
    return df_res, y_preds, best_model_name


def plot_gen3_results(df_results, y_preds, y_test):
    """Vẽ biểu đồ đối sánh cho Thế hệ 3."""
    plt.figure(figsize=(10, 5))
    df_plot = pd.melt(
        df_results, 
        id_vars=["Mô Hình"], 
        value_vars=["Accuracy (%)", "Macro F1 (%)"],
        var_name="Chỉ Số", 
        value_name="Điểm Số (%)"
    )
    ax = sns.barplot(data=df_plot, x="Mô Hình", y="Điểm Số (%)", hue="Chỉ Số", palette=["#1a9850", "#7570b3"])
    plt.title("Hiệu Năng Các Bộ Phân Loại Thế Hệ 3 (PhoBERT Contextual Embeddings)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(60, 100)
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
    plot_file = PLOTS_DIR / "gen3_phobert_metrics.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f" -> Đã lưu biểu đồ Thế hệ 3 tại: {plot_file}")


def test_complex_context_live():
    """Kiểm tra khả năng hiểu ngữ cảnh phức tạp, phủ định và đảo nghĩa của PhoBERT."""
    print("\n" + "=" * 80)
    print(" KIỂM THỬ KHẢ NĂNG HIỂU NGỮ CẢNH PHỨC TẠP CỦA PHOBERT")
    print(" (Xử lý phủ định: 'không hề đau', 'chẳng có gì phàn nàn', 'tưởng rẻ mà hóa đắt')")
    print("=" * 80)
    
    if not GEN3_CLASSIFIER_PATH.exists():
        return
        
    clf = joblib.load(GEN3_CLASSIFIER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
    model.eval()
    
    test_cases = [
        "Bác sĩ nhổ răng khôn không hề đau đớn chút nào, tay nghề quá đỉnh!",
        "Dịch vụ ở đây chẳng có điểm gì để phàn nàn cả, mọi thứ đều trên cả tuyệt vời.",
        "Quảng cáo thì hoành tráng nhưng làm ăn như đùa cợt, tưởng rẻ hóa ra phụ phí gấp đôi.",
        "Cho em hỏi làm cầu răng sứ nguyên hàm ở đây chi phí dao động khoảng bao nhiêu?",
        "Bác sĩ khám qua loa, thái độ coi thường bệnh nhân và bắt chờ cả buổi sáng."
    ]
    
    for sentence in test_cases:
        # Tách từ tiếng Việt bằng Underthesea trước khi đưa vào PhoBERT
        segmented = word_tokenize(sentence, format="text")
        encoded = tokenizer([segmented], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            emb = model(encoded["input_ids"], encoded["attention_mask"]).last_hidden_state[:, 0, :].numpy()
        pred = clf.predict(emb)[0]
        color = "🟢" if pred == "Positive" else ("🔴" if pred == "Negative" else "🟡")
        print(f"\n[Câu]: \"{sentence}\"")
        print(f" -> Underthesea: \"{segmented}\"")
        print(f" -> {color} PhoBERT dự đoán: [{pred}]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại cảm xúc Thế hệ 3 với PhoBERT")
    parser.add_argument("--fast", action="store_true", help="Chạy thử nghiệm nhanh trên 1,500 mẫu đại diện")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu nạp vào")
    args = parser.parse_args()
    
    sample_limit = 1500 if args.fast else args.limit
    df = load_data(limit=sample_limit)
    X_embeddings, y_labels = get_or_create_embeddings(df)
    
    # Chia tập train/test cùng tỉ lệ 80/20 và random_state=42 để đối sánh công bằng
    X_train_emb, X_test_emb, y_train, y_test = train_test_split(
        X_embeddings, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"\n>>> Dữ liệu phân tách:")
    print(f" -> Train: {X_train_emb.shape}")
    print(f" -> Test:  {X_test_emb.shape}")
    
    df_results, y_preds, best_model_name = benchmark_gen3_models(X_train_emb, X_test_emb, y_train, y_test)
    plot_gen3_results(df_results, y_preds, y_test)
    test_complex_context_live()
    
    print("\n" + "=" * 80)
    print(" HOÀN TẤT HUẤN LUYỆN THẾ HỆ 3 (PHOBERT TRANSFORMERS) THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
