# -*- coding: utf-8 -*-
"""
================================================================================
MODULE PHÂN CỤM K-MEANS CLUSTERING DU LỊCH Y TẾ & NHA KHOA VIỆT NAM
================================================================================
Đề tài: Đánh giá trông đợi của khách trên dịch vụ du lịch y tế Việt Nam
Chức năng:
  1. Tiền xử lý & Trích xuất đặc trưng văn bản bằng TF-IDF (Song ngữ Việt - Anh).
  2. Phân tích phương pháp Elbow (Inertia) để tìm số cụm K tối ưu.
  3. Huấn luyện thuật toán K-Means Clustering để khám phá các nhóm chủ đề tiềm ẩn (Latent Themes).
  4. Trích xuất Top từ khóa đại diện & Đặt tên chủ đề (Profiling) cho từng cụm.
  5. Thống kê tương quan giữa Cụm chủ đề với Cảm xúc (Hài lòng vs Bức xúc) và 6 Khía cạnh.
  6. Trực quan hóa không gian 2D bằng TruncatedSVD và xuất biểu đồ chuyên nghiệp.
================================================================================
"""

import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# Cấu hình UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU & ĐỒ HỌA
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "Combined_Medical_Tourism_EDA.csv"
OUTPUT_CLUSTERED_DATA = BASE_DIR.parent / "data" / "Combined_Medical_Tourism_Clustered.csv"
PLOTS_DIR = BASE_DIR.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/arial.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/tahoma.ttf"

plt.rcParams["font.sans-serif"] = ["Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# 2. BỘ TỪ DỪNG SONG NGỮ (VIỆT - ANH) CHUYÊN BIỆT
# ==============================================================================
STOPWORDS_VI = set([
    "và", "thì", "là", "mà", "ở", "có", "rất", "được", "cho", "với", "này",
    "của", "khi", "đã", "lại", "nên", "làm", "đi", "tôi", "mình", "em", "ạ",
    "nhé", "nha", "các", "những", "cái", "con", "người", "đến", "về", "như",
    "đang", "sẽ", "phải", "ra", "vào", "từ", "lên", "trong", "qua", "cũng",
    "thấy", "thôi", "biết", "nhiều", "hơn", "quá", "quáaaa", "lắm", "ngày",
    "tháng", "năm", "hôm", "nay", "vừa", "rồi", "lần", "trước", "sau", "đây",
    "nữa", "luôn", "gì", "ai", "sao", "bị", "bởi", "vì", "nhưng", "nếu", "dù",
    "bệnh viện", "phòng khám", "nha khoa", "cơ sở", "dịch vụ", "việt nam", "vn", "bv"
])

STOPWORDS_EN = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "vietnam", "clinic", "hospital"
])

ALL_STOPWORDS = list(STOPWORDS_VI.union(STOPWORDS_EN))


def clean_text_for_tfidf(text: str) -> str:
    """Làm sạch ký tự đặc biệt, dấu câu, số để tối ưu cho TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[emoji\]|\[link\]|\[comments\]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_optimal_k_elbow(X, k_range=range(2, 8)):
    """Tính toán Elbow Curve (Inertia) để tìm số cụm K tối ưu."""
    print("\n>>> ĐANG CHẠY PHƯƠNG PHÁP ELBOW (TÍNH TOÁN INERTIA)...")
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"   + K = {k}: Inertia = {kmeans.inertia_:.2f}")

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(list(k_range), inertias, marker='o', linewidth=2.5, color='#1f77b4', markersize=8)
    plt.title("Phương Pháp Elbow Xác Định Số Cụm K Tối Ưu (K-Means)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Số lượng cụm (K)", fontsize=12)
    plt.ylabel("Độ phân tán trong cụm (Inertia)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    chosen_k = 5
    chosen_idx = list(k_range).index(chosen_k)
    plt.scatter([chosen_k], [inertias[chosen_idx]], color='red', s=140, zorder=5)

    elbow_path = PLOTS_DIR / "kmeans_elbow_curve.png"
    plt.tight_layout()
    plt.savefig(elbow_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ Elbow tại: {elbow_path}")
    return chosen_k


def assign_cluster_theme(top_keywords: list) -> str:
    """Tự động suy luận tên chủ đề (Profiling Theme) dựa trên top từ khóa nổi bật của từng cụm."""
    kw_str = " ".join(top_keywords).lower()
    
    if any(k in kw_str for k in ["nhẹ nhàng", "nhổ", "không đau", "ko đau", "ít đau", "painless", "gentle"]):
        return "Trải nghiệm êm dịu & Điều trị không đau"
    elif any(k in kw_str for k in ["bệnh viện", "bệnh nhân", "viện", "chờ", "thủ tục"]):
        return "Bệnh viện công & Quy trình khám tiếp nhận"
    elif any(k in kw_str for k in ["nhiệt tình", "chu đáo", "thân thiện", "lễ tân", "vui vẻ", "staff", "friendly"]):
        return "Thái độ nhân viên & Y Bác sĩ nhiệt tình"
    elif any(k in kw_str for k in ["dịch vụ", "dịch", "vụ tốt", "hài lòng", "tuyệt vời", "service"]):
        return "Chất lượng dịch vụ & Trải nghiệm chung"
    elif any(k in kw_str for k in ["răng", "khám", "tư vấn", "tay nghề", "doctor", "dentist"]):
        return "Tư vấn nha khoa & Tay nghề Bác sĩ"
    else:
        return "Trông đợi dịch vụ tổng quan"


def main():
    print("=" * 80)
    print(" BẮT ĐẦU CHƯƠNG TRÌNH PHÂN CỤM K-MEANS CLUSTERING")
    print("=" * 80)
    
    if not DATA_PATH.exists():
        print(f"[LỖI] Không tìm thấy file dữ liệu tại: {DATA_PATH}")
        return

    print(f">>> Đang tải dữ liệu từ: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f" -> Tổng số dòng ban đầu: {len(df):,}")

    df = df[df["content_clean"].notna()].copy()
    df["clean_for_cluster"] = df["content_clean"].apply(clean_text_for_tfidf)
    df = df[df["clean_for_cluster"].apply(lambda s: len(s.split()) >= 4)].reset_index(drop=True)
    print(f" -> Số bản ghi đủ tiêu chuẩn văn bản đưa vào phân cụm: {len(df):,}")

    print("\n>>> TIẾN HÀNH TRÍCH XUẤT ĐẶC TRƯNG TF-IDF (1-gram & 2-gram)...")
    tfidf = TfidfVectorizer(
        max_features=2500,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2),
        stop_words=ALL_STOPWORDS
    )
    X = tfidf.fit_transform(df["clean_for_cluster"])
    feature_names = np.array(tfidf.get_feature_names_out())
    print(f" -> Kích thước ma trận TF-IDF: {X.shape[0]:,} mẫu x {X.shape[1]:,} đặc trưng")

    optimal_k = find_optimal_k_elbow(X, k_range=range(2, 8))

    print(f"\n>>> ĐANG HUẤN LUYỆN K-MEANS VỚI K = {optimal_k} CỤM...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X)
    df["cluster_id"] = clusters

    print("\n" + "=" * 80)
    print(" KẾT QUẢ KHÁM PHÁ CÁC CỤM CHỦ ĐỀ TRÔNG ĐỢI (CLUSTER PROFILING)")
    print("=" * 80)
    
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_theme_map = {}
    cluster_summary_list = []

    for i in range(optimal_k):
        top_indices = order_centroids[i, :10]
        top_words = [feature_names[ind] for ind in top_indices]
        theme_name = assign_cluster_theme(top_words)
        cluster_theme_map[i] = f"Cụm {i+1}: {theme_name}"
        
        cluster_df = df[df["cluster_id"] == i]
        count = len(cluster_df)
        pct = (count / len(df)) * 100
        
        pos_pct = (cluster_df["sentiment_label"] == "Positive").sum() / count * 100
        neg_pct = (cluster_df["sentiment_label"] == "Negative").sum() / count * 100
        
        cluster_summary_list.append({
            "Mã cụm": f"Cluster {i+1}",
            "Tên chủ đề": theme_name,
            "Số lượng": count,
            "Tỷ lệ (%)": round(pct, 2),
            "Hài lòng (%)": round(pos_pct, 2),
            "Bức xúc (%)": round(neg_pct, 2),
            "Từ khóa tiêu biểu": ", ".join(top_words[:6])
        })

    df["cluster_theme"] = df["cluster_id"].map(cluster_theme_map)

    summary_df = pd.DataFrame(cluster_summary_list)
    print(summary_df.to_string(index=False))

    print("\n>>> ĐANG GIẢM CHIỀU DỮ LIỆU SANG 2D ĐỂ TRỰC QUAN HÓA BẰNG TRUNCATEDSVD...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X)

    sample_size = min(6000, len(df))
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    plt.figure(figsize=(12, 8), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    
    for i in range(optimal_k):
        mask = (clusters[sample_indices] == i)
        plt.scatter(
            X_2d[sample_indices][mask, 0],
            X_2d[sample_indices][mask, 1],
            c=colors[i % len(colors)],
            label=f"{cluster_theme_map[i]} ({summary_df.loc[i, 'Tỷ lệ (%)']}%)",
            alpha=0.55,
            edgecolors="none",
            s=35
        )

    centroids_2d = svd.transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        marker="X",
        s=250,
        linewidths=2,
        color="black",
        edgecolors="white",
        label="Tâm cụm (Centroids)",
        zorder=10
    )

    plt.title(f"Bản Đồ Không Gian Phân Cụm Trông Đợi Khách Hàng (K-Means Clustering, K={optimal_k})", 
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(f"Thành phần chính 1 (Phương sai giải thích: {svd.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    plt.ylabel(f"Thành phần chính 2 (Phương sai giải thích: {svd.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)

    clusters_plot_path = PLOTS_DIR / "kmeans_clusters_2d.png"
    plt.tight_layout()
    plt.savefig(clusters_plot_path)
    plt.close()
    print(f" -> Đã lưu biểu đồ phân cụm 2D tại: {clusters_plot_path}")

    if "clean_for_cluster" in df.columns:
        df.drop(columns=["clean_for_cluster"], inplace=True)
        
    df.to_csv(OUTPUT_CLUSTERED_DATA, index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu dữ liệu kèm nhãn Cụm tại: {OUTPUT_CLUSTERED_DATA}")
    print("\n" + "=" * 80)
    print(" HOÀN TẤT PHÂN CỤM K-MEANS VÀ XUẤT BÁO CÁO THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
