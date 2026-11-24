# -*- coding: utf-8 -*-
"""
================================================================================
CHƯƠNG TRÌNH HỌC KẾT HỢP (ENSEMBLE LEARNING) & HỖN HỢP CHUYÊN GIA (MIXTURE OF EXPERTS - MoE)
PHÂN TÍCH CẢM XÚC ĐA THẾ HỆ & BÓC TÁCH KHÍA CẠNH DỊCH VỤ DU LỊCH Y TẾ - NHA KHOA
================================================================================
Tác giả  : Nguyễn Hải Đăng - MSSV: B2308351
Đề tài   : Đánh giá trông đợi của khách hàng trên dịch vụ du lịch y tế & nha khoa tại VN
Căn cứ   : 
  - Khuyến nghị của GVHD: "khen và chê theo khía cạnh thì mình có thể dùng nhiều
    mô hình tổng hợp cũng được, mixture of expert / Bagging/ Boosting, Ensemble method"
  - Mục 2.1.3, 2.2.4, 3.2.4, 3.3.4 trong Luận văn / Báo cáo Niên luận.

Kiến trúc triển khai:
  1. HETEROGENEOUS ENSEMBLE CLASSIFIER (Soft Voting có trọng số):
     - Model 1: TF-IDF N-grams (1-2) + Linear SVM (w1 = 0.30)
     - Model 2: Word2Vec Skip-Gram + Random Forest [Bagging] (w2 = 0.20)
     - Model 3: PhoBERT-base-v2 [CLS] + MLP Classifier [Deep Learning] (w3 = 0.50)
     -> Công thức: P_ensemble(y=c|x) = 0.30*P_SVM + 0.20*P_RF + 0.50*P_PhoBERT

  2. ASPECT-BASED MIXTURE OF EXPERTS (MoE):
     - Gating Router: Nhận diện liên từ đối lập (nhưng, tuy nhiên, song, mặc dù...),
       phân rã câu phức thành các mệnh đề độc lập.
     - Định tuyến từng mệnh đề tới 6 Aspect Experts tương ứng:
         [E1] Bác sĩ & Chuyên môn
         [E2] Thái độ nhân viên & Lễ tân
         [E3] Cơ sở vật chất & Thiết bị
         [E4] Hiệu quả điều trị & Không đau
         [E5] Chi phí & Tính minh bạch
         [E6] Thời gian & Tiếp nhận
     - Aggregator: Tổng hợp từ điển {Khía cạnh: Cảm xúc}, bảo toàn trọn vẹn cả vế
       khen ngợi lẫn vế phê bình trong cùng một câu đánh giá.
================================================================================
"""

import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import joblib
from scipy.special import softmax
from underthesea import word_tokenize

# Cấu hình UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stdin, "reconfigure"):
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thư mục gốc & đường dẫn mô hình
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

GEN1_MODEL_PATH = MODELS_DIR / "best_sentiment_model.pkl"
GEN1_TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

GEN2_W2V_PATH   = MODELS_DIR / "word2vec_medical.model"
GEN2_MODEL_PATH = MODELS_DIR / "gen2_word2vec_sentiment_model.pkl"
GEN2_IDF_PATH   = MODELS_DIR / "gen2_tfidf_weights.pkl"

GEN3_MODEL_PATH = MODELS_DIR / "gen3_phobert_classifier.pkl"
PHOBERT_NAME    = "vinai/phobert-base-v2"

# Danh mục nhãn chuẩn
CLASS_LABELS = ["Negative", "Neutral", "Positive"]

# ==============================================================================
# 1. BỘ TỪ ĐIỂN CHUẨN HÓA VIẾT TẮT & TEENCODE MIỀN Y TẾ
# ==============================================================================
ACRONYM_DICT = {
    r"\bcsvc\b": "cơ sở vật chất",
    r"\bco so vat chat\b": "cơ sở vật chất",
    r"\bcs\b": "cơ sở",
    r"\bpk\b": "phòng khám",
    r"\bpkđk\b": "phòng khám đa khoa",
    r"\bphong kham\b": "phòng khám",
    r"\bbv\b": "bệnh viện",
    r"\bbvien\b": "bệnh viện",
    r"\bbenh vien\b": "bệnh viện",
    r"\bbs\b": "bác sĩ",
    r"\bbsi\b": "bác sĩ",
    r"\bb/s\b": "bác sĩ",
    r"\bns\b": "nha sĩ",
    r"\bnv\b": "nhân viên",
    r"\bnvien\b": "nhân viên",
    r"\blt\b": "lễ tân",
    r"\ble tan\b": "lễ tân",
    r"\bbhyt\b": "bảo hiểm y tế",
    r"\bbao hiem y te\b": "bảo hiểm y tế",
    r"\bko\b": "không",
    r"\bk\b": "không",
    r"\bkh\b": "không",
    r"\bkhum\b": "không",
    r"\bhơm\b": "không",
    r"\bhong\b": "không",
    r"\bdc\b": "được",
    r"\bđc\b": "được",
    r"\bntn\b": "như thế nào",
    r"\bmn\b": "mọi người",
    r"\brv\b": "đánh giá",
    r"\btn\b": "thế nào",
    r"\bokie?\b": "tốt",
    r"\boki\b": "tốt",
    r"\bfeedback\b": "phản hồi"
}

def normalize_medical_text(text: str) -> str:
    """Chuẩn hóa viết tắt (bs -> bác sĩ, csvc -> cơ sở vật chất...)"""
    normalized = text
    for pattern, replacement in ACRONYM_DICT.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


# ==============================================================================
# 2. ĐỊNH NGHĨA 6 KHÍA CẠNH DỊCH VỤ SERVQUAL & TỪ KHÓA CHUYÊN GIA (ASPECT EXPERTS)
# ==============================================================================
ASPECT_EXPERTS = {
    "Bác sĩ & Chuyên môn": {
        "id": "E_doctor",
        "description": "Đánh giá tay nghề, trình độ chuyên khoa, sự tận tâm, tư vấn của bác sĩ",
        "keywords": [
            "bác sĩ", "bs", "nha sĩ", "tay nghề", "chuyên môn", "kinh nghiệm", "tận tâm",
            "tư vấn", "mát tay", "nhiệt tình", "giải thích", "khám kỹ", "y đức",
            "doctor", "dentist", "surgeon", "skilled", "professional", "consultation"
        ]
    },
    "Thái độ nhân viên & Lễ tân": {
        "id": "E_staff",
        "description": "Đánh giá thái độ đón tiếp, hướng dẫn của lễ tân, y tá, điều dưỡng",
        "keywords": [
            "nhân viên", "lễ tân", "y tá", "điều dưỡng", "phụ tá", "thái độ", "nhiệt tình",
            "chu đáo", "thân thiện", "lịch sự", "niềm nở", "khó chịu", "hách dịch", "lồi lõm",
            "coi thường", "staff", "nurse", "reception", "receptionist", "friendly", "helpful"
        ]
    },
    "Cơ sở vật chất & Thiết bị": {
        "id": "E_facility",
        "description": "Đánh giá không gian phòng khám, độ vô trùng, trang thiết bị máy móc hiện đại",
        "keywords": [
            "cơ sở", "phòng khám", "bệnh viện", "máy móc", "thiết bị", "công nghệ",
            "sạch sẽ", "vô trùng", "hiện đại", "khang trang", "chật", "bí bách", "bẩn",
            "ẩm mốc", "ghế nha", "clinic", "hospital", "facility", "clean", "sterile", "modern"
        ]
    },
    "Hiệu quả điều trị & Không đau": {
        "id": "E_quality",
        "description": "Đánh giá kết quả bọc răng sứ, niềng răng, nhổ răng khôn, kiểm soát cảm giác đau",
        "keywords": [
            "kết quả", "răng", "implant", "sứ", "bọc răng", "nhổ răng", "trám", "niềng",
            "không đau", "nhẹ nhàng", "êm ái", "ít đau", "đau đớn", "sưng", "chảy máu",
            "buốt", "ê buốt", "ăn nhai", "hài lòng", "result", "painless", "gentle", "tooth"
        ]
    },
    "Chi phí & Tính minh bạch": {
        "id": "E_cost",
        "description": "Đánh giá giá cả dịch vụ, bảng giá niêm yết, sự minh bạch, chi phí phát sinh",
        "keywords": [
            "chi phí", "giá", "giá cả", "tiền", "rẻ", "đắt", "hợp lý", "bảng giá",
            "bảo hiểm", "chặt chém", "phát sinh", "tiền nào của nấy", "phí", "đắt đỏ",
            "vừa túi tiền", "cost", "price", "cheap", "affordable", "expensive", "reasonable"
        ]
    },
    "Thời gian & Tiếp nhận": {
        "id": "E_time",
        "description": "Đánh giá thời gian chờ đợi, thủ tục đặt hẹn, tiến độ làm việc",
        "keywords": [
            "thời gian", "chờ", "đợi", "chờ lâu", "nhanh", "chậm", "đặt hẹn", "lịch hẹn",
            "thủ tục", "lấy số", "đúng giờ", "trễ hẹn", "mất thời gian", "time", "wait",
            "waiting", "schedule", "appointment", "quick", "fast", "delayed"
        ]
    }
}

# Các liên từ đối lập dùng để chia mệnh đề (Contrasting Conjunctions)
CONTRAST_CONJUNCTIONS = [
    r"\bnhưng mà\b", r"\bnhưng\b", r"\btuy nhiên\b", r"\bmặc dù\b", r"\bsong\b",
    r"\bthế nhưng\b", r"\btuy vậy\b", r"\bdẫu vậy\b", r"\bmỗi tội\b", r"\bcó điều\b",
    r"\bchỉ có điều\b", r"\bcòn\b(?=.*\bthì\b)", r"\btrái lại\b", r"\bngược lại\b"
]


# ==============================================================================
# 3. CLASS 1: HETEROGENEOUS ENSEMBLE VOTING CLASSIFIER
# ==============================================================================
class EnsembleVotingClassifier:
    """
    Mô hình Học kết hợp (Heterogeneous Soft Voting Ensemble):
    Kết hợp 3 mô hình độc lập về mặt toán học:
      - Gen 1: Linear SVM (Softmax trên decision_function, w1 = 0.30)
      - Gen 2: Word2Vec + Random Forest Bagging (predict_proba, w2 = 0.20)
      - Gen 3: PhoBERT-base-v2 + MLP Classifier (predict_proba, w3 = 0.50)
    """

    def __init__(self, w1: float = 0.30, w2: float = 0.20, w3: float = 0.50):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.classes = CLASS_LABELS
        
        # Biến giữ mô hình
        self.gen1_model = None
        self.gen1_tfidf = None
        
        self.gen2_model = None
        self.gen2_w2v = None
        self.gen2_idf = None
        
        self.gen3_model = None
        self.gen3_tok = None
        self.gen3_phobert = None
        
        self.loaded_status = {"gen1": False, "gen2": False, "gen3": False}
        self._load_all_models()

    def _load_all_models(self):
        """Tải các mô hình từ đĩa vào RAM."""
        # 1. Gen 1: TF-IDF + LinearSVC
        if GEN1_MODEL_PATH.exists() and GEN1_TFIDF_PATH.exists():
            try:
                self.gen1_model = joblib.load(GEN1_MODEL_PATH)
                self.gen1_tfidf = joblib.load(GEN1_TFIDF_PATH)
                self.loaded_status["gen1"] = True
            except Exception as e:
                print(f"[Cảnh báo] Lỗi nạp Gen 1: {e}")

        # 2. Gen 2: Word2Vec + Random Forest
        if GEN2_MODEL_PATH.exists() and GEN2_W2V_PATH.exists():
            try:
                from gensim.models import Word2Vec
                self.gen2_model = joblib.load(GEN2_MODEL_PATH)
                self.gen2_w2v = Word2Vec.load(str(GEN2_W2V_PATH))
                if GEN2_IDF_PATH.exists():
                    self.gen2_idf = joblib.load(GEN2_IDF_PATH)
                self.loaded_status["gen2"] = True
            except Exception as e:
                print(f"[Cảnh báo] Lỗi nạp Gen 2: {e}")

        # 3. Gen 3: PhoBERT + MLP
        if GEN3_MODEL_PATH.exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                self.gen3_model = joblib.load(GEN3_MODEL_PATH)
                try:
                    self.gen3_tok = AutoTokenizer.from_pretrained(PHOBERT_NAME, local_files_only=True)
                    self.gen3_phobert = AutoModel.from_pretrained(PHOBERT_NAME, local_files_only=True)
                except Exception:
                    self.gen3_tok = AutoTokenizer.from_pretrained(PHOBERT_NAME)
                    self.gen3_phobert = AutoModel.from_pretrained(PHOBERT_NAME)
                self.gen3_phobert.eval()
                self.loaded_status["gen3"] = True
            except Exception as e:
                print(f"[Cảnh báo] Lỗi nạp Gen 3: {e}")

    def _get_gen1_proba(self, seg_text: str) -> np.ndarray:
        """Tính phân phối xác suất Gen 1 qua Softmax(decision_function)."""
        if not self.loaded_status["gen1"]:
            return np.array([1/3, 1/3, 1/3])
        vec = self.gen1_tfidf.transform([seg_text])
        df = self.gen1_model.decision_function(vec)
        if df.ndim == 1:
            # Nhị phân
            probs = softmax(np.vstack([-df, df]).T, axis=1)[0]
        else:
            probs = softmax(df, axis=1)[0]
        return probs

    def _get_gen2_proba(self, seg_text: str) -> np.ndarray:
        """Tính phân phối xác suất Gen 2 qua Random Forest predict_proba."""
        if not self.loaded_status["gen2"]:
            return np.array([1/3, 1/3, 1/3])
        words = [w for w in seg_text.split() if w in self.gen2_w2v.wv]
        if not words:
            v = np.zeros(self.gen2_w2v.vector_size)
        elif self.gen2_idf:
            weights = [self.gen2_idf.get(w, 1.0) for w in words]
            w_sum = sum(weights)
            v = sum(self.gen2_w2v.wv[w] * wt for w, wt in zip(words, weights)) / (w_sum if w_sum > 0 else 1.0)
        else:
            v = np.mean([self.gen2_w2v.wv[w] for w in words], axis=0)
        return self.gen2_model.predict_proba([v])[0]

    def _get_gen3_proba(self, seg_text: str) -> np.ndarray:
        """Tính phân phối xác suất Gen 3 qua PhoBERT [CLS] + MLP predict_proba."""
        if not self.loaded_status["gen3"]:
            return np.array([1/3, 1/3, 1/3])
        import torch
        enc = self.gen3_tok([seg_text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            emb = self.gen3_phobert(enc["input_ids"], enc["attention_mask"]).last_hidden_state[:, 0, :].numpy()
        return self.gen3_model.predict_proba(emb)[0]

    def predict_proba(self, text: str) -> Dict[str, Any]:
        """
        Thực hiện Soft Voting có trọng số:
        P_ensemble = w1*P1 + w2*P2 + w3*P3
        """
        norm_text = normalize_medical_text(text)
        try:
            seg_text = word_tokenize(norm_text, format="text")
        except Exception:
            seg_text = norm_text

        p1 = self._get_gen1_proba(seg_text)
        p2 = self._get_gen2_proba(seg_text)
        p3 = self._get_gen3_proba(seg_text)

        # Trọng số thực tế dựa trên các mô hình đã tải thành công
        active_weights = []
        prob_list = []
        if self.loaded_status["gen1"]:
            active_weights.append(self.w1)
            prob_list.append(p1)
        if self.loaded_status["gen2"]:
            active_weights.append(self.w2)
            prob_list.append(p2)
        if self.loaded_status["gen3"]:
            active_weights.append(self.w3)
            prob_list.append(p3)

        if not prob_list:
            # Không có mô hình nào nạp được
            return {
                "ensemble_label": "Positive",
                "confidence": 0.333,
                "probabilities": {"Negative": 0.333, "Neutral": 0.334, "Positive": 0.333},
                "breakdown": {}
            }

        total_w = sum(active_weights)
        norm_w = [w / total_w for w in active_weights]
        p_ensemble = sum(w * p for w, p in zip(norm_w, prob_list))

        best_idx = int(np.argmax(p_ensemble))
        best_label = self.classes[best_idx]
        confidence = float(p_ensemble[best_idx])

        return {
            "ensemble_label": best_label,
            "confidence": round(confidence, 4),
            "probabilities": {cls: round(float(p_ensemble[i]), 4) for i, cls in enumerate(self.classes)},
            "breakdown": {
                "gen1_svm": {cls: round(float(p1[i]), 4) for i, cls in enumerate(self.classes)} if self.loaded_status["gen1"] else None,
                "gen2_rf":  {cls: round(float(p2[i]), 4) for i, cls in enumerate(self.classes)} if self.loaded_status["gen2"] else None,
                "gen3_phobert": {cls: round(float(p3[i]), 4) for i, cls in enumerate(self.classes)} if self.loaded_status["gen3"] else None,
            }
        }

    def predict(self, text: str) -> str:
        """Dự đoán trực tiếp nhãn cảm xúc của mô hình Ensemble."""
        return self.predict_proba(text)["ensemble_label"]


# ==============================================================================
# 4. CLASS 2: ASPECT-BASED MIXTURE OF EXPERTS ROUTER (MoE)
# ==============================================================================
class AspectMoERouter:
    """
    Kiến trúc Hỗn hợp Chuyên gia theo khía cạnh (Aspect-Based Mixture of Experts):
      1. Router (Bộ định tuyến): Phân tích liên từ đối lập và chia nhỏ câu thành
         các mệnh đề độc lập; định tuyến từng mệnh đề tới Expert tương ứng.
      2. 6 Aspect Experts: Mỗi chuyên gia đánh giá cảm xúc độc lập trên ngữ cảnh
         của khía cạnh đó.
      3. Aggregator (Bộ tổng hợp): Ghép nối thành bản đồ {Khía cạnh: Cảm xúc}.
    """

    def __init__(self, ensemble_classifier: EnsembleVotingClassifier):
        self.classifier = ensemble_classifier
        self.aspects_config = ASPECT_EXPERTS

    def split_into_clauses(self, text: str) -> List[str]:
        """
        Bóc tách câu phức thành các phân đoạn mệnh đề độc lập theo liên từ đối lập,
        dấu phẩy, chấm phẩy, chấm câu, hoặc liên từ liên kết (và, còn...).
        Ví dụ: "Bác sĩ rất mát tay, cơ sở vật chất ok, giá hơi cao" ->
               ["Bác sĩ rất mát tay", "cơ sở vật chất ok", "giá hơi cao"]
        """
        # 1. Thay thế các liên từ đối lập bằng dấu phân tách đặc biệt <SEP>
        pattern = "|".join(CONTRAST_CONJUNCTIONS)
        delimited = re.sub(pattern, "<SEP>", text, flags=re.IGNORECASE)
        
        # 2. Chia tách theo <SEP>, dấu phẩy ,, chấm phẩy ;, chấm câu .!?, hoặc liên từ và/còn nối vế
        raw_clauses = re.split(r"<SEP>|[,;!?\n]|\s+\bvà\b\s+|\s+\bcòn\b\s+", delimited)
        clauses = [c.strip() for c in raw_clauses if len(c.strip()) >= 3]
        
        # Nếu không tách được mệnh đề nào, giữ nguyên câu gốc
        return clauses if clauses else [text.strip()]

    def route_clause_to_aspects(self, clause: str) -> List[str]:
        """
        Gating Network Router: Định vị khía cạnh SERVQUAL nào được kích hoạt
        trong mệnh đề hiện tại.
        """
        clause_lower = clause.lower()
        matched = []
        for aspect_name, conf in self.aspects_config.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', clause_lower) for kw in conf["keywords"]):
                matched.append(aspect_name)
        return matched

    def analyze_aspect_sentiments(self, text: str) -> Dict[str, Any]:
        """
        Thực hiện toàn bộ quy trình MoE:
          - Bước 1: Chuẩn hóa viết tắt y tế.
          - Bước 2: Tách mệnh đề đối lập (nếu có).
          - Bước 3: Định tuyến mỗi mệnh đề tới Chuyên gia (Expert).
          - Bước 4: Chuyên gia suy luận cảm xúc riêng biệt cho từng khía cạnh.
          - Bước 5: Tổng hợp kết quả Aspect-Sentiment Mapping.
        """
        norm_text = normalize_medical_text(text)
        clauses = self.split_into_clauses(norm_text)
        
        aspect_results = {}
        has_opposing_sentiments = False
        sentiments_seen = set()

        for clause in clauses:
            activated_aspects = self.route_clause_to_aspects(clause)
            if not activated_aspects:
                continue

            # Chuyên gia đánh giá cảm xúc riêng cho mệnh đề này
            pred_info = self.classifier.predict_proba(clause)
            c_label = pred_info["ensemble_label"]
            c_conf  = pred_info["confidence"]
            sentiments_seen.add(c_label)

            for aspect_name in activated_aspects:
                aspect_results[aspect_name] = {
                    "expert_id": self.aspects_config[aspect_name]["id"],
                    "sentiment": c_label,
                    "sentiment_vi": "Tích Cực" if c_label == "Positive" else ("Tiêu Cực" if c_label == "Negative" else "Trung Lập"),
                    "confidence": c_conf,
                    "evidence_clause": clause
                }

        # Nếu không có mệnh đề nào kích hoạt từ khóa khía cạnh,
        # kiểm tra trên toàn bộ câu gốc
        if not aspect_results:
            full_aspects = self.route_clause_to_aspects(norm_text)
            overall_pred = self.classifier.predict_proba(norm_text)
            for asp in full_aspects:
                aspect_results[asp] = {
                    "expert_id": self.aspects_config[asp]["id"],
                    "sentiment": overall_pred["ensemble_label"],
                    "sentiment_vi": "Tích Cực" if overall_pred["ensemble_label"] == "Positive" else ("Tiêu Cực" if overall_pred["ensemble_label"] == "Negative" else "Trung Lập"),
                    "confidence": overall_pred["confidence"],
                    "evidence_clause": norm_text
                }

        # Kiểm tra xem câu có tình trạng "vừa khen vừa chê" hay không
        if "Positive" in sentiments_seen and "Negative" in sentiments_seen:
            has_opposing_sentiments = True

        overall_pred = self.classifier.predict_proba(norm_text)

        return {
            "original_text": text,
            "normalized_text": norm_text,
            "clauses_detected": clauses,
            "has_conflicting_sentiments": has_opposing_sentiments,
            "overall_sentiment": overall_pred["ensemble_label"],
            "overall_ensemble_sentiment": overall_pred["ensemble_label"],
            "overall_confidence": overall_pred["confidence"],
            "overall_probabilities": overall_pred["probabilities"],
            "moe_aspect_sentiments": aspect_results
        }


# ==============================================================================
# 5. TIỆN ÍCH HIỂN THỊ TRỰC QUAN & BẢNG KẾT QUẢ ĐỐI SÁNH
# ==============================================================================
def format_icon(label: str) -> str:
    if label in ["Positive", "Tích Cực"]:
        return "🟢 TÍCH CỰC (Positive)"
    elif label in ["Negative", "Tiêu Cực"]:
        return "🔴 TIÊU CỰC (Negative)"
    return "🟡 TRUNG LẬP (Neutral)"


def print_moe_result(result: Dict[str, Any]):
    """In kết quả phân tích MoE một cách trực quan, học thuật."""
    print("\n" + "=" * 80)
    print(f"📝 ĐÁNH GIÁ ĐẦU VÀO : \"{result['original_text']}\"")
    print("-" * 80)
    print(f"🔍 Mệnh đề bóc tách : {result['clauses_detected']}")
    print(f"⚡ Phát hiện vế đối lập: {'⚠️ CÓ (Vừa khen vừa chê!)' if result['has_conflicting_sentiments'] else 'Không'}")
    print(f"📊 Cảm xúc chung (Ensemble) : {format_icon(result['overall_ensemble_sentiment'])} (Độ tin cậy: {result['overall_confidence']:.2%})")
    print("\n🏛️ KẾT QUẢ BÓC TÁCH TỪNG KHÍA CẠNH TỪ HỆ THỐNG CHUYÊN GIA (MoE):")
    
    aspects = result["moe_aspect_sentiments"]
    if not aspects:
        print("  (Không tìm thấy khía cạnh chuyên biệt nào, áp dụng đánh giá chung)")
    else:
        for asp_name, info in aspects.items():
            icon = "🟢" if info["sentiment"] == "Positive" else ("🔴" if info["sentiment"] == "Negative" else "🟡")
            print(f"  • [{info['expert_id']}] {asp_name.upper():<30} ➔ {icon} {info['sentiment_vi']} ({info['confidence']:.1%})")
            print(f"      Mệnh đề minh chứng: \"{info['evidence_clause']}\"")
    print("=" * 80)


# ==============================================================================
# 6. TẬP MẪU KIỂM THỬ THỰC NGHIỆM ĐỐI KHÁNG (BENCHMARK SUITE)
# ==============================================================================
DEMO_TEST_CASES = [
    "Bác sĩ rất mát tay và êm ái nhưng phòng khám quá chật và chờ lâu.",
    "Nhân viên lễ tân thái độ lồi lõm, tuy nhiên tay nghề nha sĩ bọc răng sứ rất đẹp và không đau.",
    "Chi phí nhổ răng khôn hơi đắt nhưng bác sĩ tư vấn tận tình, cơ sở vật chất vô trùng hiện đại.",
    "Dịch vụ chất lượng tuyệt vời, bác sĩ chuyên môn cao, y tá thân thiện niềm nở.",
    "Bệnh viện sạch đẹp hiện đại song bác sĩ làm ăn quá tắc trách và đắt đỏ."
]


def run_benchmark(moe_router: AspectMoERouter):
    """Chạy toàn bộ tập kiểm thử các câu chứa cảm xúc đối lập."""
    print("\n" + "#" * 80)
    print(" CHẠY THỰC NGHIỆM ĐỐI SÁNH TRÊN TẬP CÂU CHỨA CẢM XÚC ĐỐI LẬP (MIXED SENTIMENT)")
    print("#" * 80)
    
    for idx, test_text in enumerate(DEMO_TEST_CASES, 1):
        print(f"\n[TEST CASE #{idx}]")
        res = moe_router.analyze_aspect_sentiments(test_text)
        print_moe_result(res)


# ==============================================================================
# 7. HÀM CHÍNH & GIAO DIỆN DÒNG LỆNH TƯƠNG TÁC
# ==============================================================================
def main():
    print("=" * 80)
    print(" HỆ THỐNG ENSEMBLE SOFT VOTING & MIXTURE OF EXPERTS (MoE) MIỀN Y TẾ")
    print("=" * 80)
    print(">>> Đang khởi tạo mô hình Ensemble & Gating Network Router...")
    
    t0 = time.time()
    ensemble_clf = EnsembleVotingClassifier(w1=0.30, w2=0.20, w3=0.50)
    moe_router = AspectMoERouter(ensemble_clf)
    print(f">>> Khởi tạo hoàn tất trong {time.time() - t0:.2f} giây!")
    print(f"    Trạng thái nạp: Gen1={ensemble_clf.loaded_status['gen1']} | Gen2={ensemble_clf.loaded_status['gen2']} | Gen3={ensemble_clf.loaded_status['gen3']}")

    # 1. Chạy Benchmark demo tự động
    run_benchmark(moe_router)

    # 2. Chế độ nhập tương tác trực tiếp
    print("\n" + "=" * 80)
    print(" CHẾ ĐỘ THỬ NGHIỆM TRỰC TIẾP:")
    print(" - Nhập bất kỳ câu đánh giá (đặc biệt là câu vừa khen vừa chê).")
    print(" - Gõ 'exit' hoặc 'q' để thoát.")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n👉 Nhập câu đánh giá: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát chương trình.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "q", "quit"]:
            print("Tạm biệt!")
            break

        res = moe_router.analyze_aspect_sentiments(user_input)
        print_moe_result(res)


if __name__ == "__main__":
    main()
