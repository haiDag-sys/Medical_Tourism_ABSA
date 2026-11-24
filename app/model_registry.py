# -*- coding: utf-8 -*-
"""
================================================================================
MODULE NẠP VÀ QUẢN LÝ MÔ HÌNH AI (MODEL REGISTRY - SINGLETON PATTERN)
================================================================================
Quản lý nạp trước 3 thế hệ mô hình NLP vào bộ nhớ RAM khi khởi động:
  - Thế hệ 1: TF-IDF N-grams (1-2) + Linear SVM (Calibrated Platt Scaling)
  - Thế hệ 2: Word2Vec Skip-Gram (200D) + TF-IDF Pooling + Random Forest
  - Thế hệ 3: PhoBERT-base-v2 (Deep Contextual 768D) + MLP Classifier
  - Ensemble: Heterogeneous Soft Voting (w1=0.30, w2=0.20, w3=0.50)
  - MoE: Aspect-Guided Mixture of Experts (Router + 6 Aspect Experts)
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = BASE_DIR / "models"

import numpy as np
import joblib
import torch
from underthesea import word_tokenize

# Import MoE Router
try:
    from app.moe_router import AspectMoERouter, EnsembleVotingClassifier
except ImportError:
    from moe_router import AspectMoERouter, EnsembleVotingClassifier

class ModelRegistry:
    """Singleton quản lý toàn bộ các mô hình học máy và học sâu."""
    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self.models_loaded = False
        self.gen1_tfidf = None
        self.gen1_model = None
        self.gen2_w2v = None
        self.gen2_idf = None
        self.gen2_model = None
        self.gen3_tokenizer = None
        self.gen3_model = None
        self.moe_router = None
        self.classes = np.array(["Tiêu cực", "Trung lập", "Tích cực"])

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def load_all_models(self) -> bool:
        """Nạp toàn bộ mô hình từ đĩa vào RAM."""
        if self.models_loaded:
            return True

        print(">>> [ModelRegistry] Đang nạp các mô hình AI từ đĩa vào RAM...")
        try:
            # 1. Gen 1: Linear SVM
            gen1_tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
            gen1_model_path = MODELS_DIR / "best_sentiment_model.pkl"
            if gen1_tfidf_path.exists() and gen1_model_path.exists():
                self.gen1_tfidf = joblib.load(gen1_tfidf_path)
                self.gen1_model = joblib.load(gen1_model_path)
                print(" -> [Gen 1] TF-IDF + Linear SVM: SẴN SÀNG")

            # 2. Gen 2: Word2Vec + Random Forest
            gen2_w2v_path = MODELS_DIR / "word2vec_medical.model"
            gen2_idf_path = MODELS_DIR / "gen2_tfidf_weights.pkl"
            gen2_model_path = MODELS_DIR / "gen2_word2vec_sentiment_model.pkl"
            if gen2_w2v_path.exists() and gen2_model_path.exists():
                from gensim.models import Word2Vec
                self.gen2_w2v = Word2Vec.load(str(gen2_w2v_path))
                self.gen2_idf = joblib.load(gen2_idf_path) if gen2_idf_path.exists() else None
                self.gen2_model = joblib.load(gen2_model_path)
                print(" -> [Gen 2] Word2Vec + Random Forest: SẴN SÀNG")

            # 3. Gen 3: PhoBERT + MLP
            gen3_model_path = MODELS_DIR / "gen3_phobert_classifier.pkl"
            if gen3_model_path.exists():
                from transformers import AutoTokenizer, AutoModel
                try:
                    self.gen3_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False, local_files_only=True)
                    self.gen3_backbone = AutoModel.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
                except Exception:
                    self.gen3_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
                    self.gen3_backbone = AutoModel.from_pretrained("vinai/phobert-base-v2")
                self.gen3_backbone.eval()
                self.gen3_model = joblib.load(gen3_model_path)
                print(" -> [Gen 3] PhoBERT-base-v2 + MLP SOTA: SẴN SÀNG")

            # 4. MoE Router
            ensemble_clf = EnsembleVotingClassifier()
            self.moe_router = AspectMoERouter(ensemble_classifier=ensemble_clf)
            print(" -> [MoE Router] Bộ định tuyến 6 Aspect Experts: SẴN SÀNG")

            self.models_loaded = True
            print(">>> [ModelRegistry] Toàn bộ hệ thống mô hình đã nạp thành công vào RAM!")
            return True
        except Exception as e:
            print(f"[ModelRegistry] Cảnh báo khi nạp mô hình: {e}")
            return False

    def predict_gen1(self, text_segmented: str) -> np.ndarray:
        """Dự đoán xác suất theo Gen 1 (TF-IDF + Linear SVM)."""
        if self.gen1_tfidf is None or self.gen1_model is None:
            return np.array([0.33, 0.33, 0.34])
        vec = self.gen1_tfidf.transform([text_segmented])
        if hasattr(self.gen1_model, "predict_proba"):
            return self.gen1_model.predict_proba(vec)[0]
        dec = self.gen1_model.decision_function(vec)[0]
        exp_dec = np.exp(dec - np.max(dec))
        return exp_dec / np.sum(exp_dec)

    def predict_gen2(self, text_segmented: str) -> np.ndarray:
        """Dự đoán xác suất theo Gen 2 (Word2Vec + Random Forest)."""
        if self.gen2_w2v is None or self.gen2_model is None:
            return np.array([0.33, 0.33, 0.34])
        words = text_segmented.split()
        vectors = []
        weights = []
        for w in words:
            if w in self.gen2_w2v.wv:
                vectors.append(self.gen2_w2v.wv[w])
                idf_val = self.gen2_idf.get(w, 1.0) if self.gen2_idf else 1.0
                weights.append(idf_val)
        if len(vectors) == 0:
            doc_vec = np.zeros(self.gen2_w2v.vector_size)
        else:
            doc_vec = np.average(vectors, axis=0, weights=weights)
        return self.gen2_model.predict_proba([doc_vec])[0]

    def predict_gen3(self, raw_text: str) -> np.ndarray:
        """Dự đoán xác suất theo Gen 3 (PhoBERT-base-v2 [CLS] + MLP)."""
        if self.gen3_tokenizer is None or self.gen3_model is None:
            return np.array([0.33, 0.33, 0.34])
        inputs = self.gen3_tokenizer(raw_text, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = self.gen3_backbone(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        return self.gen3_model.predict_proba(cls_emb)[0]

    def predict_ensemble(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Kết hợp dự đoán theo cơ chế Soft Voting có trọng số (0.30, 0.20, 0.50)."""
        p_ens = 0.30 * p1 + 0.20 * p2 + 0.50 * p3
        pred_idx = np.argmax(p_ens)
        return self.classes[pred_idx], float(p_ens[pred_idx]), p_ens

def get_model_registry() -> ModelRegistry:
    return ModelRegistry.get_instance()
