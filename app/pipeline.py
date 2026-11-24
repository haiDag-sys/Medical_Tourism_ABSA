# -*- coding: utf-8 -*-
"""
================================================================================
MODULE TIỀN XỬ LÝ & BỘ LỌC NGỮ CẢNH Y TẾ (PREPROCESSING PIPELINE & INTENT FILTER)
================================================================================
Bao gồm:
  1. Chuẩn hóa thuật ngữ viết tắt và tiếng lóng chuyên ngành y tế - nha khoa
  2. Tầng dịch máy nơ-ron đa ngữ (Neural Machine Translation - NMT) cho khách ngoại quốc
  3. Bộ lọc phân tích ý định (Intent Filter) & Phát hiện các khía cạnh dịch vụ SERVQUAL
================================================================================
"""

import re
from typing import Tuple, List, Dict, Any

# ------------------------------------------------------------------------------
# 1. BỘ TỪ ĐIỂN QUY CHUẨN GIẢI MÃ TỪ VIẾT TẮT Y TẾ & TEENCODE
# ------------------------------------------------------------------------------
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

def normalize_medical_text(text: str) -> Tuple[str, List[str]]:
    """Chuẩn hóa từ viết tắt y tế và trả về danh sách các từ đã thay thế."""
    normalized = text
    expanded_terms = []
    for pattern, replacement in ACRONYM_DICT.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            raw_match = re.search(pattern, normalized, flags=re.IGNORECASE).group(0)
            expanded_terms.append(f"{raw_match} ➔ {replacement}")
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized, expanded_terms

# ------------------------------------------------------------------------------
# 2. TẦNG DỊCH THUẬT TỰ ĐỘNG (CROSS-LINGUAL TRANSLATION LAYER)
# ------------------------------------------------------------------------------
COMMON_ENGLISH_WORDS = {
    "the", "is", "was", "are", "were", "and", "or", "in", "on", "at", "for", "with", "about",
    "doctor", "dentist", "clinic", "hospital", "service", "staff", "terrible", "amazing",
    "great", "bad", "good", "painless", "pain", "clean", "dirty", "price", "fee", "fees",
    "very", "too", "my", "your", "we", "they", "it", "this", "that", "dental", "treatment",
    "recommend", "recommended", "professional", "friendly", "helpful", "scam", "expensive",
    "cheap", "money", "time", "wait", "waiting", "room", "appointment", "hotel", "travel",
    "vietnam", "hanoi", "saigon", "danang", "trip", "teeth", "tooth", "implant", "crown",
    "care", "experience", "surgery", "patient", "nurses", "receptionist", "hygiene"
}

def detect_language(text: str) -> str:
    """Nhận diện nhanh văn bản có phải tiếng Anh dựa trên từ vựng phổ biến."""
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = cleaned.split()
    if not words:
        return "vi"
    eng_matches = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    ratio = eng_matches / len(words)
    has_vietnamese_diacritics = bool(re.search(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", text, re.IGNORECASE))
    if ratio >= 0.25 and not has_vietnamese_diacritics:
        return "en"
    return "vi"

def translate_to_vietnamese(text: str) -> Tuple[str, bool, str]:
    """Dịch tự động nhận xét tiếng Anh sang tiếng Việt để mô hình hiểu đúng ngữ cảnh."""
    detected_lang = detect_language(text)
    if detected_lang != "en":
        return text, False, "vi"
    
    # 1. Thử dùng deep_translator
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='vi')
        translated = translator.translate(text)
        if translated and len(translated.strip()) > 0:
            return translated.strip(), True, "en"
    except Exception:
        pass

    # 2. Từ điển dự phòng chuyên ngành
    phrase_dict = {
        r"\bterrible service\b": "dịch vụ kinh khủng",
        r"\bterrible\b": "kinh khủng tồi tệ",
        r"\bworst experience\b": "trải nghiệm tồi tệ nhất",
        r"\bworst\b": "tệ nhất",
        r"\bvery bad\b": "rất tệ",
        r"\bhorrible\b": "kinh khủng",
        r"\bamazing doctor\b": "bác sĩ tuyệt vời",
        r"\bamazing service\b": "dịch vụ tuyệt vời",
        r"\bamazing\b": "tuyệt vời",
        r"\bgreat experience\b": "trải nghiệm tuyệt vời",
        r"\bgreat\b": "tuyệt vời",
        r"\bhighly recommended\b": "rất khuyến khích trải nghiệm",
        r"\brecommended\b": "khuyên dùng",
        r"\bpainless treatment\b": "điều trị không đau",
        r"\bpainless\b": "không đau",
        r"\bvery professional\b": "rất chuyên nghiệp",
        r"\bprofessional\b": "chuyên nghiệp",
        r"\bclean and modern\b": "sạch sẽ và hiện đại",
        r"\bexpensive\b": "đắt đỏ",
        r"\bcheap\b": "giá rẻ hợp lý",
        r"\bwaste of money\b": "phí tiền lãng phí",
        r"\brude staff\b": "nhân viên thô lỗ",
        r"\bfriendly staff\b": "nhân viên thân thiện",
        r"\blong wait\b": "chờ đợi quá lâu",
        r"\bdon't go here\b": "đừng đến đây",
        r"\bdo not recommend\b": "không khuyến khích",
        r"\bscam\b": "lừa đảo"
    }
    fallback_text = text
    matched = False
    for p, r in phrase_dict.items():
        if re.search(p, fallback_text, re.IGNORECASE):
            fallback_text = re.sub(p, r, fallback_text, flags=re.IGNORECASE)
            matched = True
    return fallback_text, matched, "en"

# ------------------------------------------------------------------------------
# 3. BỘ PHÂN TÍCH Ý ĐỊNH & KHÍA CẠNH DỊCH VỤ
# ------------------------------------------------------------------------------
INTENT_REGEX_MAP = {
    "khen_ngoi": [
        r"tuyệt vời", r"rất tốt", r"hài lòng", r"tận tâm", r"nhiệt tình", r"chu đáo",
        r"mát tay", r"chuyên nghiệp", r"nhẹ nhàng", r"êm", r"sạch sẽ", r"hiện đại",
        r"uy tín", r"cảm ơn", r"ưng ý", r"không đau", r"ổn", r"tốt", r"đẹp", r"chuẩn"
    ],
    "phan_nan": [
        r"tệ", r"kém", r"thất vọng", r"kinh khủng", r"thô lỗ", r"cáu", r"bực", r"khó chịu",
        r"chờ lâu", r"chờ đợi", r"đắt", r"chặt chém", r"vẽ bệnh", r"không sạch", r"dơ",
        r"đau", r"buốt", r"chảy máu", r"nhiễm trùng", r"lừa đảo", r"chán", r"cẩu thả"
    ],
    "hoi_dap": [
        r"giá bao nhiêu", r"chi phí", r"hết bao nhiêu", r"tư vấn", r"ở đâu", r"địa chỉ",
        r"có đau không", r"mất bao lâu", r"như thế nào", r"ntn", r"không ạ", r"được không",
        r"bảo hành", r"đặt lịch", r"hẹn trước", r"liên hệ", r"inbox", r"ib"
    ],
    "canh_bao": [
        r"tránh xa", r"đừng bao giờ", r"tẩy chay", r"tiền mất tật mang", r"cảnh báo",
        r"lừa đảo", r"scam", r"phốt", r"kinh hoàng", r"tai biến", r"hỏng răng"
    ]
}

def analyze_intent_and_context(text: str) -> Tuple[Dict[str, Any], bool]:
    """Bóc tách ý định người dùng và kiểm tra xem có phải câu hỏi thông tin không."""
    lower_text = text.lower()
    intent_scores = {}
    for intent, patterns in INTENT_REGEX_MAP.items():
        score = sum(1 for p in patterns if re.search(r"\b" + p + r"\b", lower_text))
        if score > 0:
            intent_scores[intent] = score
            
    is_pure_inquiry = False
    if "hoi_dap" in intent_scores:
        has_opinion = any(k in intent_scores for k in ["khen_ngoi", "phan_nan", "canh_bao"])
        if not has_opinion:
            is_pure_inquiry = True

    intent_names = {
        "khen_ngoi": "Khen ngợi / Đánh giá tích cực",
        "phan_nan": "Phàn nàn / Đánh giá tiêu cực",
        "hoi_dap": "Hỏi đáp / Tham khảo thông tin",
        "canh_bao": "Cảnh báo cộng đồng / Tránh xa"
    }
    primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "nhan_xet_chung"
    return {
        "primary_intent": primary_intent,
        "primary_intent_label": intent_names.get(primary_intent, "Chia sẻ trải nghiệm chung"),
        "is_pure_inquiry": is_pure_inquiry
    }, is_pure_inquiry

def detect_aspects(text: str) -> List[str]:
    """Phát hiện nhanh các khía cạnh dịch vụ xuất hiện trong văn bản."""
    aspect_keywords = {
        "Bác sĩ & Chuyên môn": ["bác sĩ", "bs", "nha sĩ", "tay nghề", "chuyên môn", "mát tay", "khám"],
        "Thái độ & Nhân viên": ["nhân viên", "lễ tân", "y tá", "điều dưỡng", "thái độ", "nhiệt tình", "chu đáo"],
        "Cơ sở vật chất": ["cơ sở vật chất", "csvc", "phòng khám", "bệnh viện", "máy móc", "thiết bị", "sạch sẽ"],
        "Hiệu quả điều trị": ["kết quả", "chữa", "trám", "nhổ", "implant", "răng", "đỡ", "khỏi", "không đau"],
        "Chi phí & Giá cả": ["giá", "chi phí", "tiền", "đắt", "rẻ", "phát sinh", "bảo hiểm", "bhyt"],
        "Thời gian & Tiếp nhận": ["chờ", "đợi", "thời gian", "đặt lịch", "hẹn", "đúng giờ", "thủ tục"]
    }
    lower = text.lower()
    found = []
    for aspect, kws in aspect_keywords.items():
        if any(kw in lower for kw in kws):
            found.append(aspect)
    return found
