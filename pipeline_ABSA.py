import os
import pandas as pd
import torch
from typing import Dict, Any
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
from llama_cpp import Llama
from config import (
    QWEN_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    POLARITY_MODEL_PATH,
    CATEGORY_BATCH_SIZE,
    POLARITY_BATCH_SIZE
)

from split_clause_lib import split_and_term_extraction
from Extract_Opinion import extract_opinions_only_from_clauses
from Extract_Category import get_predicted_categories
from Extract_Polarity_lib import detect_polarity

# ---------------- QWEN LOADER (TỐI ƯU) ---------------- #
@st.cache_resource
def load_qwen_model():
    llama = Llama(
        model_path=QWEN_MODEL_PATH,
        n_gpu_layers=0,
        n_ctx=2048,
        verbose=True
    )   
    return llama, None  # Trả về None cho tokenizer vì không cần thiết với llama

# ---------------- CATEGORY MODEL (RoBERTa) ---------------- #
DEFAULT_CATEGORY_LABELS = {
    0: "Amenity",
    1: "Branding",
    2: "Experience",
    3: "Facility",
    4: "Loyalty",
    5: "Service",
}

def _normalize_id2label(raw_id2label):
    """Chuẩn hóa nhãn để tránh hiển thị LABEL_x."""
    normalized = {}
    for key, label in raw_id2label.items():
        if isinstance(label, str) and label.upper().startswith("LABEL_"):
            try:
                idx = int(label.split("_")[-1])
                normalized[key] = DEFAULT_CATEGORY_LABELS.get(idx, label)
            except ValueError:
                normalized[key] = label
        else:
            normalized[key] = label
    return normalized

@st.cache_resource
def load_category_model():
    """Tải RoBERTa category model (SequenceClassification)."""
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    torch_dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        CATEGORY_MODEL_PATH,
        torch_dtype=torch_dtype,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(CATEGORY_MODEL_PATH)

    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = _normalize_id2label(model.config.id2label)
    else:
        id2label = DEFAULT_CATEGORY_LABELS

    return model, tokenizer, id2label

# ---------------- POLARITY MODEL (DeBERTa) ---------------- #
@st.cache_resource
def load_polarity_model():
    """
    Tải mô hình polarity dưới dạng HuggingFace pipeline.
    Có thể tối ưu thêm bằng device / batch_size nếu cần.
    """
    device = 0 if torch.cuda.is_available() else -1

    polarity_classifier = pipeline(
        "text-classification",
        model=POLARITY_MODEL_PATH,
        top_k=None,        # lấy tất cả logits, sau đó chọn nhãn cao nhất
        truncation=True,
        device=device,
    )

    return polarity_classifier

# ---------------- MASTER LOADER ---------------- #
def load_all_models():
    """Load tất cả models và cache."""
    qwen_model, qwen_tokenizer = load_qwen_model()
    cat_model, cat_tokenizer, cat_id2label = load_category_model()
    polarity_classifier = load_polarity_model()

    return {
        "qwen_model": qwen_model,
        "qwen_tokenizer": qwen_tokenizer,  # Không cần với llama_cpp
        "cat_model": cat_model,
        "cat_tokenizer": cat_tokenizer,
        "cat_id2label": cat_id2label,
        "polarity_classifier": polarity_classifier,
    }


# ---------------- FULL PIPELINE ---------------- #
def run_full_pipeline(sentence: str, models: Dict[str, Any]) -> pd.DataFrame:
    """Chạy toàn bộ pipeline ABSA trên 1 câu review."""
    sentence = (sentence or "").strip()
    if not sentence:
        return pd.DataFrame([])

    qwen_model = models["qwen_model"]
    qwen_tokenizer = models["qwen_tokenizer"]
    cat_model = models["cat_model"]
    cat_tokenizer = models["cat_tokenizer"]
    cat_id2label = models["cat_id2label"]
    polarity_classifier = models["polarity_classifier"]

    # Bước 1 & 2: Tách Clause, Term và Opinion (Qwen)
    clauses_with_details = split_and_term_extraction(sentence, qwen_model, qwen_tokenizer)
    clauses_with_details = extract_opinions_only_from_clauses(
        clauses_with_details, qwen_model, qwen_tokenizer
    )

    # Bước 3: Category (RoBERTa)
    clauses_categories = get_predicted_categories(
        clauses_with_details,
        cat_model,
        cat_tokenizer,
        cat_id2label,
        batch_size=CATEGORY_BATCH_SIZE,
    )

    # Bước 4: Polarity (DeBERTa)
    final_results = detect_polarity(
        clauses_categories,
        polarity_classifier,
    )

    df = pd.DataFrame(final_results)
    columns_order = [
        "clause",
        "term",
        "opinion",
        "category",
        "category_score",
        "polarity",
        "polarity_score",
        "sentence_original",
    ]
    final_columns = [c for c in columns_order if c in df.columns]

    return df[final_columns] if final_columns else df
