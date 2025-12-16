import pandas as pd
from typing import Dict, Any
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
import torch
from llama_cpp import Llama
from config import (
    MAX_NEW_TOKENS_SPLIT,
    MAX_NEW_TOKENS_OPINION,
    QWEN_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    POLARITY_MODEL_PATH,
    OPINION_BATCH_SIZE,
    CATEGORY_BATCH_SIZE,
    POLARITY_BATCH_SIZE,
    QWEN_N_CTX,
    QWEN_N_GPU_LAYERS,
    QWEN_N_THREADS,
    DEBUG_MODE,
    PREFETCH_FACTOR,
    DEVICE,
    USE_FAST_TOKENIZER
)

from split_clause_lib import split_and_term_extraction
from Extract_Opinion_by_Batch import extract_opinions_only_from_clauses
from Extract_Category import get_predicted_categories, DEFAULT_CATEGORY_LABELS
from Extract_Polarity_v2 import detect_polarity

# ---------------- QWEN LOADER (TỐI ƯU) ---------------- #
@st.cache_resource
def load_qwen_model():
    # Load Qwen model GGUF với GPU acceleration.
    return Llama(
        model_path=QWEN_MODEL_PATH,
        n_gpu_layers=QWEN_N_GPU_LAYERS,
        n_ctx=QWEN_N_CTX,
        n_threads=QWEN_N_THREADS,
        prefetch_factor=PREFETCH_FACTOR,
        verbose=DEBUG_MODE
    ), None  # Trả về None cho tokenizer vì không cần thiết

@st.cache_resource
def load_category_model():
    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(CATEGORY_MODEL_PATH, use_fast=USE_FAST_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        CATEGORY_MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    return model, tokenizer, DEFAULT_CATEGORY_LABELS

# ---------------- POLARITY MODEL (DeBERTa) ---------------- #
@st.cache_resource
def load_polarity_model():
    # Load polarity model using the Slow tokenizer to avoid byte_fallback warnings.
    # 1. Load tokenizer explicitly with use_fast=False
    tokenizer = AutoTokenizer.from_pretrained(POLARITY_MODEL_PATH, use_fast=USE_FAST_TOKENIZER)
    # 2. Load model
    model = AutoModelForSequenceClassification.from_pretrained(POLARITY_MODEL_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    # 3. Pass both to pipeline
    polarity_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,  # Explicitly pass the slow tokenizer
        top_k=None,
        truncation=True,
        device=DEVICE
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
        "qwen_tokenizer": qwen_tokenizer,
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
    clauses_with_details = split_and_term_extraction(
        sentence,
        qwen_model,
        qwen_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS_SPLIT
    )
    clauses_with_details = extract_opinions_only_from_clauses(
        clauses_with_details,
        qwen_model,
        qwen_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS_OPINION,
        batch_size=OPINION_BATCH_SIZE
    )

    # Bước 3: Category (RoBERTa)
    clauses_categories = get_predicted_categories(
        clauses_with_details,
        cat_model,
        cat_tokenizer,
        cat_id2label,
        batch_size=CATEGORY_BATCH_SIZE
    )

    # Bước 4: Polarity (DeBERTa)
    final_results = detect_polarity(
        clauses_categories,
        polarity_classifier,
        batch_size=POLARITY_BATCH_SIZE
    )

    df = pd.DataFrame(final_results)
    
    # Map từ keys trong dict sang column names cho hiển thị
    column_mapping = {
        "clause": "Clause",
        "term": "Term", 
        "opinion": "Opinion",
        "category": "Category",
        "category_score": "Category Score",
        "polarity": "Polarity",
        "polarity_score": "Polarity Score",
        "sentence_original": "Original Sentence"
    }
    
    # Rename columns theo mapping
    df = df.rename(columns=column_mapping)
    
    # Đảm bảo thứ tự cột: Original Sentence ở cuối
    columns_order = [
        "Clause",
        "Term",
        "Opinion",
        "Category",
        "Category Score",
        "Polarity",
        "Polarity Score",
        "Original Sentence",
    ]
    final_columns = [c for c in columns_order if c in df.columns]
    return df[final_columns] if final_columns else df