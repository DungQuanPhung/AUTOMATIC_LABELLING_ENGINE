import os
import torch

# ============== CẤU HÌNH MÔ HÌNH ==============

# Thư mục gốc của app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mô hình LLM chính (Qwen) - Load từ local
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "./qwen2.5-coder-7b-instruct-q4_0.gguf")

# Mô hình Category (RoBERTa)
CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "roberta_lora_goal")

# Mô hình Polarity (DeBERTa)
POLARITY_MODEL_PATH = "yangheng/deberta-v3-base-absa-v1.1"

# ============== CẤU HÌNH BATCH SIZE ==============

# Batch size cho các mô hình phân loại
CATEGORY_BATCH_SIZE = 64
POLARITY_BATCH_SIZE = 64

# ============== CẤU HÌNH TOKEN ==============

# Số token mới sinh ra tối đa cho Opinion extraction
MAX_NEW_TOKENS_OPINION = 64
MAX_NEW_TOKENS_SPLIT = 256
MAX_INPUT_LENGTH = 2048

# ============== CẤU HÌNH KHÁC ==============
DEBUG_MODE = False

# ============== CẤU HÌNH DEVICE ==============

# Device auto-detect
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"