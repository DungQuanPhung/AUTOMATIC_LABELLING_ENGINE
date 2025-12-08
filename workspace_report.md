# Báo cáo tổng quan Workspace

## 1. Tổng quan dự án

Dự án này là một hệ thống phân tích quan điểm dựa trên khía cạnh (Aspect-Based Sentiment Analysis - ABSA) cho lĩnh vực khách sạn. Mục tiêu chính là tự động xác định các khía cạnh được đề cập trong đánh giá (ví dụ: "phòng", "dịch vụ", "vị trí") và xác định sắc thái tình cảm (tích cực, tiêu cực, trung tính) liên quan đến từng khía cạnh đó.

## 2. Cấu trúc thư mục và tệp tin

### 2.1. Tệp tin thực thi chính
| Tệp tin | Mô tả |
|---------|-------|
| `main.py`, `main_v2.py` | Các tệp khởi chạy chính của ứng dụng Streamlit |
| `api.py` | FastAPI server cung cấp REST API |
| `pipeline_ABSA.py` | Điều phối toàn bộ quy trình phân tích ABSA |
| `config.py` | Cấu hình mô hình, batch size, và các tham số hệ thống |

### 2.2. Modules xử lý
| Module | Chức năng |
|--------|-----------|
| `split_clause_lib.py` | Tách câu thành các mệnh đề và trích xuất term |
| `Extract_Opinion.py` | Trích xuất từ ngữ biểu đạt quan điểm |
| `Extract_Opinion_by_Batch.py` | Trích xuất opinion theo batch |
| `Extract_Category.py` | Phân loại khía cạnh/danh mục |
| `Extract_Polarity_lib.py`, `Extract_Polarity_v2.py` | Xác định sắc thái tình cảm |
| `Fine_Tune_RoBertaBase.py` | Script tinh chỉnh mô hình RoBERTa |
| `download_models.py` | Tải về các mô hình cần thiết |

### 2.3. Thư mục mô hình
| Thư mục | Nội dung |
|---------|----------|
| `category_model/` | Mô hình RoBERTa để nhận diện khía cạnh |
| `polarity_model/` | Mô hình phân loại sắc thái |

### 2.4. Mô hình LLM (GGUF)
- `qwen2.5-7b-instruct-q2_k.gguf` - Mô hình Qwen chính (quantized Q2_K)
- `qwen2.5-7b-instruct-q3_k_m.gguf` - Phiên bản Q3_K_M
- `qwen.gguf` - Phiên bản khác

### 2.5. Dữ liệu
- `70% sample.csv` - Dữ liệu mẫu huấn luyện/kiểm thử
- `sample_qa.csv` - Dữ liệu Q&A mẫu
- `goal.xlsx` - Dữ liệu mục tiêu/nhãn

### 2.6. Giao diện
- `index.html`, `index_v2.html` - Giao diện web demo

### 2.7. Cấu hình & Triển khai
- `requirements.txt` - Các thư viện Python cần thiết
- `Dockerfile` - Cấu hình Docker container
- `.gitignore` - Danh sách file bỏ qua khi commit

## 3. Công nghệ sử dụng

### 3.1. Ngôn ngữ & Framework
| Công nghệ | Phiên bản/Mô tả |
|-----------|-----------------|
| Python | Ngôn ngữ chính |
| FastAPI | REST API framework |
| Streamlit | Giao diện web tương tác |
| Uvicorn | ASGI server |

### 3.2. Thư viện học máy
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| transformers | >=4.36.0 | Hugging Face Transformers (RoBERTa, DeBERTa) |
| torch | >=2.0.0 | PyTorch backend |
| llama-cpp-python | - | Chạy mô hình GGUF (Qwen) với GPU |
| accelerate | >=0.25.0 | Tối ưu hóa inference |
| optimum | >=1.16.0 | Tối ưu hóa mô hình Hugging Face |
| peft | - | Parameter-Efficient Fine-Tuning |
| datasets | - | Xử lý dataset cho huấn luyện |
| scikit-learn | - | Các tiện ích ML |
| sentencepiece | - | Tokenization |

### 3.3. Xử lý dữ liệu & Visualization
- `pandas` - Xử lý dữ liệu bảng
- `matplotlib` - Biểu đồ
- `wordcloud` - Tạo word cloud

## 4. Kiến trúc hệ thống

### 4.1. Mô hình AI sử dụng
| Mô hình | Loại | Chức năng |
|---------|------|-----------|
| Qwen 2.5-7B | LLM (GGUF) | Tách clause, trích xuất term và opinion |
| RoBERTa | Transformer | Phân loại khía cạnh (Category) |
| DeBERTa | Transformer | Phân loại sắc thái (Polarity) |

### 4.2. Pipeline xử lý ABSA
```
Input Review
    │
    ▼
┌─────────────────────────────────┐
│ Bước 1: Tách Clause & Term      │ ← Qwen LLM
│ (split_clause_lib.py)           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Bước 2: Trích xuất Opinion      │ ← Qwen LLM
│ (Extract_Opinion_by_Batch.py)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Bước 3: Phân loại Category      │ ← RoBERTa
│ (Extract_Category.py)           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Bước 4: Phân loại Polarity      │ ← DeBERTa
│ (Extract_Polarity_v2.py)        │
└─────────────────────────────────┘
    │
    ▼
Output: DataFrame với các cột
- Clause, Term, Opinion
- Category, Category Score
- Polarity, Polarity Score
```

## 5. API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Health check |
| POST | `/analyze` | Phân tích một đoạn văn bản |
| POST | `/upload` | Upload file CSV/Excel/TXT để phân tích hàng loạt |

### Ví dụ request `/analyze`:
```json
{
  "text": "Phòng rất sạch sẽ nhưng dịch vụ hơi chậm"
}
```

### Response:
```json
[
  {
    "Clause": "Phòng rất sạch sẽ",
    "Term": "Phòng",
    "Opinion": "sạch sẽ",
    "Category": "ROOM",
    "Category Score": 0.95,
    "Polarity": "positive",
    "Polarity Score": 0.92
  },
  ...
]
```

## 6. Cấu hình hệ thống (config.py)

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| CATEGORY_BATCH_SIZE | 32 | Batch size cho phân loại category |
| POLARITY_BATCH_SIZE | 64 | Batch size cho phân loại polarity |
| OPINION_BATCH_SIZE | 32 | Batch size cho trích xuất opinion |
| QWEN_N_GPU_LAYERS | -1 | Sử dụng toàn bộ GPU layers |
| QWEN_N_CTX | 2048 | Context length cho Qwen |
| QWEN_N_THREADS | 8 | Số CPU threads |
| TORCH_DTYPE | bfloat16/float16 | Tự động chọn precision tối ưu |

## 7. Hướng dẫn cài đặt và chạy

### 7.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 7.2. Tải mô hình (nếu chưa có)
```bash
python download_models.py
```

### 7.3. Chạy API Server
```bash
python api.py
```
Server sẽ chạy tại `http://localhost:8000`

### 7.4. Chạy giao diện Streamlit
```bash
streamlit run main.py
```

### 7.5. Triển khai với Docker
```bash
docker build -t hotel-absa .
docker run -p 8000:8000 --gpus all hotel-absa
```

## 8. Yêu cầu phần cứng

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|-------------------|-------------|
| GPU | NVIDIA với CUDA | RTX 3060+ (12GB VRAM) |
| RAM | 16GB | 32GB |
| Storage | 20GB | SSD 50GB+ |

## 9. Ghi chú

- Hệ thống hỗ trợ cả CPU và GPU, tự động phát hiện và sử dụng GPU nếu có
- Mô hình Qwen sử dụng định dạng GGUF để tối ưu bộ nhớ và tốc độ
- API hỗ trợ xử lý tối đa 200 dòng/file khi upload
- Debug mode có thể bật/tắt qua `DEBUG_MODE` trong config.py
