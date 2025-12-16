# Sử dụng python 3.10.19 slim
FROM python:3.10.19-slim

# --- KHẮC PHỤC LỖI Ở ĐÂY ---
# Cài đặt các công cụ biên dịch C++ cần thiết (gcc, g++, cmake, make)
# build-essential: chứa gcc, g++, make
# cmake: cần thiết cho quy trình build của llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt thư viện Python
# Thêm biến môi trường để ép buộc build lại nếu cần thiết
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào sau cùng
COPY . .

# Thiết lập biến môi trường cho Cloud Run
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Chạy ứng dụng
CMD ["sh", "-c", "streamlit run main_v2.py --server.address=0.0.0.0 --server.port=$PORT"]