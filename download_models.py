"""
Script để download các models từ Hugging Face về local.
Chạy file này trước khi deploy để đảm bảo models được cache sẵn.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Thư mục lưu models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Danh sách models cần download
MODELS = {
    "visobert_absa": {
        "name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "local_path": os.path.join(BASE_DIR, "polarity_model"),
    }
}

def download_model(model_name: str, local_path: str):
    """Download một model từ Hugging Face về local."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {model_name}")
    print(f"📁 Save to: {local_path}")
    print('='*60)
    
    try:
        # Download tokenizer
        print("   ⏳ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_path)
        print("   ✅ Tokenizer saved!")
        
        # Download model
        print("   ⏳ Downloading model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(local_path)
        print("   ✅ Model saved!")
        
        # Show files
        print(f"\n   📁 Files created:")
        total_size = 0
        for f in os.listdir(local_path):
            fpath = os.path.join(local_path, f)
            fsize = os.path.getsize(fpath) / (1024 * 1024)
            total_size += fsize
            print(f"      - {f}: {fsize:.2f} MB")
        print(f"      📦 Total: {total_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_visobert():
    """Download ViSoBERT ABSA Hotel model."""
    model_info = MODELS["visobert_absa"]
    return download_model(model_info["name"], model_info["local_path"])


def download_deberta():
    """Download DeBERTa Polarity model."""
    model_info = MODELS["deberta_polarity"]
    return download_model(model_info["name"], model_info["local_path"])


def download_all():
    """Download tất cả models."""
    print("🚀 Starting model downloads...")
    print(f"   Base directory: {BASE_DIR}")
    
    results = {}
    for key, model_info in MODELS.items():
        success = download_model(model_info["name"], model_info["local_path"])
        results[key] = success
    
    # Summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    for key, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {MODELS[key]['name']}: {status}")
    
    print("\n💡 Cập nhật config.py để sử dụng models local:")
    print("""
    # ViSoBERT ABSA Hotel
    VISOBERT_MODEL_PATH = os.path.join(BASE_DIR, "visobert-absa-hotel")
    
    # DeBERTa Polarity
    POLARITY_MODEL_PATH = os.path.join(BASE_DIR, "polarity_model")
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "visobert":
            download_visobert()
        elif arg == "deberta":
            download_deberta()
        elif arg == "all":
            download_all()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python download_models.py [visobert|deberta|all]")
    else:
        # Default: download all
        download_all()
