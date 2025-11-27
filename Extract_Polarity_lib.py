from typing import Dict, List, Any
from config import POLARITY_MODEL_PATH
from transformers import pipeline
from tqdm import tqdm

# --- Hàm phát hiện polarity ---
def detect_polarity(clauses, polarity_classifier):
    results = []
    for item in clauses:
        clause = str(item.get("clause", "")).strip()
        if clause == "":
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0
            results.append(item)
            continue

        try:
            res = polarity_classifier(clause)
            # Một số model Hugging Face trả về list lồng list -> xử lý để lấy nhãn cao nhất
            if isinstance(res, list) and isinstance(res[0], list):
                res = res[0]
            top = max(res, key=lambda x: x["score"])
            item["polarity"] = top["label"].capitalize()
            item["polarity_score"] = round(top["score"], 4)
        except Exception as e:
            print(f" Lỗi khi xử lý câu '{clause}': {e}")
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0

        results.append(item)
    return results