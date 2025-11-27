import torch
from typing import List, Dict, Any

MAX_LENGTH = 128

def get_predicted_categories(
    clauses: List[Dict[str, Any]],
    model,
    tokenizer,
    id2label: Dict[int, str],
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Dự đoán category cho từng clause bằng mô hình phân loại.

    Args:
        clauses: Danh sách clause với khóa "clause".
        model: Mô hình phân loại đã load sẵn.
        tokenizer: Tokenizer tương ứng với model.
        id2label: Mapping id -> nhãn category.
        batch_size: Số lượng câu xử lý mỗi batch.

    Returns:
        List[Dict[str, Any]]: Clauses đã được gán thêm khóa 'category' và 'category_score'.
    """
    if not clauses:
        return clauses

    device = next(model.parameters()).device
    model.eval()

    texts = []
    clause_indices = []

    for idx, clause in enumerate(clauses):
        text = str(clause.get("clause", "")).strip()
        if not text:
            clause["category"] = "Unknown"
            clause["category_score"] = 0.0
            continue
        texts.append(text)
        clause_indices.append(idx)

    if not texts:
        return clauses

    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_idx = clause_indices[start : start + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                ).to(device)

                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                scores, preds = torch.max(probs, dim=1)

                for idx_local, pred, score in zip(batch_idx, preds.tolist(), scores.tolist()):
                    clauses[idx_local]["category"] = id2label.get(pred, "Unknown")
                    clauses[idx_local]["category_score"] = float(score)
    except Exception as exc:
        # Nếu có lỗi, log và gán Unknown để không dừng pipeline
        print(f"[Category] Lỗi khi suy luận: {exc}")
        for idx in clause_indices:
            clauses[idx].setdefault("category", "Unknown")
            clauses[idx].setdefault("category_score", 0.0)

    return clauses