import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
import pipeline_ABSA as absa_pipeline


app = FastAPI(title="ABSA API")


class RequestBody(BaseModel):
    text: str


def _resolve_qwen_model_path() -> str:
    """Find a usable local Qwen model file, mirroring main.py logic."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        getattr(config, "QWEN_MODEL_PATH", ""),
        os.path.join(base_dir, "qwen2.5-coder-7b-instruct-q4_0.gguf"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def _ensure_model_paths():
    """Sync Qwen path into both config and pipeline before loading models."""
    resolved = _resolve_qwen_model_path()
    if not resolved:
        raise FileNotFoundError(
            "Không tìm thấy file GGUF cho Qwen. Cập nhật config.QWEN_MODEL_PATH hoặc thêm file vào thư mục app."
        )
    config.QWEN_MODEL_PATH = resolved
    absa_pipeline.QWEN_MODEL_PATH = resolved
    return resolved


@lru_cache(maxsize=1)
def get_models():
    """Load models once for the API process."""
    _ensure_model_paths()
    loader = getattr(absa_pipeline.load_all_models, "__wrapped__", absa_pipeline.load_all_models)
    return loader()


@app.get("/health")
def health_check():
    try:
        path = _ensure_model_paths()
        return {"status": "ok", "qwen_path": path}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"status": "error", "detail": str(exc)}


@app.post("/analyze")
def analyze(body: RequestBody):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` không được để trống.")

    try:
        models = get_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi load model: {exc}")

    try:
        df = absa_pipeline.run_full_pipeline(text, models)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline lỗi: {exc}")

    return df.to_dict(orient="records")
