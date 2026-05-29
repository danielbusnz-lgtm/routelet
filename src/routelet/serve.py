"""Interactive playground: type a command, see how routelet classifies it.

Loads the trained SetFit model once at startup and serves a one-box web page at
`/` plus a `POST /route` endpoint that returns the predicted intent, its
confidence, and the full class distribution (including the `none` reject class).

This is a dev/demo tool, not the production path. Aegis runs the exported ONNX
model in-process; this loads the torch model for convenience. Inference mirrors
production: preprocess, embed, LR head, temperature-scale, softmax.

Run:
    uv pip install fastapi uvicorn   # or: uv sync --extra serve --extra train
    uv run uvicorn routelet.serve:app
then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from routelet.preprocess import preprocess

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "setfit"
PAGE = (Path(__file__).parent / "index.html").read_text()


def _load_model() -> dict:
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    head = model.model_head
    temp_path = MODEL_DIR / "temperature.json"
    temperature = json.loads(temp_path.read_text())["temperature"] if temp_path.exists() else 1.0
    return {
        "body": model.model_body,
        "coef": head.coef_,
        "intercept": head.intercept_,
        "labels": head.classes_.tolist(),
        "temperature": temperature,
    }


_MODEL = _load_model()


def classify(text: str) -> list[dict]:
    """Return the class distribution for `text`, sorted most-likely first.
    Each entry is {label, prob}; entry 0 is the prediction."""
    emb = _MODEL["body"].encode(
        [preprocess(text)], convert_to_numpy=True, show_progress_bar=False
    )
    logits = (emb @ _MODEL["coef"].T + _MODEL["intercept"]) / _MODEL["temperature"]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)[0]
    probs /= probs.sum()
    order = np.argsort(probs)[::-1]
    return [{"label": _MODEL["labels"][i], "prob": round(float(probs[i]), 4)} for i in order]


app = FastAPI(title="routelet playground")


class RouteRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.post("/route")
def route(req: RouteRequest) -> dict:
    dist = classify(req.text)
    return {"intent": dist[0]["label"], "confidence": dist[0]["prob"], "distribution": dist}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
