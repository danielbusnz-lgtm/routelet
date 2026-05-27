"""Export the trained SetFit encoder to ONNX and dump the LR head to JSON.

SetFit 1.1.3's bundled export_onnx fuses the sklearn head via skl2onnx, which
emits opset 13 while torch exports the encoder at opset 18. onnx.merge_models
refuses to combine them, so the fused path is broken on this stack.

Instead we split the model:
  embedder.onnx        fp32 BERT encoder with CLS pooling and L2 normalization
  embedder.int8.onnx   dynamic int8 quantization of the above
  head.json            LR coef/intercept/labels for the Rust consumer to apply
  tokenizer.json       copied from models/setfit/ for the Rust tokenizer

The Rust consumer runs: embedding = onnx(tokens); logits = coef @ emb + intercept
then argmax -> labels[i].
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import transformers
from onnxruntime.quantization import QuantType, quantize_dynamic
from setfit import SetFitModel

from routelet.data import load
from routelet.preprocess import preprocess

SETFIT_DIR = "models/setfit"
EMBEDDER_OUT = "models/embedder.onnx"
EMBEDDER_INT8_OUT = "models/embedder.int8.onnx"
HEAD_OUT = "models/head.json"
TOKENIZER_OUT = "models/tokenizer.json"
EVAL_FILE = "evals/holdout.jsonl"
TEMPERATURE_FILE = "models/setfit/temperature.json"
OPSET = 14


class EncoderWithPooling(nn.Module):
    """BERT encoder + CLS pooling + L2 normalization in one exportable module.

    Derives the pooling mode from the SentenceTransformer's Pooling module at
    construction time rather than hard-coding it. For bge-small-en-v1.5 the
    trained model uses CLS pooling followed by L2 normalization (modules 1 and
    2 in the ST pipeline), so the wrapper takes last_hidden_state[:, 0, :] and
    normalizes it. If the Pooling module reports a different mode a ValueError
    is raised at export time rather than silently producing wrong embeddings.
    """

    def __init__(self, sentence_transformer: object) -> None:
        super().__init__()
        pooling_mod = sentence_transformer[1]
        mode = pooling_mod.pooling_mode
        if mode != "cls":
            raise ValueError(
                f"EncoderWithPooling only implements CLS pooling; "
                f"found pooling_mode={mode!r}. Update the wrapper if the "
                "model uses a different pooling strategy."
            )
        self.bert = sentence_transformer[0].auto_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # CLS token is at position 0 for right-padded BERT inputs.
        cls = out.last_hidden_state[:, 0, :]
        # L2 normalize so the embedding lives on the unit sphere, matching
        # the Normalize module in the SentenceTransformer pipeline.
        normalized = cls / cls.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return normalized


def _export_embedder(wrapper: EncoderWithPooling, out_path: str) -> None:
    dummy_ids = torch.zeros(1, 16, dtype=torch.int64)
    dummy_mask = torch.ones(1, 16, dtype=torch.int64)
    dummy_types = torch.zeros(1, 16, dtype=torch.int64)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "embedding": {0: "batch"},
    }

    Path(out_path).parent.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask, dummy_types),
            out_path,
            dynamo=False,
            opset_version=OPSET,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    print(f"[export] legacy TorchScript exporter, opset {OPSET}")
    print(f"[export] saved {out_path}")


def _print_session_info(sess: ort.InferenceSession) -> None:
    print("\nembedder.onnx I/O signature:")
    for inp in sess.get_inputs():
        print(f"  input  {inp.name!r:20s}  dtype={inp.type!s:20s}  shape={inp.shape}")
    for out in sess.get_outputs():
        print(f"  output {out.name!r:20s}  dtype={out.type!s:20s}  shape={out.shape}")


def _run_onnx_preds(
    sess: ort.InferenceSession,
    tokenizer: transformers.PreTrainedTokenizerBase,
    texts: list[str],
    coef: np.ndarray,
    intercept: np.ndarray,
    labels: list[str],
) -> list[str]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )
    feed = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
        "token_type_ids": encoded["token_type_ids"].astype(np.int64),
    }
    (emb,) = sess.run(["embedding"], feed)
    logits = emb @ coef.T + intercept
    return [labels[int(np.argmax(row))] for row in logits]


def main() -> None:
    model = SetFitModel.from_pretrained(SETFIT_DIR)
    body = model.model_body.to("cpu")
    head = model.model_head

    # Inspect and confirm the ST pooling/normalize pipeline.
    pooling_mode = body[1].pooling_mode
    print(f"ST pooling mode: {pooling_mode}")
    print(f"ST modules: {[type(m).__name__ for m in body]}")

    # Build the exportable wrapper.
    wrapper = EncoderWithPooling(body)
    wrapper.eval()

    # Export fp32 ONNX.
    _export_embedder(wrapper, EMBEDDER_OUT)

    # Quantize to int8.
    quantize_dynamic(EMBEDDER_OUT, EMBEDDER_INT8_OUT, weight_type=QuantType.QInt8)
    print(f"[quantize] saved {EMBEDDER_INT8_OUT}")

    # Load fitted temperature T from the file written by train_setfit.py.
    temp_path = Path(TEMPERATURE_FILE)
    if temp_path.exists():
        temperature = json.loads(temp_path.read_text())["temperature"]
        print(f"[temperature] loaded T={temperature:.4f} from {TEMPERATURE_FILE}")
    else:
        temperature = 1.0
        print(f"[temperature] {TEMPERATURE_FILE} not found, using T=1.0 (no scaling)")

    # Dump LR head to JSON. classes_ order is what argmax maps to.
    labels = head.classes_.tolist()
    head_data = {
        "coef": head.coef_.tolist(),
        "intercept": head.intercept_.tolist(),
        "labels": labels,
        "temperature": temperature,
    }
    Path(HEAD_OUT).write_text(json.dumps(head_data, indent=2) + "\n")
    print(f"[head] saved {HEAD_OUT}  (coef {head.coef_.shape}, labels {labels}, T={temperature:.4f})")

    # Copy tokenizer for the Rust consumer.
    shutil.copy(f"{SETFIT_DIR}/tokenizer.json", TOKENIZER_OUT)
    print(f"[tokenizer] copied -> {TOKENIZER_OUT}")

    # Print I/O signature.
    fp32_sess = ort.InferenceSession(EMBEDDER_OUT, providers=["CPUExecutionProvider"])
    _print_session_info(fp32_sess)

    # File sizes.
    fp32_bytes = Path(EMBEDDER_OUT).stat().st_size
    int8_bytes = Path(EMBEDDER_INT8_OUT).stat().st_size
    print(f"\nfile sizes:")
    print(f"  {EMBEDDER_OUT}: {fp32_bytes / 1e6:.1f} MB")
    print(f"  {EMBEDDER_INT8_OUT}: {int8_bytes / 1e6:.1f} MB")

    # Load holdout data and PyTorch reference predictions.
    # Apply preprocess so parity is checked on the same normalization Aegis uses.
    examples = load(EVAL_FILE)
    texts = [preprocess(e.text) for e in examples]
    torch_preds = [str(p) for p in model.predict(texts)]

    coef = head.coef_
    intercept = head.intercept_

    tokenizer = transformers.AutoTokenizer.from_pretrained(SETFIT_DIR)

    # FP32 parity gate: ALL examples must match.
    fp32_preds = _run_onnx_preds(fp32_sess, tokenizer, texts, coef, intercept, labels)
    mismatches = [
        (i, texts[i], torch_preds[i], fp32_preds[i])
        for i in range(len(texts))
        if torch_preds[i] != fp32_preds[i]
    ]
    match_count = len(texts) - len(mismatches)
    print(f"\nfp32 parity: {match_count}/{len(texts)} match")
    if mismatches:
        print("MISMATCH rows (idx, torch_pred, onnx_pred, text):")
        for idx, text, tp, op in mismatches:
            print(f"  [{idx}] torch={tp!r}  onnx={op!r}  text={text!r}")
        raise SystemExit(1)
    print("fp32 parity ok")

    # INT8 check: report accuracy, do not hard-fail.
    int8_sess = ort.InferenceSession(
        EMBEDDER_INT8_OUT, providers=["CPUExecutionProvider"]
    )
    int8_preds = _run_onnx_preds(
        int8_sess, tokenizer, texts, coef, intercept, labels
    )
    int8_match = sum(tp == ip for tp, ip in zip(torch_preds, int8_preds))
    print(
        f"int8 accuracy: {int8_match}/{len(texts)} "
        f"({100 * int8_match / len(texts):.1f}%)"
    )


if __name__ == "__main__":
    main()
