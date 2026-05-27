"""Export the trained SetFit model to ONNX and verify parity with PyTorch.

Three files land in models/ after this runs:
  routelet.onnx   the fused body+head graph, ready for tract/onnxruntime
  tokenizer.json  copied from the saved SetFit model, needed by the Rust consumer
  labels.json     index -> intent string, matching the ONNX argmax axis

Run train_setfit.py first so models/setfit exists.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import transformers
from setfit import SetFitModel
from setfit.exporters.onnx import export_onnx

from routelet.data import load

SETFIT_DIR = "models/setfit"
ONNX_OUT = "models/routelet.onnx"
LABELS_OUT = "models/labels.json"
TOKENIZER_OUT = "models/tokenizer.json"
EVAL_FILE = "evals/holdout.jsonl"


def _class_order(model: SetFitModel) -> list[str]:
    # The LR head's classes_ array is what sklearn uses for argmax: index 0
    # in the softmax output corresponds to classes_[0], etc. We read it here
    # rather than hardcoding Intent enum order because sklearn sorts class
    # labels lexicographically during fit, which may differ from enum
    # declaration order.
    head = model.model_head
    # LogisticRegression wraps numpy; .tolist() gives plain Python strings.
    return list(head.classes_.tolist())


def main() -> None:
    model = SetFitModel.from_pretrained(SETFIT_DIR)

    labels = _class_order(model)
    print(f"class order from LR head: {labels}")

    Path(LABELS_OUT).parent.mkdir(exist_ok=True)
    Path(LABELS_OUT).write_text(json.dumps(labels, indent=2) + "\n")
    print(f"saved {LABELS_OUT}")

    export_onnx(model.model_body, model.model_head, opset=18, output_path=ONNX_OUT)
    print(f"saved {ONNX_OUT}")

    # Tokenizer is a standalone file the Rust consumer needs for preprocessing.
    shutil.copy(f"{SETFIT_DIR}/tokenizer.json", TOKENIZER_OUT)
    print(f"copied tokenizer -> {TOKENIZER_OUT}")

    # Inspect the ONNX graph's I/O. Printed here so the Rust tract caller knows
    # exact names, dtypes, and shapes without having to open the graph manually.
    sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
    print("\nONNX inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name!r:30s}  dtype={inp.type!s:15s}  shape={inp.shape}")
    print("ONNX outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name!r:30s}  dtype={out.type!s:15s}  shape={out.shape}")

    input_names = {inp.name for inp in sess.get_inputs()}
    has_token_type_ids = "token_type_ids" in input_names
    print(f"\ntoken_type_ids required: {has_token_type_ids}")

    # Parity check: every holdout example must produce the same intent via ONNX
    # argmax as via the PyTorch model.predict path. Any mismatch means the baked
    # head diverged and the export is broken.
    test = load(EVAL_FILE)
    texts = [e.text for e in test]

    tokenizer = transformers.AutoTokenizer.from_pretrained(SETFIT_DIR)
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )
    feed: dict[str, np.ndarray] = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if has_token_type_ids:
        feed["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

    ort_out = sess.run(None, feed)
    # export_onnx with sklearn head outputs label strings directly (skl2onnx
    # ZipMap node); fall back to argmax on float scores if output is numeric.
    raw = ort_out[0]
    if raw.dtype.kind in ("U", "O"):
        # skl2onnx emits string labels as the primary output.
        ort_preds = [str(v) for v in raw]
    else:
        ort_preds = [labels[int(np.argmax(row))] for row in raw]

    torch_preds = list(model.predict(texts))
    torch_preds = [str(p) for p in torch_preds]

    mismatches = [
        (i, texts[i], torch_preds[i], ort_preds[i])
        for i in range(len(texts))
        if torch_preds[i] != ort_preds[i]
    ]

    print(f"\nparity check: {len(texts) - len(mismatches)}/{len(texts)} match")
    if mismatches:
        print("MISMATCH rows (idx, text, torch, onnx):")
        for row in mismatches:
            print(f"  [{row[0]}] torch={row[2]!r}  onnx={row[3]!r}  text={row[1]!r}")
        raise SystemExit(1)

    print("parity ok")


if __name__ == "__main__":
    main()
