# experiments

Throwaway prototypes that are **not** part of the routelet library or its build.
Kept for reference, not maintained.

- `tool_router.py`, `tool_classifier_proto.py`, `tools_proto.py`: an earlier
  exploration of a *tool*-routing classifier (spotify / gmail / github / youtube
  / no_tool), separate from the shipped 5-intent router.
- `run_ood_eval.py`: the out-of-distribution eval for that experiment, against
  `evals/tool_routing_ood.jsonl` and `models/setfit_tools_proto`.

These import `routelet.preprocess`, so run them through the project env:

    uv run python experiments/tool_router.py
