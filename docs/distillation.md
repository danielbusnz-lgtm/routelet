# Distillation: teacher labels, student ships

The smart LLM teaches. The small model serves. Only the small model ships.

- **Teacher**: a strong LLM (Claude). Offline only, used to label and correct logged real commands. Never in production.
- **Student**: SetFit (small, fast, on-device). Trained to imitate the teacher's labels. The only model deployed.

## Loop

1. Production: the student classifies on-device. Log each `(redacted_text, predicted_intent)`.
2. Offline: the teacher labels/corrects those logged commands. Its labels are the training truth.
3. Train the student on the teacher-labeled data.
4. Gate on `evals/holdout.jsonl`: promote only if it beats the current student.
5. Ship the new student. The teacher never enters production.

## Why

- Teacher: accurate, but slow, costly, cloud-only. Fine for occasional offline labeling.
- Student: ~92% at ~2ms, on-device, free, offline, private. Fine for every command.
- Result: the teacher's quality at the student's cost.

Serving and redaction details: see `serving.md`.
