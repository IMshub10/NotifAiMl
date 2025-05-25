
# 📊 MobileBERT SMS Classifier — Fine-tuned Transformer for Multi-Class SMS Classification

This project fine-tunes `google/mobilebert-uncased` to classify SMS messages using a composite label strategy that captures score, message type hierarchy, and sub-type. It's part of the NotifAI SMS intelligence system.

---

## 📥 Dataset & Preprocessing

- Total Records: **35,000+**
- Input Format: `Sender: <sender_id> | Message: <sms_body>`
- Labels: Composite class `sms_type_n_score = (score * 10000) + ((type1 * 10 + type2) * 100) + type_id`
- Tokenization: Max length = 128 tokens using `MobileBertTokenizerFast`

---

## ⚙️ Training Setup

| Parameter         | Value                |
|------------------|----------------------|
| Model            | `google/mobilebert-uncased` |
| Batch Size       | 8                    |
| Epochs           | 5                    |
| Learning Rate    | 3e-5                 |
| Dropout          | 0.2 (attention + hidden) |
| Device           | MPS / CPU            |
| Optimizer        | Handled via `Trainer` (AdamW) |
| Evaluation       | Every epoch          |
| Best Model       | Loaded at end using eval loss |

---

## 📈 Metrics Per Epoch

| Epoch | Train Loss | Val Loss | Accuracy | Precision | Recall | F1 Score |
|-------|------------|----------|----------|-----------|--------|----------|
| 1     | 0.3137     | 0.3209   | 0.9225   | 0.9271    | 0.9225 | 0.9224   |
| 2     | 0.2255     | 0.2404   | 0.9422   | 0.9447    | 0.9422 | 0.9419   |
| 3     | 0.1290     | 0.7333   | 0.9541   | 0.9541    | 0.9541 | 0.9533   |
| 4     | 0.0828     | 0.2173   | 0.9597   | 0.9599    | 0.9597 | 0.9595   |
| 5     | 0.0488     | 0.2240   | **0.9634** | **0.9633** | **0.9634** | **0.9631** |

---

## ✅ Final Evaluation on Test Set

- **Accuracy:** 96.34%
- **Precision:** 96.33%
- **Recall:** 96.34%
- **F1 Score:** 96.31%

---

## 📦 Model Artifacts

Saved under `./mobilebert_sms_model/`:

- `pytorch_model.bin` — Trained MobileBERT weights
- `tokenizer_config.json` & `vocab.txt` — Tokenizer setup
- `label_encoder.pkl` — Scikit-learn label encoder
- `config.json` — HuggingFace model config

---

## 📌 Notes

- This model was trained to run on-device using ONNX/TFLite for Android inference.
- Tokenizer was built from a real SMS corpus to capture domain-specific tokens.
- For best results, inference should preserve the "Sender: | Message:" input format.

