
# 📬 NotifAI — SMS Intelligence Engine for Android

NotifAI is an on-device SMS classification engine designed to bring clarity and security to your inbox. Built with mobile-first performance and privacy in mind, it uses a fine-tuned transformer model to label SMS messages into meaningful categories like transactions, promotions, spam, service alerts, and more.

---

## 🚀 Why NotifAI?

📱 SMS apps today either compromise your privacy or lack smart filtering.  
🔐 NotifAI classifies and flags messages **on-device** with zero data sharing.  
🧠 Models are trained on 35K+ real-world SMS messages with 45+ labels including importance scores.

---

## ✨ Features

- ⚡ Fast & lightweight **on-device ML inference** (MobileBERT, DistilBERT)
- 📊 Classifies messages into **transactional, promotional, personal, spam**, and more
- 🚫 Flags potential **scam/fraud messages**
- 📣 Notifies based on **user-defined importance scores**
- 🔍 Intelligent grouping by sender & thread
- 🔐 **Privacy-first:** No SMS is uploaded without user consent

---

## 🧠 Machine Learning Overview

- Classical ML Models: Logistic Regression, Random Forest, XGBoost
- Transformer Models: Fine-tuned DistilBERT, MobileBERT, TinyBERT
- Label Strategy: Composite label using score + type1 + type2 + subtype
- Input Format: `"Sender: <sender_id> | Message: <sms_body>"`

---

## 🛠️ Tech Stack

| Layer           | Stack                                |
|----------------|---------------------------------------|
| ML Training     | Python, HuggingFace Transformers, Pandas, Scikit-learn |
| Tokenization    | Custom tokenizer built from SMS corpus |
| Android         | Kotlin, Jetpack, Room, LiveData       |
| Model Inference | TFLite / ONNX for Android             |
| Backend (Optional) | Spring Boot + MongoDB Atlas           |

---

## 📁 Repository Structure

```
NotifAI-ML/
│
├── classifiers/         # Logistic Regression, Random Forest, XGBoost scripts
│   └── sms_classification_baseline.py
│
├── mobile_bert/              # Fine-tuned transformer scripts
│   ├── mobile_bert_training.py
│   ├── README_mobilebert_sms.md
│   └── mobilebert_sms_model/
│
├── tokenizer_vocab.json      # Tokenizer vocab built from SMS dataset
├── sample_data_set/                     # Sample or real SMS data (not included in repo)
│   └── sample_input_file.csv
│
└── README.md                 # Main documentation (you are here)
```

---

## 📦 Model Output & Artifacts

Trained models and tokenizer saved under:

- `mobilebert_sms_model/`
  - `pytorch_model.bin`
  - `tokenizer_config.json`, `vocab.txt`
  - `label_encoder.pkl`
  - `config.json`

---

## 🧪 Sample Usage (HuggingFace Inference)

```python
from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification
import torch

model = MobileBertForSequenceClassification.from_pretrained("./mobilebert_sms_model")
tokenizer = MobileBertTokenizerFast.from_pretrained("./mobilebert_sms_model")

msg = "Sender: AXISBANK | Message: Rs.5000 debited from A/C 1234 at Amazon"
inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=-1)
print(pred)
```

---

## 📌 License

MIT — use it freely for research, productization, or contributions.

---

