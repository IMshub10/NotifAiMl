# Classical Classifier Results on SMS Classification

## Dataset:
- 35K+ SMS records
- Custom composite label: `sms_type_n_score = (score × 100) + (type1 × 10) + type2`
- Features: `transformed_body + transformed_address`

## Preprocessing:
- Lowercasing, tokenization, stopword removal, stemming (PorterStemmer)
- CountVectorizer for both body and sender address

## Classifiers Evaluated:
| Classifier     | Test Accuracy | Precision | Recall |
|----------------|---------------|-----------|--------|
| LogisticRegression | 95.1% | 95.06% | 95.1% |
| RandomForest   | **96.87%**    | **96.88%**| **96.87%** |
| GradientBoost  | 87.88%        | 88.97%    | 87.88% |
| XGBoost        | 92.94%        | 93.19%    | 92.94% |
| Bagging        | 96.17%        | 96.16%    | 96.17% |
| AdaBoost       | 73.81%        | 77.53%    | 73.81% |
| DecisionTree   | 72.61%        | 80.24%    | 72.61% |
| SVM (sigmoid)  | ~43%          | ~44%      | ~43%   |

## Notes:
- Random Forest performed best overall for generalization.
- XGBoost requires class labels to start from 0.

## Limitations:
- Doesn't scale well to unseen templates.
- Generalization to new senders is limited.

## Next Steps:
➡ Fine-tuned DistilBERT, MobileBERT, TinyBERT models with tokenizer_vocab.json for better generalization.