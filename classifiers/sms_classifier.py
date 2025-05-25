
# sms_classification_baseline.py
# Classical Machine Learning Pipeline for SMS Classification
# Author: Shubham Meher

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load dataset
data = pd.read_csv('data/input_sms_messages.csv', encoding='latin-1')

# Check for nulls and preprocess
df = data.copy()
df = df.dropna()

df['clean_score'] = df['score'].astype(int)
df['clean_sms_type1_id'] = df['sms_type1_id'].astype(int)
df['clean_sms_type2_id'] = df['sms_type2_id'].astype(int)
df['sms_type_n_score'] = (
    df['clean_score'] * 100 +
    df['clean_sms_type1_id'] * 10 +
    df['clean_sms_type2_id']
)

# Download stopwords and punkt
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
ps = PorterStemmer()

# Text transformation
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Apply transformation
df['transformed_body'] = df['body'].astype(str).apply(transform_text)
df['transformed_address'] = df['clean_address'].astype(str)

# Vectorize
cv_body = CountVectorizer()
cv_address = CountVectorizer()
X_body = cv_body.fit_transform(df['transformed_body']).toarray()
X_address = cv_address.fit_transform(df['transformed_address']).toarray()
X = np.hstack((X_body, X_address))
y = df['sms_type_n_score'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Initialize classifiers
svc = SVC(kernel='sigmoid', gamma=1.0)
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

# Evaluation function
def train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train, average='weighted')
    train_recall = recall_score(y_train, y_pred_train, average='weighted')
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    return train_accuracy, train_precision, train_recall, test_accuracy, test_precision, test_recall

# Choose classifiers to evaluate
clfs = {
     'SVC': svc,
     'DT': dtc,
     'LR': lrc,
    'RF': rfc,
     'AdaBoost': abc,
     'BgC': bc,
     'GBDT': gbdt,
     'xgb': xgb
}

# Run evaluations
for name, clf in clfs.items():
    train_accuracy, train_precision, train_recall, test_accuracy, test_precision, test_recall = train_and_evaluate_classifier(
        clf, X_train, y_train, X_test, y_test)

    print(f"\nFor {name}")
    print("Training Data:")
    print(f"  Accuracy - {train_accuracy}")
    print(f"  Precision - {train_precision}")
    print(f"  Recall - {train_recall}")
    print("Test Data:")
    print(f"  Accuracy - {test_accuracy}")
    print(f"  Precision - {test_precision}")
    print(f"  Recall - {test_recall}")
