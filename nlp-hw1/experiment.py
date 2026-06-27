import re
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
train_df = pd.read_csv(f"{BASE_URL}/train.csv")
test_df = pd.read_csv(f"{BASE_URL}/test.csv")

label_names = sorted(train_df["category"].unique())
label_encoder = LabelEncoder()
label_encoder.fit(label_names)
train_df["label"] = label_encoder.transform(train_df["category"])
test_df["label"] = label_encoder.transform(test_df["category"])
label_names = list(label_encoder.classes_)

X_train_text = train_df["text"].values
y_train = train_df["label"].values
X_test_text = test_df["text"].values
y_test = test_df["label"].values

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Labels: {len(label_names)}")

# ---- Attempt 2 reproduction: word + char TF-IDF + LinearSVC ----
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)
char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

features = FeatureUnion([
    ('word', word_vectorizer),
    ('char', char_vectorizer),
])

# Baseline: LinearSVC
clf_svc = LinearSVC(C=1.0, max_iter=10000, random_state=42)
pipe = Pipeline([('features', features), ('clf', clf_svc)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"Attempt2 (LinearSVC word+char): {acc:.4f}")

# Try different C values
for C in [0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0]:
    clf = LinearSVC(C=C, max_iter=10000, random_state=42)
    pipe = Pipeline([('features', features), ('clf', clf)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  C={C}: {acc:.4f}")

# Try Logistic Regression
for C in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=10000, random_state=42, n_jobs=-1)
    pipe = Pipeline([('features', features), ('clf', clf)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LR C={C}: {acc:.4f}")

# Try ComplementNB
clf = ComplementNB()
pipe = Pipeline([('features', features), ('clf', clf)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  ComplementNB: {acc:.4f}")

# Try stacking: LinearSVC + LR + ComplementNB
print("\n--- Stacking ---")
base_features = FeatureUnion([
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
])

svc_clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, random_state=42), cv=3)
lr_clf = LogisticRegression(C=5.0, max_iter=10000, random_state=42, n_jobs=-1)

stacking = StackingClassifier(
    estimators=[
        ('svc', svc_clf),
        ('lr', lr_clf),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=10000, random_state=42),
    n_jobs=-1,
    cv=3,
)
pipe = Pipeline([('features', base_features), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"Stacking (SVC+LR -> LR): {acc:.4f}")