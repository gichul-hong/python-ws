import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

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

def make_rich_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
    ])

# Stacking with different final estimators
print("=== Stacking: tune final estimator ===")
feats = make_rich_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

for final_C in [0.1, 0.5, 1.0, 5.0, 10.0]:
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LogisticRegression(C=final_C, max_iter=20000, random_state=42),
        cv=3,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  final_LR C={final_C}: {acc:.4f}")

# Try more base models
print("\n=== Stacking: more base models ===")
feats = make_rich_features()
svc1 = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
svc2 = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
svc3 = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr1 = LogisticRegression(C=5.0, max_iter=20000, random_state=42)
lr2 = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge = RidgeClassifier(alpha=1.0, random_state=42)

stacking = StackingClassifier(
    estimators=[('svc1', svc1), ('svc2', svc2), ('svc3', svc3), ('lr1', lr1), ('lr2', lr2), ('ridge', ridge)],
    final_estimator=LogisticRegression(C=1.0, max_iter=20000, random_state=42),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  6 base models: {acc:.4f}")

# Try cv=5 for stacking
print("\n=== Stacking: cv=5 ===")
feats = make_rich_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=LogisticRegression(C=1.0, max_iter=20000, random_state=42),
    cv=5,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  cv=5: {acc:.4f}")

# Try with final estimator = LinearSVC
print("\n=== Stacking: final=LinearSVC ===")
feats = make_rich_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  final=LinearSVC: {acc:.4f}")