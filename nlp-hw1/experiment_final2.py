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

def make_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
    ])

# ---- 1. Best config: svc_C=10, lr_C=50, tune final C ----
print("=== svc_C=10, lr_C=50: tune final C ===")
for final_C in [0.3, 0.5, 0.7, 1.0, 2.0, 3.0]:
    feats = make_features()
    svc_clf = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
    lr_clf = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
    ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LinearSVC(C=final_C, max_iter=20000, random_state=42, dual='auto'),
        cv=3,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  final_C={final_C}: {acc:.4f}")

# ---- 2. Best config + class_weight=balanced ----
print("\n=== svc_C=10, lr_C=50 + balanced ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto', class_weight='balanced'), cv=3)
lr_clf = LogisticRegression(C=50.0, max_iter=20000, random_state=42, class_weight='balanced')
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42, class_weight='balanced')
for final_C in [0.5, 1.0]:
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LinearSVC(C=final_C, max_iter=20000, random_state=42, dual='auto'),
        cv=3,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  all balanced, final_C={final_C}: {acc:.4f}")

# ---- 3. Tune ridge alpha ----
print("\n=== svc_C=10, lr_C=50: tune ridge alpha ===")
for ridge_alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
    feats = make_features()
    svc_clf = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
    lr_clf = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
    ridge_clf = RidgeClassifier(alpha=ridge_alpha, random_state=42)
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'),
        cv=3,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  ridge_alpha={ridge_alpha}: {acc:.4f}")

# ---- 4. Try cv=5 for stacking ----
print("\n=== svc_C=10, lr_C=50: cv=5 ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'),
    cv=5,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  cv=5: {acc:.4f}")

# ---- 5. Try final=LR with best base models ----
print("\n=== svc_C=10, lr_C=50: final=LR ===")
for final_C in [0.5, 1.0, 5.0, 10.0]:
    feats = make_features()
    svc_clf = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
    lr_clf = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
    ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LogisticRegression(C=final_C, max_iter=20000, random_state=42),
        cv=3,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  final=LR C={final_C}: {acc:.4f}")