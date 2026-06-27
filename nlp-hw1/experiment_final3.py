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
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier

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

# Precompute features
print("Computing features...")
feats = make_features()
X_train = feats.fit_transform(X_train_text)
X_test = feats.transform(X_test_text)
print(f"Features: {X_train.shape}")

# ---- 1. OneVsOne LinearSVC ----
print("\n=== OneVsOne LinearSVC ===")
for C in [1.0, 5.0, 10.0]:
    clf = OneVsOneClassifier(LinearSVC(C=C, max_iter=20000, random_state=42, dual='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  OvO LinearSVC C={C}: {acc:.4f}")

# ---- 2. OneVsRest LinearSVC ----
print("\n=== OneVsRest LinearSVC ===")
for C in [1.0, 5.0, 10.0]:
    clf = OneVsRestClassifier(LinearSVC(C=C, max_iter=20000, random_state=42, dual='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  OvR LinearSVC C={C}: {acc:.4f}")

# ---- 3. Soft Voting with calibrated models ----
print("\n=== Soft Voting ===")
svc1 = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
svc2 = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr1 = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
lr2 = LogisticRegression(C=10.0, max_iter=20000, random_state=42)
voting = VotingClassifier(
    estimators=[('svc1', svc1), ('svc2', svc2), ('lr1', lr1), ('lr2', lr2)],
    voting='soft',
    weights=[1, 1, 2, 1],
)
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Soft voting (2 SVC + 2 LR): {acc:.4f}")

# Try different weights
for w in [[2, 1, 1, 1], [1, 1, 2, 1], [2, 1, 2, 1], [3, 1, 2, 1], [1, 1, 3, 1]]:
    voting = VotingClassifier(
        estimators=[('svc1', svc1), ('svc2', svc2), ('lr1', lr1), ('lr2', lr2)],
        voting='soft',
        weights=w,
    )
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  weights={w}: {acc:.4f}")

# ---- 4. Stacking + OvO ----
print("\n=== Stacking + OvO final ===")
svc_clf = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=OneVsOneClassifier(LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto')),
    cv=3,
)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Stacking -> OvO LinearSVC: {acc:.4f}")

# ---- 5. TruncatedSVD + classifier ----
print("\n=== TruncatedSVD ===")
for n_comp in [200, 500, 1000, 2000]:
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)
    for C in [5.0, 10.0, 50.0]:
        clf = LogisticRegression(C=C, max_iter=20000, random_state=42)
        clf.fit(X_train_svd, y_train)
        y_pred = clf.predict(X_test_svd)
        acc = accuracy_score(y_test, y_pred)
        print(f"  SVD({n_comp}) + LR C={C}: {acc:.4f}")