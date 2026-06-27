import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

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
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
    ])

# Precompute features
print("Computing features...")
feats = make_features()
X_train_feat = feats.fit_transform(X_train_text)
X_test_feat = feats.transform(X_test_text)
print(f"Feature shape: {X_train_feat.shape}")

# L2 normalize for kNN
X_train_norm = normalize(X_train_feat, norm='l2')
X_test_norm = normalize(X_test_feat, norm='l2')

# Test kNN with cosine similarity (metric='cosine' on normalized = euclidean)
print("\n=== kNN ===")
for k in [3, 5, 7, 10, 15, 20]:
    for weights in ['uniform', 'distance']:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights=weights)
        knn.fit(X_train_norm, y_train)
        y_pred = knn.predict(X_test_norm)
        acc = accuracy_score(y_test, y_pred)
        print(f"  k={k}, weights={weights}: {acc:.4f}")

# Test LinearSVC with L2 normalized features
print("\n=== LinearSVC with normalized features ===")
for C in [1.0, 5.0, 10.0, 20.0]:
    clf = LinearSVC(C=C, max_iter=20000, random_state=42, dual='auto')
    clf.fit(X_train_norm, y_train)
    y_pred = clf.predict(X_test_norm)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LinearSVC C={C} (normalized): {acc:.4f}")

# Test LR with normalized features
print("\n=== LR with normalized features ===")
for C in [5.0, 10.0, 20.0, 50.0]:
    clf = LogisticRegression(C=C, max_iter=20000, random_state=42)
    clf.fit(X_train_norm, y_train)
    y_pred = clf.predict(X_test_norm)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LR C={C} (normalized): {acc:.4f}")