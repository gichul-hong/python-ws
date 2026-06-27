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

# Rich feature union: word (1-3), char_wb (3-5), char (3-5) + binary word
def make_rich_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
    ])

# Test: Rich features + LinearSVC
print("=== Rich features + LinearSVC ===")
for C in [1.0, 3.0, 5.0, 10.0]:
    feats = make_rich_features()
    clf = LinearSVC(C=C, max_iter=20000, random_state=42, dual='auto')
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LinearSVC C={C}: {acc:.4f}")

# Test: Rich features + LR
print("\n=== Rich features + LR ===")
for C in [5.0, 10.0, 20.0, 50.0]:
    feats = make_rich_features()
    clf = LogisticRegression(C=C, max_iter=20000, random_state=42)
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LR C={C}: {acc:.4f}")

# Test: Stacking with rich features
print("\n=== Stacking with rich features ===")
feats = make_rich_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=LogisticRegression(C=1.0, max_iter=20000, random_state=42),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"Stacking rich: {acc:.4f}")

# Test: Just word (1-3) + char_wb (3-5) but with different min_df
print("\n=== min_df tuning ===")
for min_df in [1, 2, 3, 5]:
    feats = FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=min_df, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=min_df, max_df=0.95)),
    ])
    clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  min_df={min_df} LR C=20: {acc:.4f}")