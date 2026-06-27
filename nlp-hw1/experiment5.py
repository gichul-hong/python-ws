import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
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

# Tune final LinearSVC C
print("=== Stacking: final=LinearSVC tune C ===")
feats = make_rich_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

for final_C in [0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]:
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LinearSVC(C=final_C, max_iter=20000, random_state=42, dual='auto'),
        cv=3,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  final_LinearSVC C={final_C}: {acc:.4f}")

# Try wider n-gram ranges
print("\n=== Wider n-gram ranges ===")
for word_g in [(1,3), (1,4), (1,5)]:
    for char_g in [(3,5), (3,6), (2,6)]:
        feats = FeatureUnion([
            ('word', TfidfVectorizer(analyzer='word', ngram_range=word_g, sublinear_tf=True, min_df=2, max_df=0.95)),
            ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=char_g, sublinear_tf=True, min_df=2, max_df=0.95)),
            ('char', TfidfVectorizer(analyzer='char', ngram_range=char_g, sublinear_tf=True, min_df=2, max_df=0.95)),
            ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
        ])
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
        print(f"  word={word_g}, char={char_g}: {acc:.4f}")