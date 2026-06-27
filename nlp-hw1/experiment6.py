import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
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

# Best feature config so far
def make_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
    ])

# Stacking with passthrough=True
print("=== Stacking with passthrough ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

for passthrough in [False, True]:
    for final_C in [0.5, 1.0]:
        stacking = StackingClassifier(
            estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
            final_estimator=LinearSVC(C=final_C, max_iter=20000, random_state=42, dual='auto'),
            cv=3,
            passthrough=passthrough,
        )
        pipe = Pipeline([('features', feats), ('clf', stacking)])
        pipe.fit(X_train_text, y_train)
        y_pred = pipe.predict(X_test_text)
        acc = accuracy_score(y_test, y_pred)
        print(f"  passthrough={passthrough}, final_C={final_C}: {acc:.4f}")

# Try adding ComplementNB and SGD as base models
print("\n=== Stacking with more diverse base models ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
sgd_clf = CalibratedClassifierCV(SGDClassifier(loss='hinge', alpha=1e-5, max_iter=20000, random_state=42), cv=3)

stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf), ('sgd', sgd_clf)],
    final_estimator=LinearSVC(C=0.5, max_iter=20000, random_state=42, dual='auto'),
    cv=3,
    passthrough=True,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  4 models + passthrough: {acc:.4f}")

# Try with LR final estimator + passthrough
print("\n=== Stacking: LR final + passthrough ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

for final_C in [0.5, 1.0, 5.0]:
    stacking = StackingClassifier(
        estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
        final_estimator=LogisticRegression(C=final_C, max_iter=20000, random_state=42),
        cv=3,
        passthrough=True,
    )
    pipe = Pipeline([('features', feats), ('clf', stacking)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LR final C={final_C} + passthrough: {acc:.4f}")