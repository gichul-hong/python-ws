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

def make_features(word_g=(1,4), char_g=(3,5)):
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=word_g, sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=char_g, sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=char_g, sublinear_tf=True, min_df=2, max_df=0.95)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)),
    ])

# ---- 1. class_weight='balanced' ----
print("=== class_weight=balanced ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto', class_weight='balanced'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42, class_weight='balanced')
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
print(f"  balanced: {acc:.4f}")

# ---- 2. class_weight on final only ----
print("\n=== class_weight on final only ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto', class_weight='balanced'),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  balanced final: {acc:.4f}")

# ---- 3. Best so far with tuned base model hyperparams ----
print("\n=== Best config: tune base models ===")
for svc_C in [3.0, 5.0, 10.0]:
    for lr_C in [10.0, 20.0, 50.0]:
        feats = make_features()
        svc_clf = CalibratedClassifierCV(LinearSVC(C=svc_C, max_iter=20000, random_state=42, dual='auto'), cv=3)
        lr_clf = LogisticRegression(C=lr_C, max_iter=20000, random_state=42)
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
        print(f"  svc_C={svc_C}, lr_C={lr_C}: {acc:.4f}")

# ---- 4. Try max_features to limit vocabulary ----
print("\n=== max_features tuning ===")
for max_feat in [30000, 50000, 80000, None]:
    feats = FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95, max_features=max_feat)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95, max_features=max_feat)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95, max_features=max_feat)),
        ('word_binary', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95, max_features=max_feat)),
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
    print(f"  max_features={max_feat}: {acc:.4f}")

# ---- 5. Try adding a 5th base model: ComplementNB ----
print("\n=== 5 base models (add ComplementNB) ===")
from sklearn.naive_bayes import ComplementNB
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
svc_clf2 = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
cnb_clf = ComplementNB()
stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('svc2', svc_clf2), ('lr', lr_clf), ('ridge', ridge_clf), ('cnb', cnb_clf)],
    final_estimator=LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_text, y_train)
y_pred = pipe.predict(X_test_text)
acc = accuracy_score(y_test, y_pred)
print(f"  5 base models: {acc:.4f}")