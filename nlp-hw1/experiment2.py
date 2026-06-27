import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"(\d+)", " NUM ", text)
    text = re.sub(r"([.,!?;:()\[\]{}\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

X_train_prep = np.array([preprocess(t) for t in X_train_text])
X_test_prep = np.array([preprocess(t) for t in X_test_text])

def make_features(word_grams=(1,3), char_grams=(3,5), min_df=2, max_df=0.95):
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=word_grams, sublinear_tf=True, min_df=min_df, max_df=max_df)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=char_grams, sublinear_tf=True, min_df=min_df, max_df=max_df)),
    ])

# Test 1: Preprocessing effect with LinearSVC
print("=== Preprocessing + LinearSVC ===")
for C in [5.0, 10.0]:
    feats = make_features()
    clf = LinearSVC(C=C, max_iter=20000, random_state=42)
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_prep, y_train)
    y_pred = pipe.predict(X_test_prep)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LinearSVC C={C}: {acc:.4f}")

# Test 2: LR with preprocessing
print("\n=== Preprocessing + LR ===")
for C in [10.0, 20.0, 50.0]:
    feats = make_features()
    clf = LogisticRegression(C=C, max_iter=20000, random_state=42)
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_prep, y_train)
    y_pred = pipe.predict(X_test_prep)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LR C={C}: {acc:.4f}")

# Test 3: Stacking SVC+LR with preprocessing
print("\n=== Stacking SVC+LR (prep) ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42), cv=3)
lr_clf = LogisticRegression(C=10.0, max_iter=20000, random_state=42)
stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf)],
    final_estimator=LogisticRegression(C=1.0, max_iter=20000, random_state=42),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_prep, y_train)
y_pred = pipe.predict(X_test_prep)
acc = accuracy_score(y_test, y_pred)
print(f"Stacking prep: {acc:.4f}")

# Test 4: No preprocessing, higher C for LR
print("\n=== LR no prep, higher C ===")
for C in [10.0, 20.0, 50.0]:
    feats = make_features()
    clf = LogisticRegression(C=C, max_iter=20000, random_state=42)
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    print(f"  LR C={C}: {acc:.4f}")