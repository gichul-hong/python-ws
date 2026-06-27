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
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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

stemmer = PorterStemmer()

def stem_text(text):
    text = str(text).lower()
    text = re.sub(r"([.,!?;:()\[\]{}\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join([stemmer.stem(w) for w in words])

X_train_stem = np.array([stem_text(t) for t in X_train_text])
X_test_stem = np.array([stem_text(t) for t in X_test_text])

print("=== Stemming + word+char TF-IDF ===")
def make_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)),
        ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
    ])

for C in [1.0, 5.0, 10.0]:
    feats = make_features()
    clf = LinearSVC(C=C, max_iter=20000, random_state=42, dual='auto')
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_stem, y_train)
    y_pred = pipe.predict(X_test_stem)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Stem LinearSVC C={C}: {acc:.4f}")

for C in [10.0, 20.0, 50.0]:
    feats = make_features()
    clf = LogisticRegression(C=C, max_iter=20000, random_state=42)
    pipe = Pipeline([('features', feats), ('clf', clf)])
    pipe.fit(X_train_stem, y_train)
    y_pred = pipe.predict(X_test_stem)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Stem LR C={C}: {acc:.4f}")

# Stacking with stemmed features
print("\n=== Stacking with stemming ===")
feats = make_features()
svc_clf = CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=20.0, max_iter=20000, random_state=42)
ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)

stacking = StackingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf), ('ridge', ridge_clf)],
    final_estimator=LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto'),
    cv=3,
)
pipe = Pipeline([('features', feats), ('clf', stacking)])
pipe.fit(X_train_stem, y_train)
y_pred = pipe.predict(X_test_stem)
acc = accuracy_score(y_test, y_pred)
print(f"  Stem stacking: {acc:.4f}")

# Error analysis: look at confusion matrix top pairs
print("\n=== Error Analysis ===")
feats = FeatureUnion([
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
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
print(f"Best stacking: {acc:.4f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Find most confused pairs
np.fill_diagonal(cm, 0)
pairs = []
for i in range(len(label_names)):
    for j in range(len(label_names)):
        if cm[i][j] > 3:
            pairs.append((cm[i][j], label_names[i], label_names[j]))
pairs.sort(reverse=True)
print("\nMost confused pairs:")
for count, true_label, pred_label in pairs[:20]:
    print(f"  {count}: {true_label} -> {pred_label}")