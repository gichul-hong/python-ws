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

# Approach: each base model gets a different feature pipeline
# Model 1: word (1-3) + char_wb (3-5) + LinearSVC
# Model 2: word (1-4) + char (3-6) + LR
# Model 3: word (1-2) + char_wb (4-6) + RidgeClassifier
# Model 4: word (1-3) + char_wb (3-5) + LinearSVC (different C)

# Precompute features for different configurations
print("Precomputing features...")

# Feature set A: word(1-3) + char_wb(3-5)
feats_A = FeatureUnion([
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)),
])
X_train_A = feats_A.fit_transform(X_train_text)
X_test_A = feats_A.transform(X_test_text)

# Feature set B: word(1-4) + char(3-6) + char_wb(3-6)
feats_B = FeatureUnion([
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
])
X_train_B = feats_B.fit_transform(X_train_text)
X_test_B = feats_B.transform(X_test_text)

# Feature set C: word(1-2) + char_wb(4-6)
feats_C = FeatureUnion([
    ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)),
    ('char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 6), sublinear_tf=True, min_df=2, max_df=0.95)),
])
X_train_C = feats_C.fit_transform(X_train_text)
X_test_C = feats_C.transform(X_test_text)

print(f"Features A: {X_train_A.shape}, B: {X_train_B.shape}, C: {X_train_C.shape}")

# Test each model on each feature set
for name, Xtr, Xte in [("A", X_train_A, X_test_A), ("B", X_train_B, X_test_B), ("C", X_train_C, X_test_C)]:
    for C in [1.0, 5.0, 10.0]:
        clf = LinearSVC(C=C, max_iter=20000, random_state=42, dual='auto')
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {name} LinearSVC C={C}: {acc:.4f}")

# Now use individual feature pipelines within StackingClassifier
print("\n=== Stacking with separate feature pipelines ===")

# We need to create a custom approach since StackingClassifier uses the same X for all estimators
# We'll use Pipeline within each estimator

# Actually, let's try a different approach: create the stacking manually
from sklearn.model_selection import StratifiedKFold

def manual_stacking():
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_classes = len(label_names)
    
    # Base models
    models = {
        'svc_A': lambda: CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=20000, random_state=42, dual='auto'), cv=3),
        'svc_B': lambda: CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3),
        'lr_A': lambda: LogisticRegression(C=20.0, max_iter=20000, random_state=42),
        'lr_B': lambda: LogisticRegression(C=10.0, max_iter=20000, random_state=42),
        'ridge_C': lambda: RidgeClassifier(alpha=1.0, random_state=42),
    }
    
    feature_sets = {
        'svc_A': (X_train_A, X_test_A),
        'svc_B': (X_train_B, X_test_B),
        'lr_A': (X_train_A, X_test_A),
        'lr_B': (X_train_B, X_test_B),
        'ridge_C': (X_train_C, X_test_C),
    }
    
    # Generate out-of-fold predictions for training the meta-learner
    n_models = len(models)
    meta_train = np.zeros((len(y_train), n_models * n_classes))
    meta_test = np.zeros((len(y_test), n_models * n_classes))
    
    for i, (name, model_fn) in enumerate(models.items()):
        X_tr, X_te = feature_sets[name]
        print(f"  Training {name}...")
        oof_pred = np.zeros((len(y_train), n_classes))
        test_pred = np.zeros((len(y_test), n_classes))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
            model = model_fn()
            model.fit(X_tr[train_idx], y_train[train_idx])
            if hasattr(model, 'predict_proba'):
                oof_pred[val_idx] = model.predict_proba(X_tr[val_idx])
                test_pred += model.predict_proba(X_te) / 3
            else:
                # For RidgeClassifier, use decision_function
                proba = np.zeros((len(val_idx) if fold == 0 else 0, n_classes))
                dec = model.decision_function(X_tr[val_idx])
                # softmax-like normalization
                dec = dec - dec.max(axis=1, keepdims=True)
                exp_dec = np.exp(dec)
                oof_pred[val_idx] = exp_dec / exp_dec.sum(axis=1, keepdims=True)
                
                dec_test = model.decision_function(X_te)
                dec_test = dec_test - dec_test.max(axis=1, keepdims=True)
                exp_test = np.exp(dec_test)
                test_pred += (exp_test / exp_test.sum(axis=1, keepdims=True)) / 3
        
        meta_train[:, i*n_classes:(i+1)*n_classes] = oof_pred
        meta_test[:, i*n_classes:(i+1)*n_classes] = test_pred
    
    # Train meta-learner
    print("  Training meta-learner...")
    for meta_C in [0.1, 0.5, 1.0]:
        meta = LinearSVC(C=meta_C, max_iter=20000, random_state=42, dual='auto')
        meta.fit(meta_train, y_train)
        y_pred = meta.predict(meta_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Meta LinearSVC C={meta_C}: {acc:.4f}")
    
    for meta_C in [0.5, 1.0, 5.0]:
        meta = LogisticRegression(C=meta_C, max_iter=20000, random_state=42)
        meta.fit(meta_train, y_train)
        y_pred = meta.predict(meta_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Meta LR C={meta_C}: {acc:.4f}")

manual_stacking()