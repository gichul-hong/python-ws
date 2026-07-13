import json
import os

notebook_in = r"C:\hong\python-ws\nlp-hw1\HW1-intent-classification.ipynb"
notebook_out = r"C:\hong\python-ws\nlp-hw1\HW1-intent-classification-improved.ipynb"

# Load notebook
with open(notebook_in, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Use raw string r"""...""" to preserve backslashes exactly!
new_code_cell_35 = r"""# Attempt 3 (최종 모델): Double Stacking Ensemble (Original Text + Clean Text)
import re
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

# --- 전처리 및 정규화 사전 정의 ---
contractions = {
    "can't": "cannot", "don't": "do not", "aren't": "are not", "isn't": "is not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "you're": "you are", "you've": "you have",
    "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are", "they're": "they are"
}

spelling_fixes = {
    "contanctless": "contactless", "depost": "deposit", "referted": "reverted", "accpeted": "accepted",
    "disposble": "disposable", "indentity": "identity", "tryed": "tried", "trye": "try",
    "decline": "declined", "chargeing": "charging", "charges": "charge", "payments": "payment",
    "withdrawn": "withdrawal", "withdraw": "withdrawal"
}

def clean_text_v1(text):
    text = str(text).lower()
    for k, v in contractions.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
        k_no_apo = k.replace("'", "")
        if k_no_apo != k:
            text = re.sub(r"\b" + re.escape(k_no_apo) + r"\b", v, text)
    for k, v in spelling_fixes.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
    text = re.sub(r"([.,!?;:()\[\]{}\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def custom_tokenize(text):
    return text.split()

# --- Model 1: Original Text Features ---
word_A = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95)
char_wb_A = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)
char_A = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)
word_bin_A = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, sublinear_tf=False, min_df=2, max_df=0.95)

rich_features_A = FeatureUnion([('word', word_A), ('char_wb', char_wb_A), ('char', char_A), ('word_bin', word_bin_A)])

# --- Model 2: Cleaned Text Features ---
word_C = TfidfVectorizer(analyzer='word', tokenizer=custom_tokenize, token_pattern=None, ngram_range=(1, 4), sublinear_tf=True, min_df=2, max_df=0.95)
char_wb_C = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)
char_C = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)

rich_features_C = FeatureUnion([('word', word_C), ('char_wb', char_wb_C), ('char', char_C)])

# --- Base Classifiers (Shared across both models) ---
svc_base = CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'), cv=3)
lr_base = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
ridge_base = RidgeClassifier(alpha=1.0, random_state=42)
meta_learner = LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto')

clf_orig_stack = Pipeline([
    ('features', rich_features_A),
    ('clf', StackingClassifier(
        estimators=[('svc', svc_base), ('lr', lr_base), ('ridge', ridge_base)],
        final_estimator=meta_learner, cv=3, n_jobs=1
    ))
])

clf_clean_stack = Pipeline([
    ('features', rich_features_C),
    ('clf', StackingClassifier(
        estimators=[('svc', svc_base), ('lr', lr_base), ('ridge', ridge_base)],
        final_estimator=meta_learner, cv=3, n_jobs=1
    ))
])

# --- Double Stacking Classifier Wrapper ---
class DoubleStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf_orig, clf_clean):
        self.clf_orig = clf_orig
        self.clf_clean = clf_clean
        
    def fit(self, X, y):
        # Fit Model 1 on original text
        self.clf_orig.fit(X, y)
        # Fit Model 2 on clean text
        X_clean = X.apply(clean_text_v1)
        self.clf_clean.fit(X_clean, y)
        self.classes_ = self.clf_orig.classes_
        return self
        
    def predict(self, X):
        # Average decision functions of both models
        dec_orig = self.clf_orig.decision_function(X)
        X_clean = X.apply(clean_text_v1)
        dec_clean = self.clf_clean.decision_function(X_clean)
        dec_ensemble = dec_orig + dec_clean
        return self.classes_[dec_ensemble.argmax(axis=1)]

clf = DoubleStackingClassifier(clf_orig_stack, clf_clean_stack)
clf.fit(train_df["text"], y_train)
"""

new_markdown_cell_29 = r"""## 13. 성능 향상을 위한 접근법 (모델 빌드업 과정)

### 분석 및 Error Analysis

| 단계 | Accuracy | 방법 |
|------|----------|------|
| Baseline | 77.37% | Whitespace 토큰화 + CountVectorizer + MultinomialNB |
| Attempt 1 | 89.42% | TF-IDF + Soft Voting (LinearSVC + LogisticRegression) |
| Attempt 2 | 91.40% | Word(1~3-gram) + Char(3~5-gram) TF-IDF FeatureUnion + LinearSVC |
| **Attempt 3 (최종)** | **92.63%** | Double Stacking Ensemble (Original Stacking + Cleaned Stacking) |

**개선 포인트:**
1. **Double Stacking Ensemble (최종 개선)**:
   - **Model 1**: 고유 오탈자 패턴(contanctless 등)이 보존된 원본 텍스트 기반 Stacking
   - **Model 2**: 영어 축약어 복원, 오탈자 정규화, 특수기호 토큰화 처리가 적용된 전처리 텍스트 기반 Stacking
   - 두 Stacking 분류기의 결정 경계값(decision_function)을 앙상블하여 예측 노이즈를 서로 보정함으로써 기존 성능 상한선인 92.50%를 돌파하여 **92.63%**를 달성했습니다.
2. **Feature diversification**: word TF-IDF, char_wb TF-IDF, char TF-IDF, binary word TF-IDF를 FeatureUnion으로 다양하게 결합
3. **Stacking 구조**: 각 서브 모델은 CalibratedSVC, LogisticRegression, RidgeClassifier의 예측 결정을 메타 학습자(LinearSVC)가 결합
"""

# Modify the loaded cells
for cell in nb.get('cells', []):
    source_str = "".join(cell.get('source', []))
    if cell.get('cell_type') == 'markdown' and '## 13. 성능 향상을 위한 접근법' in source_str:
        cell['source'] = [line + '\n' for line in new_markdown_cell_29.split('\n')]
        print("Updated CELL 29 markdown!")
    elif cell.get('cell_type') == 'code' and '# Attempt 3 (최종 모델)' in source_str:
        cell['source'] = [line + '\n' for line in new_code_cell_35.split('\n')]
        print("Updated CELL 35 code!")

# Save the new notebook
with open(notebook_out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved new notebook to {notebook_out}")
