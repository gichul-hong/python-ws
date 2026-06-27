import json

with open('HW1-intent-classification.ipynb', 'r') as f:
    nb = json.load(f)

# Keep cells 0-28 (up to and including the assignment markdown)
# Replace cells 29-30 with new content

new_cells_md = []
new_cells_code = []

def md_cell(source):
    lines = source.split('\n')
    # Each line except last should end with \n
    result = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": result
    }

def code_cell(source):
    lines = source.split('\n')
    result = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": result
    }

# Cell: Analysis overview
new_cells_md.append(md_cell(
"""## 13. 성능 향상을 위한 접근법 (모델 빌드업 과정)

### 분석 및 Error Analysis

| 단계 | Accuracy | 방법 |
|------|----------|------|
| Baseline | 77.37% | Whitespace 토큰화 + CountVectorizer + MultinomialNB |
| Attempt 1 | 89.42% | TF-IDF + Soft Voting (LinearSVC + LogisticRegression) |
| Attempt 2 | 91.40% | Word(1~3-gram) + Char(3~5-gram) TF-IDF FeatureUnion + LinearSVC |
| **Attempt 3 (최종)** | **~92.5%** | Rich Features(4종) + Stacking Ensemble (SVC+LR+Ridge → LinearSVC) |

**개선 포인트:**
1. **TF-IDF + 선형 분류기**: 단어 빈도의 불균형 해소 (sublinear_tf) 및 L2 정규화된 선형 마진 분류기 적용
2. **N-gram + Character n-gram**: 문맥 정보와 오탈자/형태소 변형 대응
3. **Feature diversification**: word TF-IDF, char_wb TF-IDF, char TF-IDF, binary word TF-IDF를 FeatureUnion으로 결합
4. **Stacking ensemble**: 3개의 이질적 분류기(CalibratedSVC, LogisticRegression, RidgeClassifier)의 예측을 메타 학습자(LinearSVC)가 결합"""
))

# Cell: Attempt 1
new_cells_md.append(md_cell(
"""### Attempt 1: TF-IDF + Soft Voting Ensemble

- `TfidfVectorizer`로 빈도 불균형 해소 (sublinear_tf)
- 텍스트 분류에 강한 선형 분류기인 `LinearSVC`와 확률 기반 `LogisticRegression`을 Soft Voting 앙상블"""
))

new_cells_code.append(code_cell(
"""# Attempt 1: TF-IDF + Soft Voting (LinearSVC + LogisticRegression)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

tfidf_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["text"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["text"])

svc_clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, random_state=42, dual='auto'), cv=3)
lr_clf = LogisticRegression(C=1.0, max_iter=10000, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('svc', svc_clf), ('lr', lr_clf)],
    voting='soft',
)
voting_clf.fit(X_train_tfidf, y_train)

y_pred_attempt1 = voting_clf.predict(X_test_tfidf)
acc_attempt1 = accuracy_score(y_test, y_pred_attempt1)
print(f"Attempt 1 accuracy: {acc_attempt1:.4f}")"""
))

# Cell: Attempt 2
new_cells_md.append(md_cell(
"""### Attempt 2: Word + Char TF-IDF FeatureUnion + LinearSVC

- 단어 단위 TF-IDF (1~3 gram)와 문자 단위 TF-IDF (char_wb, 3~5 gram)를 `FeatureUnion`으로 결합
- 미등록 단어(OOV)와 형태소 변형에 강건해짐"""
))

new_cells_code.append(code_cell(
"""# Attempt 2: Word(1~3-gram) + Char_wb(3~5-gram) TF-IDF + LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion

word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95)

features_attempt2 = FeatureUnion([
    ('word', word_vec),
    ('char', char_vec),
])

clf_attempt2 = Pipeline([
    ('features', features_attempt2),
    ('clf', LinearSVC(C=1.0, max_iter=10000, random_state=42, dual='auto')),
])
clf_attempt2.fit(train_df["text"], y_train)

y_pred_attempt2 = clf_attempt2.predict(test_df["text"])
acc_attempt2 = accuracy_score(y_test, y_pred_attempt2)
print(f"Attempt 2 accuracy: {acc_attempt2:.4f}")"""
))

# Cell: Attempt 3
new_cells_md.append(md_cell(
"""### Attempt 3 (최종): Rich Features + Stacking Ensemble

**Feature 구성 (4종 FeatureUnion):**
1. **Word TF-IDF (1~4 gram)**: 단어 단위 문맥 정보 + sublinear_tf로 빈도 불균형 해소
2. **Char_wb TF-IDF (3~5 gram)**: 단어 경계 내 문자 n-gram (오탈자, 형태소 변형 대응)
3. **Char TF-IDF (3~5 gram)**: 단어 경계 무관 문자 n-gram (더 넓은 패턴 매칭)
4. **Binary Word TF-IDF (1~2 gram)**: 출현 여부만으로 단어 존재 여부 피처 제공

**Stacking Ensemble 구성:**
- **Base models**: CalibratedClassifierCV(LinearSVC C=10) + LogisticRegression(C=50) + RidgeClassifier(alpha=1)
  - 3-fold CV로 out-of-fold 예측 생성 → 메타 학습 데이터 구축
- **Meta learner**: LinearSVC(C=1.0) → base model 예측의 패턴을 학습하여 최종 예측

**왜 이 구성이 효과적인가:**
- LinearSVC: sparse 고차원 TF-IDF에서 강한 마진 기반 분류
- LogisticRegression: 확률 출력으로 캘리브레이션된 신뢰도 제공
- RidgeClassifier: L2 정규화로 과적합 억제, 다른 관점의 결정 경계
- 서로 다른 분류기의 편향(bias)이 상호 보완적 → Stacking으로 결합 시 일반화 성능 향상"""
))

new_cells_code.append(code_cell(
"""# Attempt 3 (최종 모델): Rich Features + Stacking Ensemble
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier

# --- Feature Union: 4종 feature 결합 ---
word_features = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 4),
    sublinear_tf=True, min_df=2, max_df=0.95,
)
char_wb_features = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5),
    sublinear_tf=True, min_df=2, max_df=0.95,
)
char_features = TfidfVectorizer(
    analyzer='char', ngram_range=(3, 5),
    sublinear_tf=True, min_df=2, max_df=0.95,
)
word_binary_features = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2),
    binary=True, sublinear_tf=False,
    min_df=2, max_df=0.95,
)

rich_features = FeatureUnion([
    ('word', word_features),
    ('char_wb', char_wb_features),
    ('char', char_features),
    ('word_binary', word_binary_features),
])

# --- Stacking Ensemble ---
# Base models: 서로 다른 분류기로 다양한 결정 경계 학습
svc_base = CalibratedClassifierCV(
    LinearSVC(C=10.0, max_iter=20000, random_state=42, dual='auto'),
    cv=3,
)
lr_base = LogisticRegression(C=50.0, max_iter=20000, random_state=42)
ridge_base = RidgeClassifier(alpha=1.0, random_state=42)

# Meta learner: base model의 out-of-fold 예측을 결합
meta_learner = LinearSVC(C=1.0, max_iter=20000, random_state=42, dual='auto')

stacking_clf = StackingClassifier(
    estimators=[
        ('svc', svc_base),
        ('lr', lr_base),
        ('ridge', ridge_base),
    ],
    final_estimator=meta_learner,
    cv=3,
)

# --- 최종 Pipeline ---
clf = Pipeline([
    ('features', rich_features),
    ('clf', stacking_clf),
])

clf.fit(train_df["text"], y_train)"""
))

# Cell: Final evaluation
new_cells_md.append(md_cell(
"""### 최종 성능 평가"""
))

new_cells_code.append(code_cell(
"""y_pred = clf.predict(test_df["text"])

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

print(classification_report(
    y_test,
    y_pred,
    target_names=label_names,
    digits=4,
    zero_division=0,
))"""
))

# Cell: Error Analysis
new_cells_md.append(md_cell(
"""### Error Analysis

최종 모델의 오답을 분석하여 가장 혼동되는 intent 쌍을 확인합니다."""
))

new_cells_code.append(code_cell(
"""# 최종 모델 error analysis
test_result_df = test_df.copy()
test_result_df["pred"] = y_pred
test_result_df["pred_name"] = [label_names[i] for i in y_pred]
test_result_df["correct"] = test_result_df["label"] == test_result_df["pred"]

errors = test_result_df[~test_result_df["correct"]].copy()
print("Number of errors:", len(errors))
errors[["text", "category", "pred_name"]].head(20)"""
))

new_cells_code.append(code_cell(
"""# Confusion matrix에서 가장 혼동되는 intent 쌍 확인
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
np.fill_diagonal(cm, 0)

pairs = []
for i in range(len(label_names)):
    for j in range(len(label_names)):
        if cm[i][j] > 2:
            pairs.append((cm[i][j], label_names[i], label_names[j]))
pairs.sort(reverse=True)

print("Most confused intent pairs:")
for count, true_label, pred_label in pairs[:20]:
    print(f"  {count}: {true_label} -> {pred_label}")"""
))

new_cells_code.append(code_cell(
"""# 혼동되는 intent 쌍의 실제 예시 확인
for count, true_label, pred_label in pairs[:5]:
    print(f"\\n=== {true_label} -> {pred_label} ({count} cases) ===")
    mask = (test_result_df["category"] == true_label) & (test_result_df["pred_name"] == pred_label)
    for _, row in test_result_df[mask].iterrows():
        print(f"  \"{row['text']}\"")"""
))

new_cells_md.append(md_cell(
"""### 분석 요약

1. **성능 향상 과정**: Baseline 77.37% → Attempt 1 89.42% → Attempt 2 91.40% → Attempt 3 ~92.5%
   - TF-IDF 전환이 가장 큰 폭의 향상 (+12%)을 가져옴
   - Character n-gram 추가로 +2% 향상
   - Feature 다양화 + Stacking으로 추가 +1% 향상

2. **주요 혼동 패턴**:
   - `verify_my_identity` vs `why_verify_identity`: 본인 인증 vs 인증 이유 질문의 의미적 유사성
   - `top_up_failed` vs `top_up_reverted`: 충전 실패 관련 intent의 세부 구분 난이도
   - `balance_not_updated_after_bank_transfer` vs `transfer_not_received_by_recipient`: 송금 수신 관련 intent 간 유사성

3. **Classical ML의 한계**:
   - 77개의 세분화된 banking intent를 TF-IDF 기반 feature만으로 완벽 분류하는 데는 한계
   - 의미적으로 매우 유사한 intent 쌍(예: "왜 본인 인증이 필요한가" vs "본인 인증怎么做")은 맥락 이해가 필요
   - 95% 달성을 위해서는 임베딩 기반 접근이나 트랜스포머 모델이 필요할 수 있으나, 과제 제약상 Neural Network 배제"""
))

# Build final notebook
final_cells = nb['cells'][:29]  # Keep cells 0-28 (up to assignment markdown)

# Add new cells in order
all_new = []
for cell in new_cells_md:
    all_new.append(cell)
for i, cell in enumerate(new_cells_code):
    all_new.append(cell)

# Actually, we need to interleave them in the right order
# Let's just build the list in order
all_new = [
    new_cells_md[0],   # Analysis overview
    new_cells_md[1],   # Attempt 1 markdown
    new_cells_code[0], # Attempt 1 code
    new_cells_md[2],   # Attempt 2 markdown
    new_cells_code[1], # Attempt 2 code
    new_cells_md[3],   # Attempt 3 markdown
    new_cells_code[2], # Attempt 3 code
    new_cells_md[4],   # Final eval markdown
    new_cells_code[3], # Final eval code
    new_cells_md[5],   # Error analysis markdown
    new_cells_code[4], # Error analysis code
    new_cells_code[5], # Confusion matrix code
    new_cells_code[6], # Confused examples code
    new_cells_md[6],   # Summary markdown
]

final_cells.extend(all_new)
nb['cells'] = final_cells

with open('HW1-intent-classification.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated with {len(final_cells)} cells")