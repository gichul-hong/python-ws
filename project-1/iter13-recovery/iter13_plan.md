# Iter13: Fallback & Recovery Plan — 1등 방어 전략

## 1. 작업 개요 (Objective)
Iter12의 OOF 가중치 최적화 앙상블 모델이 겉보기(OOF 점수)는 화려했으나, 내부적으로 과대적합(Overfitting) 리스크가 발견되었습니다. 유저의 냉철한 분석에 따라, **통계적으로 가장 안정적이고 밸런스가 좋았던 `Iter11 ConvNeXt-Tiny (3-Fold)` 단일 모델을 최종 제출안으로 확정**합니다.
어제 덮어쓰기 실수로 인해 소실된 해당 모델을 `iter13-recovery` 환경에서 완벽히 동일한 조건으로 재학습하여 복구합니다.

## 2. 왜 Iter11 ConvNeXt 단일 모델인가? (Rationale)
1. **가장 우수한 단일 OOF 점수**: `0.9960`이라는 단일 모델 최고 수치 달성.
2. **압도적인 안정성 (Proxy 오차 최소화)**: Test 데이터 분포와 비교했을 때 최대 오차(`max|diff|`)가 단 **8개**에 불과함. 이는 특정 클래스에서 뭉텅이로 오답을 내는 과대적합 현상이 없음을 증명함. (Iter12 앙상블은 11개로 튀었음)
3. **앙상블의 독(Poison) 배제**: 성능이 현저히 떨어지는 Scratch CNN에 높은 가중치(56%)를 부여하여 생기는 억지 점수 맞추기 리스크를 완전히 제거.

## 3. 세부 실행 계획 (Execution Plan)

### 단계 1: 격리된 복구 환경 세팅
* `iter13-recovery` 폴더를 생성하여 모든 작업을 기존 폴더들과 완벽히 격리.

### 단계 2: ConvNeXt-Tiny@128 3-Fold 롤백 및 학습
* `N_FOLDS = 3` 으로 원복.
* Batch Size (64), Epochs (20), Image Size (128) 등 Iter11 당시 최고의 성과를 냈던 설정값들을 정확히 복사하여 실행.

### 단계 3: 최종 산출물(csv, ipynb) 클린 생성
* 학습이 끝나면 생성되는 OOF 점수 및 Proxy 지표를 `iter13_report.md`에 정리.
* 최종 추론 결과물에서 불필요한 열을 모두 날리고 `id`와 `class` 열만 남겨 `DS2_challenge_team1_final.csv`로 저장.
* 이 모델을 돌리는데 쓰인 코드를 `DS2_challenge_team1_final.ipynb` 형태로 단일화하여 제출 완비.

## 4. 예상 완료 시간 및 결과
* **소요 시간**: 약 2.5시간 (현재 백그라운드 구동 중)
* **예상 결과**: OOF Macro F1 0.9960 내외 복구, Proxy `max|diff|` 한 자릿수(8 이하)의 가장 단단하고 방어력 높은 1위 굳히기 모델 완성.
