# Iter12: 5-Fold Heterogeneous Ensemble & OOF Optimization — Action Plan

## 1. 작업 개요 (Objective)
Iter11에서 검증된 두 개의 이종 모델(ConvNeXt, Scratch CNN)의 시너지를 극대화하고, 제출 기한 전 마지막 야간(Overnight) 자원을 활용하여 수학적 한계치까지 점수를 끌어올리기 위한 최종 병기(Final Pipeline)를 구축합니다.

## 2. 세부 실행 계획 (Execution Plan)

### 단계 1: 5-Fold 교차 검증 (Cross-Validation) 확장
* **기존**: 시간적 제약으로 3-Fold 학습 진행
* **변경**: 모델별 전체 데이터셋을 5등분하여 5번 교차 검증하는 **5-Fold (N_FOLDS=5)** 로 확장.
* **기대 효과**: 단일 모델의 OOF 신뢰도 증가 및 5개 모델 앙상블을 통한 일반화(Generalization) 성능 극대화 (특히 미세한 엣지 케이스 방어력 향상).

### 단계 2: 파일 덮어쓰기 방지 (격리된 파이프라인 구축)
* Iter11에서 발생한 모델 저장 파일 덮어쓰기 오류를 방지하기 위해 파일 네이밍 룰(Naming Rule)을 철저히 분리합니다.
* **ConvNeXt**: `conv_best_model_fold{f}.pth`, `conv_oof_probs.npy`, `conv_test_probs.npy`
* **Scratch CNN**: `scratch_best_model_fold{f}.pth`, `scratch_oof_probs.npy`, `scratch_test_probs.npy`

### 단계 3: SciPy 기반 OOF 앙상블 가중치 자동 최적화
* **기존**: 직관에 의존한 고정 가중치(ConvNeXt 0.65 : Scratch 0.35) 사용.
* **변경**: `scipy.optimize.minimize` 알고리즘을 도입하여, 두 모델의 OOF 확률값을 가장 완벽하게 섞어 **Macro F1을 극대화하는 최적의 수학적 가중치(Optimal Weight)**를 역산합니다.
* **동작 방식**: 
  1. 두 모델의 OOF 확률(26,010장) 로드
  2. $w \times \text{ConvNeXt} + (1-w) \times \text{Scratch}$ 조건 하에서 Macro F1을 최대화하는 $w$ 스캐닝
  3. 찾아낸 최적 가중치 $w$를 실제 Test Inference에 적용

## 3. 자동화 워크플로우 (Automated Workflow)
아래 과정이 `iter12-gemini/run_all.ps1` 스크립트에 의해 사람의 개입 없이 야간에 순차적으로 자동 실행됩니다.
1. `train_convnext.py` (약 4.5시간)
2. `train_scratch.py` (약 2.5시간)
3. `ensemble_5fold.py` (최적 가중치 계산 및 최종 `DS2_challenge_team1_final.csv` 생성)

## 4. 예상 결과 (Expected Outcome)
* OOF Macro F1 점수의 추가 상승 (0.9960 → 0.9965 이상 예상)
* 클래스 간 혼동(예: 60km/h vs 80km/h, 우선통행권 vs 빙판주의)에 대한 기계적이고 완벽한 방어
* 내일 아침, 아무런 추가 작업 없이 **대회 1등 점수(0.9971) 갱신을 위한 최종 제출 파일 완성**
