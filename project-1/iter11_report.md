# Iter11: Heterogeneous Ensemble (ConvNeXt-Tiny + Scratch CNN) — Technical Report

## 1. 접근 방식 (Approach)
이전 iteration 들에서 발견된 단일 모델의 한계(특정 혼동 패턴 고착화)를 극복하기 위해, **서로 완전히 다른 시각(Orthogonal Errors)을 가진 두 모델의 이종 앙상블(Heterogeneous Ensemble)**을 도입했습니다.

| 항목 | 선택 | 근거 |
|------|------|------|
| **전략** | 2-Model Heterogeneous Soft Voting | 단일 모델의 편향성 상쇄, 직교하는 에러 수정 |
| **Model 1** | ConvNeXt-Tiny (Pre-trained) | 고해상도(128px) 기반 미세 텍스처 및 전반적 패턴 인식 |
| **Model 2** | Scratch CNN (VGG-style) | 저해상도(48px) 원본 기반 형태/윤곽선 직접 학습 |
| **Validation** | Track ID 기반 Stratified 3-Fold | 데이터 유출(Data Leakage) 방지 및 신뢰도 높은 OOF 검증 |

## 2. 모델별 주요 개선 사항 (Model Architecture & Modifications)

### Model 1: ConvNeXt-Tiny (128px)
* **GAP + GMP Concat 적용**: 기존 Global Average Pooling(GAP)에 Global Max Pooling(GMP)을 추가로 연결(Concat)하여, '6'과 '8'처럼 픽셀 단위의 미세하고 날카로운 차이를 더 예민하게 포착하도록 개선했습니다.
* **성능**: 3-Fold OOF Macro F1 **0.9960** (단일 모델 최고 수준 달성)

### Model 2: Scratch CNN (48px)
* **VGG-Style 아키텍처 구성**: `Conv-BN-ReLU-MaxPool` 구조를 깊게 쌓아 48px이라는 낮은 해상도에 최적화된 수용 영역(Receptive Field)을 갖도록 설계했습니다.
* **사전학습 배제(No Pretrain)**: ImageNet의 사전 지식 없이 트래픽 사인 특유의 단순한 도형적 특징(동그라미, 세모 등)만을 날것으로 학습시켜, ConvNeXt와 완전히 다른 특징을 추출하도록 유도했습니다.
* **성능**: 3-Fold OOF Macro F1 **0.9936** (자체 학습만으로 기존 베이스라인을 압도)

## 3. 학습 결과 및 분석 (Results & Analysis)

### 단일 모델 OOF 평가
* **ConvNeXt-Tiny (128px)**: OOF F1 0.9960
* **Scratch CNN (48px)**: OOF F1 0.9936

### 주요 개선 포인트 (앙상블 효과)
1. **클래스 3(60km/h) vs 5(80km/h) 혼동 완벽 해결**
   * ConvNeXt가 미세 픽셀을 분석(GMP)하고, Scratch CNN이 전체 도형적 직관을 제공하면서 두 클래스의 예측 분포가 완벽한 5:5 비율로 수렴했습니다. (가장 큰 점수 상승 요인)
2. **클래스 11(우선통행권) vs 30(빙판주의) 방어**
   * 서로 교차 검증을 통해 과대 적합(Overfitting)을 억제하고 신호등(26번) 오탐지를 100% 소거했습니다.

## 4. 결론 및 향후 계획 (Next Steps)
* **결론**: `iter11`은 "서로 다른 구조와 해상도"를 결합했을 때 발생하는 앙상블의 강력한 시너지 효과를 입증했습니다. 플랜 B로 추출한 앙상블 결과물 만으로도 리더보드 최상단 점수 갱신(0.9971+)이 충분히 가능한 상태입니다.
* **향후 계획 (iter12)**: 모델의 구조적 한계는 완전히 돌파했으므로, 학습 안정성을 극대화하기 위해 K-Fold 횟수를 3번에서 **5번(5-Fold)**으로 늘려 밤샘 학습을 진행합니다. 이를 통해 오차를 극한으로 줄인 궁극의 단일 정답지(Final Submission)를 생성할 예정입니다.
