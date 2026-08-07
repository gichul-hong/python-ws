# Iter18 작업 계획 (최종 4-Model 앙상블)

## 1. 실험 목적
- `iter17`의 실패(Scratch 모델의 붕괴)를 거울삼아, 소량의 데이터 셋(4,340장)에서는 Scratch 학습을 포기하고 **100% Pretrained 모델로만 구성된 엘리트 앙상블**을 구축합니다.
- 데이터 정제(Pruning)와 대조학습(SupCon)의 시너지 위에, 4개의 강력한 이기종 모델을 결합하여 최종 SOTA 점수(최고의 Test F1 Score)를 갱신하는 것이 목표입니다.

## 2. 작업 내용
1. **모델 라인업 개편**:
   - 기존의 ConvNeXt와 Scratch CNN 조합에서 Scratch를 퇴출.
   - 4개의 강력한 Pretrained 모델 라인업 확정: **ConvNeXt, ResNet50, EfficientNet-B0, MobileNetV3-Small**
2. **코드 구조화 및 디버깅**:
   - 각 모델별 학습 스크립트 분리 (`train_resnet.py` 등) 및 결과 파일명 독립 (`resnet_oof_probs.npy`).
   - Colab 구동을 위한 경로 버그(`__file__`, `DATA_ROOT`) 완벽 수정 및 통합 노트북 `iter18_colab.ipynb` 제공.
3. **학습 및 평가**:
   - 각 모델마다 Pruned Dataset(4,340장) + SupCon Loss를 적용하여 5-Fold 학습.
   - `ensemble_4models.py`를 통해 4개 모델의 OOF 확률값을 최적화하여 앙상블 가중치 탐색 및 최종 Test CSV 도출.

## 3. 기대 효과
- 모델 간의 구조적 차이(Uncorrelated errors)를 통한 강력한 앙상블 효과 발현.
- 최소한의 데이터(4,340장)로 기존 풀 데이터(26,010장) 앙상블 점수를 뛰어넘는 최고의 효율 달성 (학습 시간 최소화 + 정확도 극대화).
