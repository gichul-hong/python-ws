# Iter20 작업 계획 (ResNet Custom Stem - 1등 조 비법 벤치마킹 & 0.9980+ 도전)

## 1. 실험 목적
- **1등 조(Macro F1 0.9979)의 핵심 전략 벤치마킹**: 표준 ResNet50의 7x7 Conv + MaxPool 다운샘플링 stem 구조가 GTSRB 표지판과 같은 소형 이미지(32x32~128x128)의 미세 고주파 특성을 초반부터 파괴하는 문제를 해결.
- **Custom Stem 개조**:
  - `conv1`: `7x7 (stride 2)` -> `3x3 (stride 1, padding 1)`
  - `maxpool`: `MaxPool2d(3, stride 2)` -> `nn.Identity()` (다운샘플링 전면 제거)
- **목표**: 공간 해상도를 최대로 유지한 상태에서 Full Data(26,010장) 5-Fold 학습을 통해 **0.9980+ 최고 기록 달성**.

## 2. 작업 내용
1. `ResNetWithFeatures` 클래스 내 Stem 레이어 커스텀 개조.
2. Full Data (26,010장) 100% 적용 (`MAX_FRAMES_PER_TRACK = 99999`).
3. 결과물 `resnet_stem_result.csv` 저장 및 최종 채점.
