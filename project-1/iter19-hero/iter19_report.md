# Iter19 결과 보고서 (Hero Model - 단일 ConvNeXt SOTA 경신)

## 1. 실험 결과
- **Test Accuracy**: **99.60% (0.995963)**
- **Test Macro F1**: **0.9942 (0.994226)**
- **OOF Accuracy**: 99.52%
- **OOF Macro F1**: 0.9932
- **Distribution Proxy Diff**: 62 (역대 최소 수치)

## 2. 결과 분석
1. **SOTA 경신**: 기존 최고 기록이었던 `iter3` (Acc 99.49%) 및 베이스라인 `iter6` (Macro F1 0.9892)를 압도적인 차이로 뛰어넘어 **99.60%의 신기록**을 달성했습니다.
2. **단일 영웅(Hero) 모델의 위력**: 불필요한 경량 모델(MobileNet, EfficientNet) 앙상블을 제거하고 가장 강력한 ConvNeXt 1개에 자원을 집중한 전략이 완벽하게 적중했습니다.
3. **Full Data + SupCon 시너지**: 26,010장 전체 데이터셋과 대조학습(SupCon)이 결합하면서 클래스간 경계 구분이 극대화되었습니다.

## 3. 결론 및 향후 계획
- `iter19`를 통해 단일 모델만으로도 99.60%라는 압도적인 성능을 보증받았습니다.
- 현재 **`iter20-resnet-custom-stem`** (1등 조 비법: ResNet Stem 3x3 s1 및 Downsampling 제거) 학습이 즉시 착수되었으며, 0.9980+ 최고 고지 점령을 시도 중입니다.
