# Iter17 작업 계획 (SupCon 도입)

## 1. 실험 목적
- 데이터 정제(Pruning)로 인해 4,340장으로 줄어든 환경(`iter16`)에서, 랜덤 가중치 모델(Scratch CNN)은 특징을 추출하지 못하고 붕괴(40% 정확도)했습니다.
- 이를 해결하기 위해, 같은 표지판(Track)의 이미지들을 특징 공간에서 묶어주는 **지도 대조학습(Supervised Contrastive Learning)** 기법을 도입하여 Scratch 모델의 학습력을 강제로 끌어올릴 수 있는지 검증합니다.

## 2. 작업 내용
1. **모델 구조 변경**: 
   - ConvNeXt 및 Scratch CNN이 최종 분류 결과(Logits)뿐만 아니라, 512차원의 특징 벡터(Feature Vector)를 함께 반환하도록 수정.
2. **손실 함수(Loss) 추가**:
   - `ContrastiveTrackLoss` 구현: 같은 `track_id`를 가진 이미지 쌍은 당기고(Pull), 다른 쌍은 밀어내는(Push) Loss 추가.
   - 최종 Loss = `CrossEntropyLoss + 0.5 * ContrastiveTrackLoss`
3. **5-Fold 교차 검증**:
   - ConvNeXt (Pretrained) 및 Scratch CNN (Random Init) 모두 5-Fold 학습 후 OOF 앙상블 진행.

## 3. 기대 효과
- Scratch CNN의 정답률이 40%에서 90% 이상으로 복구되는지 확인.
- 만약 복구되지 않는다면, **"데이터 기근 상황에서는 아무리 최신 기법(SupCon)을 사용해도 전이 학습(Pretrained Weights)의 힘을 이길 수 없다"**는 핵심 결론 도출.
