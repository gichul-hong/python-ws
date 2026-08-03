# Iter2 (fable): Improved CNN for Traffic Sign Classification — Technical Report

## 1. 개요

iter1 (DeepSeek 구현, Val Macro F1 0.9987) 을 기반으로 macro F1을 개선한 iteration.

- **결과: Best Val Macro F1 = 0.9999, Val Acc = 0.9998** (iter1 대비 +0.0012)
- 코드: `iter2-fable/train.py`
- 제출 파일: `iter2-fable/result.csv` (8,670건, 43개 클래스 전부 커버, `id,class` 헤더 형식)
- 모델 가중치: `iter2-fable/best_model.pth`

## 2. iter1 대비 변경점

| 항목 | iter1 | iter2-fable | 근거 |
|------|-------|-------------|------|
| 입력 해상도 | 30×30 | **48×48** | 미세한 표지판 특징 포착 |
| 데이터 증강 | 없음 | **RandomAffine(±12°, translate 0.1, scale 0.85–1.15, shear 5) + ColorJitter(brightness/contrast 0.3, saturation 0.2)** | 실외 촬영 조명/각도 변화 대응 |
| Conv 블록 | 3단 (32/64/128) | **4단 (32/64/128/256)** | 해상도 증가에 맞춘 receptive field 확장 |
| 파라미터 수 | 622K | 2.38M | |
| Optimizer | Adam | **AdamW (wd=1e-4)** | 정규화 강화 |
| Loss | CE + class weight | CE + class weight + **label smoothing 0.05** | 과신 방지, 일반화 |
| Scheduler | ReduceLROnPlateau | **CosineAnnealingLR (eta_min=1e-5)** | 후반 미세 수렴 |
| Best 모델 기준 | Val Accuracy | **Val Macro F1** | 평가 지표와 직접 정렬 |
| 추론 | 단일 forward | **TTA 4종 soft voting** (identity, ±8° 회전, affine scale 1.1) | 예측 안정성 |
| Epochs / Batch | 30 / 256 | 50 / 128 | 증강으로 인한 수렴 지연 보상 |

## 3. 모델 구조

```
[conv_block × 4]  각 블록: Conv-BN-ReLU ×2 → MaxPool → Dropout2d
  3→32   (drop 0.2)   48→24
  32→64  (drop 0.25)  24→12
  64→128 (drop 0.3)   12→6
  128→256(drop 0.3)   6→3
Flatten → FC(2304→512)-BN-ReLU-Dropout(0.5) → FC(512→43)
```

## 4. 학습 곡선

| Epoch | Train Loss | Train Acc | Val Acc | Val Macro F1 |
|-------|-----------|-----------|---------|--------------|
| 1  | 3.5468 | 0.0834 | 0.2086 | 0.1575 |
| 5  | 1.3262 | 0.7587 | 0.9239 | 0.9341 |
| 10 | 0.8614 | 0.9599 | 0.9948 | 0.9964 |
| 15 | 0.7743 | 0.9852 | 0.9985 | 0.9987 |
| 20 | 0.7403 | 0.9933 | 0.9998 | **0.9999** |
| 30 | 0.7118 | 0.9963 | 0.9998 | 0.9999 |
| 40 | 0.6985 | 0.9981 | 0.9998 | 0.9999 |
| 50 | 0.6946 | 0.9987 | 0.9998 | 0.9999 |

- Epoch 20에서 iter1의 best(0.9987)를 넘어섰고 이후 0.9999로 안정 유지
- Train loss가 0.69 부근에서 수렴하는 것은 label smoothing에 의한 floor

## 5. 학습 설정

| 항목 | 값 |
|------|----|
| Split | train/val 8:2, stratified (20,808 / 5,202) |
| Class weight | `compute_class_weight('balanced')` |
| LR | 1e-3 (cosine → 1e-5) |
| Batch size | 128 (val/test 256) |
| Seed | 42 |
| 학습 시간 | 50 epochs 약 8분 (RTX 2060, batch=128) |
| 환경 | conda `gpu-torch`, Python 3.12, PyTorch 2.5.1+cu121 |

## 6. Test 예측 분포 검증 (`check_dist.py`)

P0.1 강의자료 10p의 test 라벨 분포 차트(60/210/420 3단계)는 **train 클래스 수 ÷ 3**과 정확히 일치하고 총합도 8,670으로 test 이미지 수와 같다. 이를 기대 분포로 삼아 `result.csv` 예측 분포와 비교했다.

| 지표 | 값 |
|------|----|
| 총합 | 예측 8,670 = 기대 8,670 |
| Pearson 상관 | 0.9992 |
| 절대 오차 합 | 184 (전체의 2.12%) |
| Chi-square | 14.5 (df=42 대비 매우 작음 → 분포 일치) |
| diff=0 클래스 | 43개 중 12개 |

**주요 편차 클래스**
- 과소 예측: class 12 (−23), class 38 (−15)
- 과대 예측: class 20 (+14), class 11 (+14), class 34 (+13), class 32 (+11)

**해석**
- 예측 분포가 기대 분포와 전반적으로 잘 일치하며 특정 클래스 붕괴는 없음.
- 오분류 1건은 diff 두 칸(−1/+1)에 기여하므로 절대 오차 합 184 → **최소 92건(≈1.06%) 오분류 존재**라는 하한이 나옴. 즉 실제 test 정확도 상한은 약 **98.94%**로 val(0.9998)보다 낮을 것으로 추정.
- 혼동 방향 추정: 12(priority road)↔11(right-of-way), 38(keep right)↔34(turn left ahead)/20(dangerous curve right) 등 시각적으로 유사한 표지판 간 혼동 가능성이 높음 → iter3에서 ensemble + 해당 클래스 집중 증강으로 대응.

## 7. 추가 개선 방향

1. **Ensemble**: seed 3~5개로 학습 후 soft voting — val이 사실상 포화(오분류 ~1건)라 test 일반화 확보 목적
2. **Pretrained backbone**: ResNet-18/EfficientNet transfer learning (해상도 96+ 필요)
3. **Mixup/CutMix**: 추가 정규화 — 현 수준에서는 효과 제한적일 수 있음
4. **전체 데이터 재학습**: val split 없이 26,010장 전부로 최종 모델 학습 후 제출

## 8. 참고 사항

- `DATA_ROOT` 하드코딩 (`C:\hong\python-ws\project-1\dataset\data 2`) — 환경에 따라 수정 필요
- test id 순서는 상위 `result.csv` 템플릿을 그대로 따르므로 row 순서 변경 없음
- 제출 형식 규칙 준수: integer class, 헤더 `id,class`, index 없음
