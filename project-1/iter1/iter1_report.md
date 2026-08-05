# Iter1: Baseline CNN for Traffic Sign Classification — Technical Report

## 1. 접근 방식 (Approach)

| 항목 | 선택 | 근거 |
|------|------|------|
| **Framework** | PyTorch 2.5.1 (CUDA 12.1) | 강의 skeleton 준수, GPU 가속 |
| **이미지 크기** | 30×30×3 | skeleton 명세, 경량 모델 가능 |
| **전처리** | float32 변환 + 0-1 정규화 | 안정적 학습 |
| **Split** | train/val 8:2, stratified | class 불균형 고려 |
| **Class weight** | `compute_class_weight('balanced')` | 소수 클래스 penalty 강화 |

## 2. 모델 구조 (Model Architecture)

```
Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout(0.25)  # 32 채널
Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout(0.25)  # 64 채널
Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout(0.30)  # 128 채널
Flatten
FC(→256)-BN-ReLU-Dropout(0.5)
FC(→128)-BN-ReLU-Dropout(0.5)
FC(→43)
```

- **파라미터 수**: 622,283
- **구조 선택 이유**: GTSRB 과제에서는 깊이보다 batch norm + dropout 조합이 작은 이미지에서 overfitting을 효과적으로 억제함. 3단 conv 블록으로 충분한 feature 추출이 가능하고, FC 2단으로 classifier 구성.

## 3. 학습 설정 (Training Config)

| 항목 | 값 |
|------|----|
| Optimizer | Adam (lr=1e-3) |
| Loss | CrossEntropyLoss (class weight 적용) |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=5) |
| Batch size | 256 |
| Epochs | 30 |
| Seed | 42 |

## 4. 결과 (Results)

### 학습 곡선

| Epoch | Train Loss | Train Acc | Train F1 | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|----------|---------|--------|
| 1 | 3.5872 | 0.0672 | 0.0500 | 3.1251 | 0.1190 | 0.0802 |
| 5 | 0.9059 | 0.6915 | 0.6919 | 0.3176 | 0.8787 | 0.9123 |
| 10 | 0.1178 | 0.9659 | 0.9683 | 0.0155 | 0.9925 | 0.9955 |
| 15 | 0.0668 | 0.9809 | 0.9799 | 0.0074 | 0.9969 | 0.9982 |
| 20 | 0.0431 | 0.9861 | 0.9857 | 0.0044 | 0.9985 | **0.9992** |
| 25 | 0.0341 | 0.9904 | 0.9897 | 0.0077 | 0.9967 | 0.9973 |
| 30 | 0.0158 | 0.9959 | 0.9960 | 0.0042 | 0.9983 | 0.9980 |

### Best Model
- **Best Val Accuracy**: 99.87% (epoch 19~20 부근)
- **Best Val Macro F1**: 0.9992 (epoch 20)
- **데이터**: Train 20,808장 / Val 5,202장 (class min=180, max=1260, 평균 605)

### Test 추론
- 총 8,670장 예측, 43개 클래스 모두 커버
- 클래스별 예측 분포: min=46, max=439

## 5. 추가 개선 방향 (Future Work)

### 5.1 데이터 증강 (Data Augmentation) — 높은 우선순위
```
현재는 augmentation 없음. 30×30 이미지에 적합한 augmentation:
- RandomRotation(±15°) → 회전 불변성
- ColorJitter(brightness=0.2, contrast=0.2) → 조명 변화 대비
- RandomAffine(translate=(0.1, 0.1)) → 위치 변동
```
GTSRB는 실외 촬영 이미지라 조명/각도 차이가 크므로 augmentation 효과가 클 것으로 예상.

### 5.2 입력 해상도 증가
- 현재 30×30 → 48×48 또는 64×64로 키우면 미세한 표지판 특징 포착 가능
- 모델 용량이 작아(622K) 해상도 증가에 따른 오버헤드 적음
- 단, 학습 시간 약 2~3배 증가 예상

### 5.3 모델 구조 실험
- ResNet-18 pretrained weight 사용 (transfer learning)
- 더 큰 해상도에서만 효과적 (224×224 필요)
- 간단한 backbone으로도 99%+ 달성 가능하므로 필수는 아님

### 5.4 Ensemble
- 서로 다른 seed로 3~5개 학습 후 soft voting
- 단일 모델 대비 0.2~0.5% 추가 개선 가능

### 5.5 Hyperparameter Tuning
- 더 긴 epoch (50~100) + early stopping
- Learning rate warmup
- Batch size 실험 (128, 512)

### 5.6 Test-time augmentation (TTA)
- 추론 시 augmentation 적용 후 평균 예측 → 안정성 확보
- 비용이 크므로 ensemble이 어려울 때 대안

## 6. 참고 사항 (Notes for Other Agents)

- **절대경로**: `DATA_ROOT`가 하드코딩 되어 있음. 환경에 따라 수정 필요.
- **model path**: iter1/best_model.pth 에 학습 완료된 가중치 저장됨.
- **제출 형식**: `result.csv` → `id,class` (헤더 있음, 인덱스 없음, int).
- **conda env**: `gpu-torch`, Python 3.12, PyTorch 2.5.1+cu121.
- **GPU**: NVIDIA RTX 2060 (6GB VRAM), batch=256 기준 약 2GB 사용.
- **학습 시간**: 30 epochs 약 2~3분 (RTX 2060 기준).