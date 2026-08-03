# Iter1: Baseline CNN 구현 정리

## 파일 구성
- `trafficsign_cnn.ipynb` — 전체 파이프라인 노트북 (학습 + 추론)
- `best_model.pth` — 학습 완료된 모델 가중치 (Best Val Acc 기준)

## 구현 개요
German Traffic Sign(43 클래스) 분류를 위한 baseline CNN.
스켈레톤(`cnn-trafficsign-torch-skeleton.ipynb`)의 데이터 로딩 방식을 유지하면서 전체 파이프라인을 완성.

## 파이프라인

### 1. 데이터 로딩 및 전처리
- 경로: `dataset/data 2/Train/{0~42}/`
- cv2로 로딩 → PIL로 30×30 리사이즈
- float32 변환, 0-1 정규화
- Train/Val 8:2 **stratified** split (seed=42)
- 총 26,010장 → Train 20,808 / Val 5,202

### 2. Class Imbalance 대응
- 클래스별 이미지 수: min 180 ~ max 1,260 (불균형)
- `compute_class_weight('balanced')` → CrossEntropyLoss weight로 적용
- weight 범위: 0.480 ~ 3.360

### 3. 모델 구조 (622,283 파라미터)
```
[Conv Block 1] Conv3x3(3→32) - BN - ReLU - Conv3x3(32→32) - BN - ReLU - MaxPool2 - Dropout(0.25)
[Conv Block 2] Conv3x3(32→64) - BN - ReLU - Conv3x3(64→64) - BN - ReLU - MaxPool2 - Dropout(0.25)
[Conv Block 3] Conv3x3(64→128) - BN - ReLU - Conv3x3(128→128) - BN - ReLU - MaxPool2 - Dropout(0.3)
[Classifier]   Flatten - FC(1152→256) - BN - ReLU - Dropout(0.5)
               - FC(256→128) - BN - ReLU - Dropout(0.5) - FC(128→43)
```
- FC 입력 크기는 dummy tensor로 동적 계산 (`_get_conv_output`)
- **주의**: forward에서 `x.reshape()` 사용 (`.view()`는 non-contiguous 에러 발생)

### 4. 학습 설정
| 항목 | 값 |
|------|----|
| Optimizer | Adam (lr=1e-3) |
| Loss | CrossEntropyLoss + class weight |
| Scheduler | ReduceLROnPlateau (max, factor=0.5, patience=5) |
| Batch / Epochs | 256 / 30 |
| Best model 기준 | Val Accuracy |

### 5. 추론 및 결과 저장
- `result.csv`의 id 순서 그대로 Test 이미지(8,670장) 로딩
- 읽기 실패 시 zero 이미지로 대체 (순서 유지)
- 예측값을 `class` 열에 int로 기록, `index=False`로 저장

## 결과
| 지표 | 값 |
|------|----|
| Best Val Accuracy | **99.87%** |
| Best Val Macro F1 | **0.9992** (epoch 20) |
| 학습 시간 | 약 2~3분 (RTX 2060, 30 epochs) |

## 실행 방법
```
conda activate gpu-torch
jupyter notebook trafficsign_cnn.ipynb  # 전체 셀 순차 실행
```
- 경로가 절대경로(`C:\hong\python-ws\project-1\...`)로 하드코딩되어 있으므로 환경에 맞게 수정 필요

## 알려진 이슈 / 개선 포인트
- Data augmentation 미적용 → iter2에서 적용 권장
- 입력 해상도 30×30 → 48×48 이상으로 확대 시 개선 여지
- 단일 모델 → ensemble / TTA 시 추가 개선 가능
- 상세 개선 방향은 `../iter1_report.md` 참조