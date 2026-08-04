# Iter8 (deepseek): Track-aware Split + Heterogeneous Ensemble — Technical Report

## 1. 결론 요약

**제출 비권고.** track-aware split은 기계적으로 정상 동작했으나, 새로 도입한 도메인 강건성 증강이 삼각형 경고 표지판의 변별력을 훼손해 iter6보다 성능이 낮을 것으로 추정됩니다.

| 지표 | iter6 | **iter8** |
|------|-------|-----------|
| 앙상블 val Macro F1 | 0.9996 (누수 val) | 0.9997 (track-aware val) |
| 실제 test Macro F1 | **0.9892** | 미제출 (추정 ~0.985) |
| 분포 오차 sum\|diff\| | 78 | **88** |
| 분포 오차 max\|diff\| | 12 | **22** |

## 2. 개별 모델 성적 (track-aware val)

| 모델 | 파라미터 | best val F1 | TTA solo F1 |
|------|----------|-------------|-------------|
| **ConvNeXt-Tiny@96** | 28M | **0.9969** | **0.9964** |
| ResNet-50@96 | 25.6M | 0.9930 | 0.9904 |
| EfficientNet-B0@128 | 5.3M | 0.9827 | 0.9839 |
| 3-모델 앙상블 | — | — | 0.9997 (Acc 0.9994) |

**ConvNeXt-Tiny가 최강 아키텍처**로 확인됨 (다음 iteration에서 중심 모델로 사용 권장).

## 3. Track-aware split 검증 (성공)

- 파일명 `{class}_{track}_{frame}.png` 파싱 → **868 트랙**, 트랙당 정확히 30프레임 (GTSRB 구조 일치, 파싱 실패 0건)
- Train 21,150장(706 트랙) / Val 4,860장(162 트랙), **트랙 중복 0**
- iter6의 random split은 같은 표지판 30프레임 중 약 24장을 train, 6장을 val에 배치 → val 0.9999의 원인이 확정됨

## 4. 실패 분석: 증강 변경(D)의 역효과

### 4.1 관측된 오분류
iter6 대비 예측 변경 65건(0.75%) 중 지배적 패턴:

| 변경 | 건수 |
|------|------|
| **11 → 30** | **15** |
| 26 → 12 | 10 |
| 9 → 17, 18 → 21 | 각 3 |

클래스별 분포 이탈:

| 클래스 | 의미 | 기대 | iter6 | iter8 |
|--------|------|------|-------|-------|
| 11 | 우선통행권 (삼각형) | 420 | 415 (-5) | **398 (-22)** |
| 30 | 빙판 주의 (삼각형) | 60 | 65 (+5) | **82 (+22)** |

### 4.2 원인
- 클래스 11과 30은 **동일한 빨간 삼각형**이고 내부 픽토그램만 다름
- 신규 증강 `GaussianBlur(5, sigma≤2.0, p=0.35)` + 저해상도 시뮬레이션(half-downscale, p=0.25)이 **픽토그램 세부를 훼손**
- 여기에 소수클래스 2x 오버샘플 + class weight(balanced)가 소수 클래스(30) 쪽으로 결정경계를 이동 → 다수 클래스(11)가 소수 클래스로 유출

### 4.3 macro F1 영향 추정
- 클래스 30: 예측 82건 중 실제 최대 60건 → precision ≤ 0.73, F1 ≈ 0.85 (약 -0.0035 macro)
- 클래스 11: recall 398/420 = 0.948 (약 -0.0007 macro)
- **합계 약 -0.004** → iter8 예상 test F1 ≈ 0.985

## 5. 구조적 한계: track-aware val이 소수 클래스를 측정하지 못함

val 혼동 쌍은 13→3, 7→3, 1→8 (전부 원형 속도제한) **3건뿐**이고 11→30은 0건.

원인: 소수 클래스는 전체 **6트랙**(180장 ÷ 30프레임)뿐 → val 비율 20% 적용 시 **1트랙(약 30장)**만 배정. macro F1에서 동일 가중을 받는 소수 클래스를 신뢰성 있게 측정할 수 없음.

추가로 앙상블 val 0.9997이 iter6의 누수 val 0.9996과 사실상 동일한 이유:
- **best-epoch 선택 편향**: 3개 모델이 각각 동일 val에서 30 epoch 중 최고점 선택 → val 과적합
- **세션 유사성 잔존**: GTSRB 트랙은 촬영 세션 단위로 묶여 있어 다른 트랙도 동일 주행/조명 조건 공유. 공식 test는 별도 세션

## 6. 산출물

| 파일 | 내용 |
|------|------|
| `best_model_resnet50_96.pth` | ResNet-50 best (val F1 0.9930) |
| `best_model_efficientnet_128.pth` | EfficientNet-B0 best (0.9827) |
| `best_model_convnext_96.pth` | ConvNeXt-Tiny best (0.9969) |
| `result.csv` | 3-모델 앙상블 예측 (**제출 비권고**) |
| `test_probs.npy` | 앙상블 softmax 확률 (8670×43) |

## 7. iter9 계획 (우선순위)

1. **증강 롤백 + 선별 적용** (최우선)
   - `GaussianBlur` 강도/확률 대폭 축소 또는 제거, 저해상도 시뮬레이션 제거
   - 광도 증강(brightness/contrast)만 유지 — 픽토그램을 훼손하지 않음
   - 근거: 11→30 혼동의 직접 원인

2. **해상도 상향 (96 → 128)**
   - 삼각형 내부 픽토그램 보존에 직결. ConvNeXt-Tiny@128 우선

3. **ConvNeXt 중심 앙상블**
   - ConvNeXt-Tiny(0.9964) + ResNet-50(0.9904) 2종. EfficientNet-B0(0.9839)는 기여도 낮아 제외 검토

4. **소수 클래스 측정 개선**
   - 트랙 단위 **k-fold(5-fold)** 로 전체 데이터를 검증에 활용 → 소수 클래스도 6트랙 전부 평가
   - 또는 클래스별 최소 2트랙 val 보장
   - best-epoch 선택은 fold 평균으로 결정하여 선택 편향 완화

5. **클래스 11/30, 26/12 표적 점검**
   - 해당 쌍만 뽑아 혼동 행렬 직접 확인, 필요 시 class weight 완화(소수 클래스 과도한 우대 제거)

## 8. 잔여 status check 관리

- iter6로 1회 사용, **잔여 3회** 추정
- iter8은 분포 프록시가 악화되어 제출 시 정보 이득 대비 손실 큼 → **보류**
- 다음 제출은 iter9(증강 롤백 + 128px + ConvNeXt 중심)이 분포 프록시에서 iter6(78/12)을 개선했을 때 진행 권장

## 9. 학습 설정

| 항목 | 값 |
|------|----|
| Split | Track-aware 80/20 (클래스별 트랙 20%, seed 42) |
| 공통 | AdamW(backbone 3e-4 / head 1e-3, wd 1e-4), warmup 3ep + cosine(eta_min 1e-6), 30 epochs, AMP |
| 손실 | CE + class weight(balanced) + label smoothing 0.05 |
| 증강 | 회전 ±12/18°, translate/scale/shear, ColorJitter 0.5, **GaussianBlur p=0.35**, **저해상도 sim p=0.25**, RandomErasing p=0.2 (소수클래스), flip 금지 |
| 소수클래스 | 22개 2x 오버샘플 + 강증강 |
| TTA | 4-op (원본, ±8° 회전, affine) |
| 학습 시간 | 약 6시간 (RTX 2060) — ResNet 5분/ep, EfficientNet 3.2분/ep, ConvNeXt 3.75분/ep |

## 10. 교훈

- **"강건성 증강"은 무해하지 않다**: 세부 패턴 변별이 핵심인 과제에서 blur/저해상도 증강은 정보를 파괴할 수 있음. 증강 추가 시 클래스별 영향 확인 필요
- **val 설계가 클래스 분포를 반영해야 함**: 소수 클래스가 6트랙뿐인 구조에서 단일 hold-out은 macro F1 측정에 부적합 → k-fold 필요
- **분포 프록시(sum\|diff\|)가 val보다 유용했다**: 실제 test와 연결된 유일한 신호로서, val 0.9997이 놓친 문제를 조기에 포착
