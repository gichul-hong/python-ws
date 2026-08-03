# Iter3 (fable): Ensemble + Hard-Class Focused Augmentation — Technical Report

## 1. 개요

iter2의 test 예측 분포 분석(기대 분포 = train/3)에서 발견된 혼동 클래스를 겨냥한 iteration.

- **Ensemble Val Macro F1 = 0.9996** (Val Acc 0.9998)
- **분포 오차 sum|diff| = 74** — iter2의 184 대비 **60% 감소** → test 일반화 개선의 핵심 근거
- 오분류 하한: iter2 ≥92건(acc 상한 ~98.94%) → iter3 **≥37건(acc 상한 ~99.57%)**
- 코드: `iter3-fable/train.py` / 제출: `iter3-fable/result.csv` / 가중치: `best_model_seed{42,123,777}.pth`

## 2. iter2 대비 변경점

| 항목 | iter2 | iter3-fable |
|------|-------|-------------|
| 모델 수 | 단일 | **3개 앙상블 (seed 42/123/777)**, soft voting |
| 하드클래스 처리 | 없음 | **{11,12,20,21,30,32,34,38} 2배 오버샘플링** (20,808→24,552) |
| 하드클래스 증강 | 공통 파이프라인 | **강화 파이프라인**: ±18° 회전, translate 0.15, scale 0.75–1.25, shear 8, hue 0.05, GaussianBlur(p=0.3), RandomErasing(p=0.25) |
| Epochs | 50 | 45 × 3모델 |
| TTA | 4-op | 4-op × 3모델 = 12 예측 평균 |
| Val split | seed 42 고정 | 동일 (앙상블 val F1 측정 가능하도록 고정) |

하드클래스 선정 근거 (iter2 분포 diff): 12(−23), 38(−15), 20(+14), 11(+14), 34(+13), 32(+11), 30(−9), 21(−8).

## 3. 학습 결과

| Seed | Best Val Macro F1 |
|------|------|
| 42  | 0.9995 |
| 123 | 0.9996 |
| 777 | 0.9995 |
| **Ensemble (×4 TTA)** | **0.9996** (Acc 0.9998) |

- 개별 모델 val 점수는 iter2(0.9999)보다 미세하게 낮음 — 강한 증강으로 val 난이도가 아닌 train 난이도가 올라간 효과. val은 이미 포화 상태(오분류 1~2건 수준)라 변별력이 없음.

## 4. Test 예측 분포 검증 (기대 분포 = train/3, 총 8,670)

| 지표 | iter2 | iter3 |
|------|-------|-------|
| sum \|diff\| | 184 | **74** |
| max \|diff\| | 23 | **14** |
| 오분류 하한 (sum/2) | 92건 (1.06%) | **37건 (0.43%)** |

- 분포 불일치가 60% 감소 → 혼동 클래스(12↔11, 38↔34/20 등) 오분류가 실제로 줄었다는 신호.
- val 점수가 아닌 test 분포 기반 지표에서 개선이 확인된 것이 이번 iteration의 핵심 성과.

## 5. 학습 설정

| 항목 | 값 |
|------|----|
| 공통 | 48×48, AdamW(lr 1e-3, wd 1e-4), label smoothing 0.05, CosineAnnealingLR, batch 128, class weight balanced |
| Best 선택 | val macro F1 |
| 학습 시간 | 모델당 약 30분, 전체 약 95분 (RTX 2060) — 오버샘플링+강한 증강으로 iter2보다 epoch당 느림 |
| 환경 | conda `gpu-torch`, PyTorch 2.5.1+cu121 |

## 6. 추가 개선 방향

1. **분포 사전(prior) 보정**: 기대 분포(train/3)를 알고 있으므로 헝가리안/Sinkhorn 방식으로 예측을 기대 분포에 정렬 — sum|diff|를 0 근처로 강제 가능 (오히려 정확도를 해칠 수 있어 soft 보정 권장)
2. **하드클래스 쌍 대상 fine-tuning**: 12/11, 38/34/20 만의 binary/ternary 분류기 추가
3. **모델 다양성 확대**: 해상도(48/64) 또는 아키텍처를 섞은 heterogeneous ensemble
4. **전체 데이터 재학습**: val 없이 26,010장 전량으로 최종 모델 학습

## 7. 참고 사항

- `DATA_ROOT` 하드코딩 — 환경에 따라 수정 필요
- test id 순서는 상위 `result.csv` 템플릿 그대로 유지 (row 순서 불변)
- 분포 체크 로직은 `train.py` 말미에 내장되어 실행 시 자동 출력
