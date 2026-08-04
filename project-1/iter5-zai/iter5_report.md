# Iter5 (zai): ResNet18 Frozen Features + ML Classifiers — Technical Report

## 1. 개요

ResNet18(ImageNet pretrained)을 **frozen feature extractor**로 사용해 512-dim 특징을 추출하고, 머신러닝 알고리즘(XGBoost / RandomForest / LinearSVM / RBF-SVM)으로 분류하는 접근. CNN finetune의 수렴 문제를 회피하면서 특징 기반 분류의 가능성을 탐색한 iteration.

- **최고 성능: RBF-SVM, Val Macro F1 = 0.9908** (Val Acc 0.9877)
- **Test 분포 오차 sum|diff| = 1356** — 일반화 한계 명확 (특정 클래스 붕괴, max|diff|=220)
- 코드: `iter5-zai/train.py` / 제출: `iter5-zai/result.csv`

## 2. 방법론

### 2.1 Feature Extractor
- **ResNet18, ImageNet pretrained, fully frozen** (`requires_grad=False`, fc → Identity)
- 입력 224×224 (ImageNet 학습 분포 정렬), `avgpool` 출력 512-dim 특징
- 특징 추출은 1회 수행, `_feat_cache/`에 캐싱하여 재실행 시 즉시 로드
- 증강 variant: 소수 클래스 오버샘플링 시 `aug_tf`(±10° 회전, ColorJitter) 적용한 이미지로부터 특징 추출 → 특징 공간에서 다양성 확보

### 2.2 ML 분류기 4종 비교

| 분류기 | 핵심 설정 | 특징 |
|--------|-----------|------|
| XGBoost | n_estimators=200, max_depth=6, lr=0.2, tree_method=hist, sample_weight=class weight | 부스팅, multiclass softmax |
| RandomForest | n_estimators=400, class_weight=balanced | 배깅, 비선형 |
| LinearSVM | C=1.0, class_weight=balanced, one-vs-rest | 선형 결정경계 |
| **RBF-SVM** | C=10.0, kernel=rbf, gamma=scale, class_weight=balanced | 비선형 커널 |

- 모든 모델에 `StandardScaler` 전처리(SVM/RF), class weight balanced로 소수 클래스 보정
- `SMALL_CLASSES`(180장 22개) 2x 오버샘플링은 iter4와 동일(train-only 기준)

## 3. 학습 결과

| 분류기 | Val Acc | Val Macro F1 | 순위 |
|--------|---------|--------------|------|
| **RBF-SVM** | 0.9877 | **0.9908** | 1 |
| XGBoost | 0.9354 | 0.9334 | 2 |
| RandomForest | 0.9216 | 0.9324 | 3 |
| LinearSVM | 0.9083 | 0.8975 | 4 |

### 관찰
- **RBF-SVM이 압도적** — frozen ImageNet 특징의 비선형 구조를 RBF 커널이 효과적으로 포착. SMO 기반 전체 데이터 학습(24K 샘플)으로 결정경계 정교
- XGBoost/RandomForest는 0.93대 → 트리 기반 모델은 512-dim 밀집 특징에서 분할 한계. ImageNet 특징이 연속적이라 트리 분할에 불리
- LinearSVM 0.90 → 선형 결정경계로는 클래스 간 비선형 분리 불충분

## 4. Test 예측 분포 검증 (기대 분포 = train/3, 총 8,670)

| 지표 | iter2 | iter3 | iter4 | **iter5** |
|------|-------|-------|-------|----------|
| sum \|diff\| | 184 | 74 | 96 | **1356** |
| max \|diff\| | 23 | 14 | 15 | **220** |
| 오분류 하한 | 92건 (1.06%) | 37건 (0.43%) | 48건 (0.55%) | **678건 (7.82%)** |

- 분포 오차 폭증(1356) → 특정 클래스 심각 붕괴(max|diff|=220)
- frozen ImageNet 특징이 교통표지판(그래픽 기호) 변별에 충분하지 않아 일부 클래스 시스템적 오분류 발생

## 5. iteration 간 비교

| 지표 | iter2 (CNN) | iter3 (CNN+hard) | iter4 (CNN+small) | **iter5 (ResNet+ML)** |
|------|-------------|------------------|--------------------|-----------------------|
| Val Macro F1 | 0.9999 | 0.9996 | 0.9996 | **0.9908** |
| Test sum\|diff\| | 184 | 74 | 96 | 1356 |
| 정당성 | 정당 | test 참조 위반 | 정당 | 정당 |

## 6. 핵심 인사이트

1. **도메인 갭이 핵심 제약**: ImageNet(자연 이미지)과 교통표지판(그래픽 기호)은 도메인 차이가 큼. frozen 특징은 finetune 없이 이 갭을 극복 불가 → iter4에서 finetune조차 느렸던 원인과 일관
2. **RBF-SVM의 강점**: 제한된 특징에서도 비선형 커널이 0.99 달성 → frozen 특징에 비선형 구조 존재, 단 분류 경계 정밀도는 CNN 미달
3. **트리 모델 한계**: 밀집 연속 특징(512-dim float)에서 트리 분할은 비효율적 → XGBoost/RF는 이미지 특징에 부적합
4. **일반화 한계**: val(0.9908)과 test 분포 오차(1356)의 괴리 → val/test 분포 차이에서 frozen 특징의 변별력이 더 저하

## 7. 학습 설정

| 항목 | 값 |
|------|----|
| Feature extractor | ResNet18 ImageNet frozen, 224×224, 512-dim avgpool |
| 특징 캐싱 | `_feat_cache/features.npz`, `feat_aug_0.npz` |
| 소수클래스 | SMALL_CLASSES 22개, 2x 오버샘플링 (augmented variant) |
| 전처리 | StandardScaler (SVM/RF) |
| Best 분류기 | RBF-SVM (C=10, gamma=scale, class_weight=balanced) |
| 학습 시간 | 특징 추출 ~2분(캐싱 후 0), XGBoost ~9분, RF ~2분, LinearSVM ~7분, RBF-SVM ~2분 |
| 환경 | conda `gpu-torch`, PyTorch 2.5.1+cu121, xgboost 3.3.0, sklearn 1.9.0 |

## 8. 결론 및 향후 방향

- **frozen 특징 + ML은 0.99 도달(RBF-SVM)했으나 end-to-end CNN(0.9996)에 미치지 못함**. 일반화(test 분포)에서 더 큰 격차
- frozen 접근은 도메인 갭 극복에 근본적 한계 → finetune 또는 end-to-end 학습이 우위

### 추가 개선 방향
1. **ResNet18 finetune + ML 결합**: iter4의 느린 수렴을 100+ epoch 장기 학습으로 극복 후, finetune 특징으로 RBF-SVM 재적용 → 도메인 특화 특징 + 비선형 분류 시너지
2. **특징 앙상블**: frozen 특징(RBF-SVM 확률) + CNN soft voting 가중 결합 → 다양성 활용
3. **소수 클래스 특징 증강 강화**: N_AUG_SMALL 증가(2~3)로 특징 공간 다양성 확보
4. **CNN 우선 전략 유지**: 현 상태에서는 iter4(자작 CNN, F1 0.9996)가 가장 균형 잡힌 정당한 방법론

## 9. 참고 사항

- `DATA_ROOT` 하드코딩 — 환경에 따라 수정 필요
- 특징 캐싱으로 재실행 시 추출 단계 생략 가능
- xgboost 3.3.0 설치 필요 (`pip install xgboost`)
- test id 순서는 상위 `result.csv` 템플릿 유지 (row 순서 불변)
- 분포 체크는 검증 지표로만 사용, 클래스 선정에는 미사용
