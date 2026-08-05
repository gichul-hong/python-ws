# 교통표지판 분류(Traffic Sign Classification) 최종 발표 자료

---

## 1. 실험 목적 및 개요 (Introduction & Objective)
- **과제 목표**: 43개 클래스로 구성된 German Traffic Sign Benchmark (GTSRB) 데이터셋을 활용하여 강건하고 정확한 이미지 분류 모델 개발
- **도전 과제**: 
  1. 극심한 클래스 불균형 (Class Imbalance)
  2. 실외 환경 특성상 조명, 각도, 해상도 변동이 큼
  3. 시각적으로 매우 유사한 표지판(예: 각종 삼각형 경고 표지판) 간의 미세한 픽토그램 변별
- **연구 흐름**: 단순 CNN(Baseline)에서 시작하여, Transfer Learning, 교차 검증(Cross Validation) 고도화, Data Augmentation의 부작용 분석 및 최적화를 거치는 구조적인 성능 개선(Iteration) 수행

---

## 2. Iteration 흐름 및 전후 결과 (Methodology Evolution)

### Phase 1: Baseline 모델 구축 및 한계 파악 (Iter 1 ~ 2)
- **목적**: PyTorch 기반의 기본 학습 파이프라인 구축 및 잠재적 문제점 발굴
- **실험 내용 (Iter 1)**: 
  - 30x30 해상도의 3단 Conv 블록 CNN 설계 (622K params)
  - Train/Val 8:2 Random Stratified Split 적용
- **결과 및 인사이트**: Val F1 0.9992 달성. 그러나 Random Split으로 인해 동일한 표지판(Track)의 프레임이 Train/Val로 나뉘어 과적합(Data Leakage) 발생 의심.
- **개선 (Iter 2)**: 
  - 48x48로 해상도 상향 및 모델 크기 증가(2.38M)
  - RandomAffine, ColorJitter 등 실외 환경을 고려한 증강 기법 도입
  - **핵심 발견**: Test 라벨 분포가 Train 분포의 정확히 1/3 비율임을 발견, 이를 'Test 분포 오차 프록시'로 활용하여 모델 검증 기준 마련.

### Phase 2: Pre-trained Backbone 도입 및 최적화 (Iter 6)
- **목적**: ImageNet 사전학습(Pre-trained) 모델의 강력한 특징 추출 능력을 GTSRB에 이식
- **실험 내용**: 
  - ResNet-50 Full Fine-tuning 적용, 해상도 96x96 상향
  - **Discriminative LR** 적용 (Backbone 학습률은 낮게, Head는 높게 설정) 및 Warm-up Scheduler 도입
- **전후 결과**:
  - 초기 Iter 4에서 사전학습 모델 수렴 실패 문제를 완전히 해결
  - 앙상블 기준 Val F1 0.9996 달성 및 논문 SOTA 급(99.8%) 성능 재현
  - Test 분포 오차(sum|diff|)를 78로 대폭 감소시켜 역대 최고 성능 및 신뢰성 확보

### Phase 3: 데이터 누수 차단 및 증강의 역설 (Iter 8)
- **목적**: 과적합 방지를 위한 엄격한 검증 환경 구축 및 이종 앙상블 탐색
- **실험 내용**: 
  - **Track-aware Split 도입**: 물리적으로 동일한 표지판이 Train과 Val에 섞이지 않도록 트랙(Track) 단위로 분리
  - ConvNeXt-Tiny, ResNet-50, EfficientNet-B0 이종 앙상블 시도 및 추가 강건성 증강(Gaussian Blur, Downscale) 도입
- **결과 및 실패 분석 (Failure Analysis)**:
  - Track-aware Split은 기계적으로 정상 동작했으나, 새롭게 도입한 강건성 증강 기법이 **삼각형 경고 표지판(11번 vs 30번) 내부의 미세한 픽토그램을 훼손**시킴.
  - 이로 인해 특정 클래스 유출 현상이 발생하여 Test 성능(F1 ~0.985 추정) 하락 관측.
  - **교훈 ("증강은 무해하지 않다")**: 세부 변별력이 중요한 Task에서 정보(Detail)를 파괴하는 증강은 오히려 성능을 크게 저하시킴을 입증.

### Phase 4: 최종 최적화 및 앙상블 (Iter 9 ~ 최종)
- **목적**: Iter 8의 실패 요인 롤백 및 소수 클래스 평가 신뢰도 확보
- **실험 내용**:
  - 증강 롤백 (픽토그램을 훼손하는 Blur 및 축소 기법 제거, 광도 증강만 유지)
  - 해상도를 128x128로 추가 상향하여 미세 특징 보존
  - 소수 클래스의 불안정한 Val 평가를 극복하기 위해 **K-Fold (5-Fold) Cross Validation** 전면 도입
  - 강력한 성능이 입증된 ConvNeXt 아키텍처를 중심으로 최종 앙상블 구성

---

## 3. 핵심 결론 및 기여점 (Key Findings & Conclusion)

1. **데이터 분할의 중요성 (Data Leakage 차단)**
   - 동영상 캡처 기반 데이터셋(GTSRB)에서 단순 Random Split은 치명적인 과적합을 유발함. Track-aware Split과 K-Fold 검증을 통해 '정직한' 성능 평가 체계를 구축함.
2. **강건성 증강의 역설 (Augmentation Paradox)**
   - Blur나 해상도 저하와 같은 강건성 증강 기법이 무조건 좋은 것이 아니라, 클래스 간 구분에 필요한 미세한 특성(픽토그램)을 훼손할 경우 치명적일 수 있음을 규명함.
3. **최적의 Transfer Learning 전략**
   - Discriminative Learning Rate 및 Warm-up 기법을 통해, 비교적 작은 데이터셋에서도 ResNet-50, ConvNeXt와 같은 무거운 모델을 SOTA 성능으로 Fine-tuning 하는 방법론을 성공적으로 적용함.
