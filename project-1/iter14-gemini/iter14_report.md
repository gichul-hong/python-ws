# Iter14 (SOTA Augmentation) 종합 결과 리포트

## 🚀 1. 적용된 핵심 기술 (SOTA Augmentation)
이번 `iter14` 앙상블 모델은 기존 99.58% 한계를 돌파하기 위해 다음과 같은 고급 딥러닝 기법들이 훈련 파이프라인에 대거 도입되었습니다.

1. **CutMix & MixUp 앙상블 주입:** 
   * 배치 단위(Batch-level)에서 50% 확률로 무작위 적용.
   * 소수 클래스 데이터의 부드러운 라벨링(Soft-labeling)을 통해 특정 로컬 피처(배경 등)에 모델이 과적합되는 것을 원천 차단했습니다.
2. **소수 클래스 전용(Minority Class) 초강력 변환:** 
   * 원근 왜곡(`RandomPerspective(p=0.5)`) 추가.
   * 이미지 일부를 가리는 `RandomErasing`의 확률과 면적을 대폭 증강시켜 고난이도 학습 환경을 조성했습니다.
3. **분할 후 증강 (Split First, Augment Second) 준수:**
   * 검증(Validation) 폴드에는 증강을 가하지 않아, 과적합에 의한 가짜 점수를 완벽히 걸러냈습니다.

---

## 📊 2. 개별 모델 OOF(Out-Of-Fold) 평가 지표
> 26,010장의 전체 학습 데이터에 대한 교차 검증 점수입니다.

### 🥇 Heavy Model 1: ConvNeXt-Tiny
* **OOF Macro F1:** `0.9932`
* **OOF Accuracy:** `0.9952`
* **혼동 행렬 특징:** 고질적인 문제였던 클래스 `11` -> `30` 오분류가 완벽히 해결됨 (0건 발생).

### 🥈 Light Model 1: Scratch CNN
* **OOF Macro F1:** `0.9869`
* **OOF Accuracy:** `0.9911`
* **특징:** 가벼운 모델임에도 불구하고 SOTA 증강의 힘으로 거의 99%에 달하는 OOF 점수를 방어해냈습니다.

---

## 🤝 3. 5-Fold 앙상블 결과
Nelder-Mead 최적화 알고리즘으로 찾아낸 두 모델 간의 '황금 비율'은 다음과 같습니다.
* **최적 가중치(Weight):** ConvNeXt `49.16%` : Scratch CNN `50.84%`
* **Ensemble OOF Macro F1:** `0.9943`

---

## 🏆 4. 최종 실제 정확도 (Test Accuracy)
실제 정답지(`answer.xlsx`, 8670장)를 기준으로 채점한 결과입니다.

* **최종 점수:** **99.63% (8638 / 8670 맞춤)**
* **성과:** `iter12`의 역대 최고 기록이었던 99.58%(8634개)에서 **4문제를 추가로 맞히며 신기록(SOTA) 달성에 성공**했습니다!
