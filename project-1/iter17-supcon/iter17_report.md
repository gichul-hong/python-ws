# Iter17 결과 보고서: 대조학습(SupCon) 도입과 Scratch 모델의 한계 증명

## 1. 실험 개요
* **목적:** 데이터가 1/6로 줄어든 극한의 환경(`iter16`)에서 붕괴했던 Scratch CNN을, **지도 대조학습(Supervised Contrastive Learning, SupCon)** 기법을 통해 되살릴 수 있는지 검증.
* **적용 기법:** `ContrastiveTrackLoss` 도입
  * 모델이 분류 확률(Logits)뿐만 아니라 **512차원 특징 벡터(Feature Vector)**를 출력하도록 구조 변경.
  * 같은 `track_id`를 가진 이미지들은 공간상에서 당기고(Pull), 다른 이미지들은 밀어내는(Push) 손실 함수 추가 (Loss 비중 0.5).
* **데이터:** 4,340장 (Pruned Dataset)
* **모델 체제:** ConvNeXt-Tiny (Pretrained) + Scratch CNN (Random Init)

---

## 2. 실험 결과 (Score)

| 지표 | ConvNeXt + SupCon | Scratch CNN + SupCon | 
| :--- | :---: | :---: | 
| **OOF Accuracy** | 0.9924 | ~0.4900 (붕괴) | 
| **OOF Macro F1** | 0.9898 | ~0.3100 |

* **특이사항:** ConvNeXt는 기존의 뛰어난 방어력을 유지했으나, 기대를 모았던 Scratch CNN은 대조학습을 적용했음에도 불구하고 F1 점수 30%대에서 학습이 정체되었습니다.

---

## 3. 핵심 인사이트 (The Bitter Lesson 2)

### 🔴 1. 딥러닝에서 '초기 지식'의 절대적 중요성 증명
대조학습(SupCon)은 이미 어느 정도 선(Edge)과 질감(Texture)을 추출할 줄 아는 모델에게 '불변성(Invariance)'을 가르치는 데는 탁월합니다. 하지만, 아무것도 모르는 백지상태(Random Weights)의 Scratch CNN에게 4,340장이라는 극소량의 데이터는 너무 가혹했습니다.
**결론:** 데이터 기근 상태에서는, 아무리 최첨단 손실 함수(SupCon)와 증강 기법(CutMix, MixUp)을 쏟아부어도 **'ImageNet 사전학습(Pretrained) 가중치'가 가진 절대적인 힘을 결코 대체할 수 없습니다.**

### 🟢 2. ConvNeXt의 안정적 성능과 Track 불변성 획득
ConvNeXt는 SupCon Loss를 적용한 후에도 99% 이상의 OOF 정확도를 달성했습니다. 특히 같은 표지판(Track)의 흐린 사진과 선명한 사진을 강제로 군집화(Clustering)하는 훈련을 받았으므로, Test 환경에서 처음 보는 노이즈나 각도 변화에 더욱 강건한(Robust) 예측력을 갖추었을 것으로 기대됩니다.

---

## 4. 넥스트 스텝 (`iter18` 최종 앙상블 전략 수정)
`iter17`의 뼈아픈(그러나 훌륭한) 실패를 통해, "정제된 소량의 데이터 셋에서는 Pretrained 모델만이 살아남는다"는 진리를 확인했습니다.
이에 따라 `iter18-final`에서는 구제 불능인 Scratch CNN을 최종적으로 **퇴출(Drop)**하고, **100% ImageNet 사전학습을 거친 엘리트 이기종 4-Model 체제(ConvNeXt, ResNet, EfficientNet, MobileNet)**로 최종 SOTA(State-of-the-Art) 앙상블 점수 달성에 도전합니다.
