# Iter12 Analysis & Retrospective: 검증셋 과대적합(Overfitting)의 함정

## 1. 개요
Iter12는 5-Fold 이종 앙상블(ConvNeXt + Scratch CNN)과 SciPy 기반 OOF 확률 최적화를 도입하여 리더보드 1위 달성을 목표로 한 야심 찬 시도였습니다. OOF 점수는 0.9976으로 역대 최고치를 갱신했으나, 심층 지표 분석 결과 실전(Test) 환경에서 치명적인 약점을 노출할 가능성이 확인되어 **폐기 및 Iter11 롤백**으로 결정되었습니다.

## 2. 세부 지표 분석 (Iter12 vs Iter11)

### [단일 모델 지표 비교]
| 지표 | Iter12 ConvNeXt@128 (5-fold) | Iter12 Scratch@48 (5-fold) | Iter11 ConvNeXt@128 (3-fold) |
| :--- | :--- | :--- | :--- |
| **OOF F1** | 0.9952 | 0.9949 | **0.9960** |
| **OOF Acc** | 0.9957 | 0.9951 | **0.9962** |
| **Fold Noise(σ)**| ±0.0022 | ±0.0035 | ±0.0029 |
| **OOF 오류** | 111개 | 128개 | **100개** |
| **Proxy sum\|d\|**| 70 | 84 | **68** |
| **Proxy max\|d\|**| 13 | 19 | **8** |

* **분석**: 단일 모델 기준으로는 이전 버전인 Iter11(3-Fold)이 Iter12(5-Fold)보다 모든 지표에서 우세합니다. 3-Fold에서 5-Fold로 넘어가면서 Fold당 Validation 사이즈가 줄어들어 소수 클래스 측정이 불안정해진 것(Fold Noise)이 악영향을 미친 것으로 보입니다.

### [앙상블 지표 비교]
* **Iter12 최적 가중치**: ConvNeXt 0.44 / Scratch 0.56
* **Iter12 앙상블 OOF F1**: 0.9976
* **Iter12 앙상블 Proxy max\|d\|**: ~11 (class 12에서 -11 오차)

## 3. 폐기 결정의 핵심 사유 (Risk Analysis)
1. **취약한 모델에 과도한 가중치 배정**: 단독 Proxy 지표가 84/19로 매우 나쁜 Scratch CNN에 0.56이라는 높은 가중치가 배정되었습니다. 이는 OOF 내부의 오차를 상호보완하기 위한 기계적 편법일 뿐, Test 셋에서의 실전 성능을 보장하지 않습니다.
2. **Proxy max\|d\| 악화**: OOF 점수는 0.0016 올랐지만, 가장 중요한 안정성 지표인 Proxy max|d|는 8(Iter11)에서 11(Iter12)로 오히려 악화되었습니다. 이는 특정 클래스를 과도하게 잘못 짚는 현상이 심해졌음을 의미합니다.

## 4. 결론 및 넥스트 스텝 (Action Item)
* **결론**: Iter12의 OOF 점수 상승은 Fold Noise 안에서의 변동이자 '검증셋 과대적합(Overfitting to OOF)'의 전형적인 함정입니다.
* **조치**: Iter12 앙상블 파이프라인을 전면 폐기하고, 모든 지표가 가장 안정적이었던 **Iter11 ConvNeXt@128 (3-Fold) 단일 모델(Proxy 68/8)을 복구(Iter13-recovery)하여 최종 제출안으로 확정**합니다.
