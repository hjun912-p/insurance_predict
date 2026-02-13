# insurance_predict
팀 조별 프로젝트 데이터 
이용된 사이트 : https://www.kaggle.com/datasets/mirichoi0218/insurance/data
               https://www.kaggle.com/code/ashishcode/medical-expense-99-accuracy-without-tuning/notebook
# 🏥 미국 의료비용 예측 모델링 프로젝트
> **데이터 전처리와 앙상블 모델을 활용한 보험료 예측 최적화**

이 프로젝트는 개인의 인적 사항(나이, 성별, BMI, 자녀 수, 흡연 여부, 지역)을 바탕으로 연간 의료비(`charges`)를 정확하게 예측하는 머신러닝 모델을 구축하는 것을 목표로 합니다.

---

## 👥 팀원 및 역할
- **이지수**: 탐색적 데이터 분석(EDA), 흡연 여부에 따른 데이터 분포 및 상관관계 분석
- **박효준 (Main Modeler)**: 데이터 전처리 전략 수립(Label Encoding, Log 변환), 베이스라인 모델 구축 및 성능 개선 (XGBoost 0.91 달성)
- **김성일**: 데이터 세분화를 통한 정밀 이상치 제거 및 모델 안정성 검토
- **박창민**: 외부 모델(Kaggle)과의 성능 벤치마킹 및 결과 검증

---

## 🚩 프로젝트 히스토리: 데이터셋 변경
초기에는 '모바일 기기 사양 데이터'를 분석했으나, 해당 데이터셋에 페이크(Fake) 데이터가 포함되어 있고 타겟 변수와의 상관관계가 전무함을 확인했습니다. 
팀 회의 끝에 분석 가치가 높고 실제 비즈니스 인사이트 도출이 가능한 **'미국 의료비용 데이터셋'**으로 변경하여 진행하였습니다.

---

## 🛠️ 핵심 데이터 전처리 (박효준 기여분)
모델의 예측력을 극대화하기 위해 다음과 같은 정석적인 전처리 파이프라인을 구축했습니다.

### 1. 데이터 클리닝 및 이상치 제거
- **Boxplot 분석**: 보험료 데이터의 우측 꼬리 부분에 밀집된 이상치를 확인.
- **정제 작업**: 모델이 일반적인 패턴을 학습하는 데 방해가 되는 극단적 수치를 제거하여 데이터의 품질을 높였습니다.

### 2. 범주형 데이터 인코딩
- **Label Encoding**: 성별(`sex`), 흡연 여부(`smoker`) 등 이진(Binary) 변수를 숫자로 변환하여 연산 효율성을 높였습니다.
- **One-Hot Encoding**: 지역(`region`)과 같이 순서가 없는 다중 범주 데이터에 적용하여 모델의 불필요한 편향을 방지했습니다.

### 3. 타겟 데이터 로그 변환 (`np.log1p`)
- 보험료 데이터의 심한 왜곡(Skewness)을 해결하기 위해 로그 변환을 수행했습니다.
- 이를 통해 데이터 분포를 **정규 분포**에 가깝게 만들어, 선형 회귀 모델 및 트리 기반 모델의 학습 성능을 비약적으로 향상시켰습니다.

---

## 📈 모델링 성과
단순 회귀부터 최신 부스팅 알고리즘까지 단계별로 적용하며 성능을 비교 분석했습니다.

| Model | R2 Score (Before Preprocessing) | R2 Score (After Optimization) |
| :--- | :---: | :---: |
| **Linear Regression** | 0.75 | 0.87 |
| **RandomForest** | 0.80 | 0.89 |
| **XGBoost** | **0.81** | **0.91** |

> **Key Discovery**: 전처리 전 0.81이었던 XGBoost 모델이 로그 변환과 이상치 제거 후 **0.91**까지 상승하며, 데이터 가공의 중요성을 입증했습니다.

---

## 💡 주요 인사이트 및 결론
1. **흡연과 비만의 상관관계**: 분석 결과, '흡연 여부'가 가장 강력한 변수였으며 특히 **BMI 30 이상의 고도비만 흡연자**는 의료비가 기하급수적으로 증가하는 비선형적 패턴을 보였습니다.
2. **모델 선택의 이유**: 선형 회귀보다 **XGBoost**가 우수했던 이유는 변수 간의 복잡한 상호작용(Interaction)을 트리 구조로 더 정교하게 잡아냈기 때문입니다.
3. **전처리의 힘**: 단순히 복잡한 모델을 쓰는 것보다, 타겟 데이터를 **로그 변환**하여 데이터의 질을 높이는 것이 성능 향상에 더 결정적인 역할을 했습니다.


# 🏥 미국 의료비용 예측 모델링 최적화 프로젝트
> **그룹 세분화 이상치 제거와 앙상블 모델을 통한 R² 0.97 달성**

---

## 🛠️ [심화] 데이터 정제 전략 (Advanced Outlier Removal)
본 프로젝트의 핵심 성능 향상 비결은 단순히 전체 수치를 기준으로 이상치를 제거한 것이 아니라, **'데이터 재해석을 통한 그룹별 정제'**에 있습니다.

### 🔍 세분화된 그룹화 (Granular Grouping)
보험료가 비슷하게 형성되어야 하는 구간을 정의하기 위해 3단계로 데이터를 분할했습니다.
1. **나이대별 분할**: 10단위 (20대, 30대, 40대...)
2. **흡연 여부 분할**: Yes / No
3. **BMI 지수 분할**: WHO 기준 (저체중, 정상, 과체중, 전비만, 고도비만)



### 🧪 그룹별 정규분포 95% 필터링
각 세분화된 그룹 내에서 평균을 구하고, **정규분포 기준 상하위 2.5%(총 5%)를 제외한 95%의 데이터만 유지**했습니다. 이를 통해 '20대 비흡연자' 중 유독 보험료가 높은 특이치를 제거하여 모델의 학습 노이즈를 완벽에 가깝게 제거했습니다.


### 📈 모델 성능 향상 지표 (Model Performance)

| 분석 단계 | 모델명 | R² Score | 비고 |
| :--- | :--- | :---: | :--- |
| **Step 1** | Linear Regression (Raw) | 0.75 | 프로젝트 초기 베이스라인 |
| **Step 2** | Linear Regression (Optimized) | 0.87 | 전처리 및 로그 변환 적용 |
| **Step 3** | XGBoost / RandomForest | 0.91 | 앙상블 알고리즘 적용 |
| **Final** | **Refined Model (Final)** | **0.97** | **그룹 세분화 및 정밀 정제 완료** |
---
99퍼센트 모델과의 비교
<img width="1160" height="677" alt="image" src="https://github.com/user-attachments/assets/8945d6d9-ebd0-4d70-b1a0-5574a4ba7d03" />
<img width="681" height="638" alt="image" src="https://github.com/user-attachments/assets/f223a6ad-c916-4e75-9c18-0ac0087e3197" />
<img width="887" height="638" alt="image" src="https://github.com/user-attachments/assets/0ef9e8ba-a78a-4668-b977-9d1ae1599319" />
<img width="1223" height="603" alt="image" src="https://github.com/user-attachments/assets/fe9671d7-5a5d-4eae-b6ac-f30097865335" />
<img width="789" height="636" alt="image" src="https://github.com/user-attachments/assets/93271eb5-07a3-478c-97f0-8b65ec4fa632" />
<img width="1233" height="678" alt="image" src="https://github.com/user-attachments/assets/b0918dc5-eba1-48aa-b117-4832e2ba5c85" />

## ⚙️ 기술 스택
- **언어**: Python 3.x
- **라이브러리**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Seaborn, Matplotlib
- **도구**: Jupyter Notebook, Google Colab
