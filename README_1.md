# 🏥 환자 의료비 예측 (Insurance Cost Prediction)

## 📌 프로젝트 개요
이 프로젝트는 환자의 신상 정보와 건강 데이터를 기반으로 **개별 의료비(Insurance Charges)**를 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다.
다양한 회귀 알고리즘(Linear Regression, XGBoost, LightGBM)과 앙상블 기법을 활용하여 예측 정확도를 높였으며, 탐색적 데이터 분석(EDA)을 통해 중요 변수를 파악했습니다.

## 📂 데이터셋 설명
- **데이터 소스**: `insurance.csv`
- **데이터 크기**: 1,338개의 샘플, 7개의 컬럼
- **주요 컬럼**:
  - `age`: 나이
  - `sex`: 성별
  - `bmi`: 체질량지수 (BMI)
  - `children`: 자녀 수
  - `smoker`: 흡연 여부
  - `region`: 거주 지역
  - `charges`: 의료비 (Target Variable)

## 📊 탐색적 데이터 분석 (EDA) 및 전처리
1. **Target Variable (`charges`) 분석**
   - 초기 데이터 분포가 심하게 치우친(Skewed) 형태임을 확인했습니다.
   - **로그 변환(Log Transformation)**을 적용하여 데이터 분포를 정규분포에 가깝게 변환, 모델의 안정성을 확보했습니다.

2. **파생 변수 생성 (Feature Engineering)**
   - **`obese_smoker`**: 흡연 여부(`smoker`)와 비만 여부(`is_obese`)의 상호작용이 의료비에 큰 영향을 미친다는 점에 착안하여 새로운 변수를 생성했습니다.
   - 이는 모델이 고위험군을 더 잘 식별하도록 돕습니다.

## 🤖 모델링 (Modeling)
본 프로젝트에서는 세 가지 단계로 모델링을 수행했습니다:

### 1. 베이스라인 모델 (Baseline)
- **Linear Regression**: 변수 간의 선형 관계를 파악하기 위한 기본 모델로 사용했습니다.

### 2. 고급 회귀 모델 (Advanced Models)
- **XGBoost Regressor**: 강력한 Gradient Boosting 알고리즘으로, 비선형적인 패턴 학습에 뛰어납니다. 과적합 방지를 위해 하이퍼파라미터 튜닝을 적용했습니다.
- **LightGBM Regressor**: XGBoost보다 학습 속도가 빠르고 대용량 데이터 처리에 유리한 모델입니다. 역시 튜닝을 통해 최적의 성능을 도출했습니다.

### 3. 앙상블 (Ensemble)
- **Weighted Ensemble**: XGBoost와 LightGBM의 예측 결과를 가중 평균하여 개별 모델보다 더 견고하고 정확한 예측 성능을 달성했습니다.

## 📈 분석 결과 및 성능
- **성능 지표**: 모델 평가는 결정 계수(**$R^2$ Score**)와 평균 절대 오차(**MAE**)를 기준으로 수행했습니다.
- **주요 발견**:
  - `smoker`(흡연 여부)가 의료비 결정에 가장 결정적인 영향을 미치는 변수였습니다.
  - `bmi`와 `age` 또한 중요한 예측 인자로 작용했습니다.
  - `obese_smoker` 파생 변수가 모델 성능 향상에 기여했습니다.
- **시각화**: XGBoost의 `feature_importance`를 시각화하여 변수 중요도를 직관적으로 확인했습니다.

## 💻 실행 방법

### 요구 사항 (Requirements)
이 프로젝트를 실행하기 위해 다음 라이브러리가 필요합니다:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost lightgbm
```

### 노트북 실행
`IN_predic.ipynb` 파일을 Jupyter Notebook 또는 Google Colab에서 열고 모든 셀을 순차적으로 실행하세요.
```bash
jupyter notebook IN_predic.ipynb
```
