# Models

> 각 model의 특징을 이해하고, 적합한 model을 찾는다

</br>

</br>

## KNN

* regression : KNeighborsRegressor()
* classification : KNeighborsClassifier()

</br>

### 주요 매개변수

weight : uniform, distance

n_neighbors : 이웃 수

</br>

### 특징

dataSet의 shape(sampe, feature)이 너무 큰 경우 예측이 느리다

전처리 과정 중요하다

</br>

</br>

## 선형 모델

* regression : linearRegression(), Ridge()(L2), Lasso()(L1), ElasticNet(), SGDRegressor()(dataSet 대용량 일 때)

* classification : LogisticRegression(), LinearSVC(), SGDClassifier()(dataSet 대용량 일 때)

</br>

### 주요 매개변수

알파(regression) :  regularization 강도. 클 수록 regularization 크다, model 단순하다

C(classification) : regularization 강도. 클 수록 regularization 적다, model 단순하다

solver(regression) : "sag", 더 빠르게 학습한다

</br>

### 특징

학습속도 및 예측 빠르다

저차원의 dataSet에서는 다른 model의 일반화 성능이 더 좋다

feature가 많은 dataSet의 경우 성능 매우 우수(특히 sample보다 feature 개수가 더 많은 경우는 완벽하게 예측)

regression의 경우, 보통 Ridge()를 선호하지만 특성이 많고, 그 중 일부 feature만 중요하다면 Lasso()

ElasticNet()은 Ridge()와 Lasso()의 조합. L1, L2 매개변수 정해줘야한다

</br>

</br>





