# Models

> 각 model의 특징을 이해하고, dataSet에 따라 적합한 model을 찾는 것이 목표

</br>

</br>

## K-NN(K Nearest Neighbors)

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

feature의 값들이 대부분 0인 경우, 잘 안맞는다

</br>

</br>

## Linear model

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

훈련과정 이해하기 쉽다

저차원의 dataSet에서는 다른 model의 일반화 성능이 더 좋다

feature가 많은 dataSet의 경우 성능 매우 우수(특히 sample보다 feature 개수가 더 많은 경우는 완벽하게 예측)

regression의 경우, 보통 Ridge()를 선호하지만 특성이 많고, 그 중 일부 feature만 중요하다면 Lasso()

ElasticNet()은 Ridge()와 Lasso()의 조합. L1, L2 매개변수 정해줘야한다

</br>

</br>

## Naive Bayes Classifier

classifier : GaussianNB()(연속적인 data), BernoulliNB()(이진 data), MultinomialNB()(카운트 data)

</br>

### 주요 매개변수

알파 : regularization 강도. 클 수록 regularization 크다, model 단순하다

</br>

### 특징

훈련과 예측 속도 빠르다

훈련 과정 이해하기 쉽다

비교적 매개변수에 민감하지 않다

선형 분류보다 훈련 속도가 빠른 편이지만, 일반화 성능은 조금 뒤진다

GaussianNB()는 대부분 고차원 dataSet

MNB()는 보통 0이 아닌 feature가 비교적 많은 dataSet에서 BNB()보다 성능이 좋다

</br>

</br>

## Decision Tree

> Decision Tree를 학습한다는 것은 정답에 가장 빨리 도달하는 yes or no의 질문 목록(test)을 학습하는 것

</br>

완전한 tree : 모든 leaf node가 순수 node인 tree(train dataSet 100% 정확도. overfitting)

</br>

* regression : DecisionTreeRegressor()
* classifiacaton : DecisionTreeClassifier()

</br>

### 주요 매개변수

max_depth : 최대 연속된 질문 목록(test). default는 무한정 깊어질 수 있다.(max_depth, max_leaf_nodes, min_samples_leaf 중 하나만 지정해도 overfitting을 막는데 충분하다)

</br>

</br>

### 특징



## Decision Tree Ensemble



</br>

</br>