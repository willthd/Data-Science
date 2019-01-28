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

sparse dataSet의 경우 잘 안맞는다

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

매우 큰 dataSet에서 잘 작동한다

희소한 dataSet에서 잘 작동한다

저차원의 dataSet에서는 **다른 model의 일반화 성능이 더 좋다**

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

희소한 고차원 dataSet에서 잘 작동한다

비교적 매개변수에 민감하지 않다

선형 분류보다 훈련 속도가 빠른 편이지만, 일반화 성능은 조금 뒤진다

GaussianNB()는 대부분 고차원 dataSet

MNB()는 보통 0이 아닌 feature가 비교적 많은 dataSet에서 BNB()보다 성능이 좋다

</br>

</br>

## Decision Tree

Decision Tree를 학습한다는 것은 정답에 가장 빨리 도달하는 yes or no의 질문 목록(test)을 학습하는 것

완전한 tree : 모든 leaf node가 순수 node인 tree(train dataSet 100% 정확도. overfitting)

</br>

* regression : DecisionTreeRegressor()(최종 영역의 타깃값의 평균값을 예측 결과로 한다)
* classifiacaton : DecisionTreeClassifier()(최종 영역의 타깃값 중 다수인 것을 예측 결과로 한다)

</br>

### 주요 매개변수

max_depth : 최대 연속된 질문 목록(test). default는 무한정 깊어질 수 있다.(max_depth, max_leaf_nodes, min_samples_leaf 중 하나만 지정해도 overfitting을 막는데 충분하다)

min_samples_leaf : leaf node가 되기 위한 최소한의 sample개수

min_samples_split : 매개변수를 사용해 node가 분기할 수 있는 최소 sample 개수

</br>

### 특징

시각화 수월하다

data scaling에 구애받지 않는다. 정규화나 표준화 같은 전처리 과정 필요없다

overfitting 가능성 다분하다 -> Decision Tree Ensemble

</br>

</br>

## Random Forest(Decision Tree Ensemble)

잘 작동하되 서로 다른 방향으로 overfit된 tree를 많이 만들어 그 결과의 평균을 사용해 overfitting줄인다. tree를 만들 땐 random하게 많이 만든다

Decision Tree는 각 node에서 전체 특성을 대상으로 최선의 test를 찾지만, Random Forest는 알고리즘이 각 node에서 후보 feature를 무작위로 선택한 후(max_features로 제한) 이 후보들 중에서 최선의 test를 찾는다

tree를 만들기 위해 먼저 data의 bootstrap sample을 생성한다. 즉 n_samples개의 data 포인트 중에서 무작위로 data를 n_samples 횟수만큼 반복 추출한다(중복 가능)

bootstrap sampling은 tree가 조금씩 다른 dataSet을 이용해 만들어지도록 하고, tree의 각 분기는 다른 feature 부분 집합을 사용해 모든 tree가 서로 달라지도록 돕는다

</br>

* regression : RandomForestRegressor()
* classifier : RandomForestClassifier()

</br>

### 주요 매개변수

n_estimators : 생성할 tree의 개수. 이 값이 크면 tree들은 서로 매우 비슷해지고, 낮추면 서로 많이 달라지며 깊이가 깊어진다. 클 수록 좋다

n_jobs : 사용할 core 개수(-1 모든 core, default값은 1)

max_features : feature 개수 제한. 작을 수록 overfitting 줄어든다. 기본값을 사용하는 것이 좋다(regressiong의 경우 max_features = n_features, classification의 경우 max_features = sqrt(n_features))

</br>

### 특징

hyperparameter tuning을 많이 하지 않아도 된다

data scaling에 구애받지 않는다. 정규화나 표준화 같은 전처리 과정 필요없다

의사 결정 과정을 간소하게 표현할 경우 Decision Tree가 더 낫다

차원이 높고, 희소한 dataSet에서 잘 작동하지 **않는다**

매우 큰 dataSet에서도 잘 작동한다

linear model보다 많은 메모리를 사용하며 훈련과 예측이 보다 느리다

</br>

</br>

## Gradient Boosting Tree(Decision Tree Ensemble)

