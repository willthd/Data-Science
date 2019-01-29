# Models

> 각 model의 특징을 이해하고, dataSet에 따라 적합한 model을 찾는 것이 목표

</br>

새로운 dataSet으로 작업할 때는 linear model이나 naibe bayes 또는 knn과 같은 간단한 model로 시작해서 성능이 얼마나 나오는지 가늠해보는 것이 좋다. data를 충분히 이해한 뒤에 random forest나 gradient boosting decision tree, SVM, nn같은 복잡한 model을 고려한다

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

### 장점

이해하기 쉽다

</br>

### 단점

dataSet의 shape(sampe, feature)이 너무 큰 경우 잘 작동하지 않고 느리다

전처리 과정 중요하다

sparse dataSet의 경우 잘 안맞는다

</br>

</br>

## Linear model

regression의 경우, 보통 Ridge()를 선호하지만 특성이 많고, 그 중 일부 feature만 중요하다면 Lasso()

ElasticNet()은 Ridge()와 Lasso()의 조합. L1, L2 매개변수 정해줘야한다

</br>

* regression : linearRegression(), Ridge()(L2), Lasso()(L1), ElasticNet(), SGDRegressor()(dataSet 대용량 일 때)
* classification : LogisticRegression(), LinearSVC(), SGDClassifier()(dataSet 대용량 일 때)

</br>

### 주요 매개변수

alpha(regression) :  regularization 강도. 클 수록 regularization 크다, model 단순하다

C(classification) : regularization 강도. 클 수록 regularization 적다, model 단순하다

solver(regression) : "sag", 더 빠르게 학습한다

</br>

### 장점

첫 번째로 시도할 알고리즘

학습속도 및 예측 빠르다

훈련과정 이해하기 쉽다

대용량 dataSet에서 잘 작동한다

고차원 data에서도 잘 작동한다(특히 sample보다 feature 개수가 더 많은 경우는 완벽하게 예측)

sparse dataSet에서 잘 작동한다

</br>

### 단점

저차원의 dataSet에서는 **다른 model의 일반화 성능이 더 좋다**

</br>

</br>

## Naive Bayes Classifier

분류만 있다

GaussianNB()는 대부분 고차원 dataSet

MNB()는 보통 0이 아닌 feature가 비교적 많은 dataSet에서 BNB()보다 성능이 좋다

classification : GaussianNB()(연속적인 data), BernoulliNB()(이진 data), MultinomialNB()(카운트 data)

</br>

### 주요 매개변수

alpha : regularization 강도. 클 수록 regularization 크다, model 단순하다

</br>

### 장점

훈련과 예측 속도 빠르다

훈련 과정 이해하기 쉽다

linear model보다 훨씬 빠르다 덜 정확하다

대용량 dataSet 가능하다

고차원 dataSet 가능하다

희소한 고차원 dataSet에서 잘 작동한다

비교적 매개변수에 민감하지 않다

</br>

### 단점

linear model보다 덜 정확하다(일반화 성능 조금 뒤진다)

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

### 장점

매우 빠르다

시각화 수월하다

data scaling에 구애받지 않는다. 정규화나 표준화 같은 전처리 과정 필요없다

</br>

### 단점

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

### 장점

decision tree보다 거의 항상 좋은 성능을 낸다

hyperparameter tuning을 많이 하지 않아도 된다

data scaling에 구애받지 않는다. 정규화나 표준화 같은 전처리 과정 필요없다

매우 큰 dataSet에서도 잘 작동한다

</br>

### 단점

의사 결정 과정을 간소하게 표현할 경우 Decision Tree가 더 낫다

고차원 희소 dataSet에서 잘 작동하지 **않는다**

linear model보다 많은 메모리를 사용하며 훈련과 예측이 보다 느리다

</br>

</br>

## Gradient Boosting Tree(Decision Tree Ensemble)

이전에 만든 tree의 예측과 타깃값 사이의 오차를 줄이는 방향으로 새로운 tree를 추가하는 알고리즘. 이를 위해 손실 함수를 정의하고 경사 하강법을 사용해 다음에 추가될 tree가 예측해야 할 값을 보정해나간다

랜덤포레스트와 달리 무작위성이 없다. 대신 강력한 사전 가지치기가 사용된다. 보통 하나에서 다섯 정도의 깊지 않은 tree를 사용하므로 메모리를 적게 사용하고 예측도 빠르다. 이런 얕은 tree같은 간단한 model(weak learner)을 많이 연결하는 것이 근본 아이디어이며, tree는 많을 수록 좋다

</br>

* regression : GradientBoostingRegressor()

* classification : GradientBoostingClassifier() 

</br>

### 주요 매개변수

n_estimator : 트리 개수. n_estimator의 크기 커지면 model 복잡해지기 때문에 정확도는 높아질 수 있지만 overfitting될 가능성 있다

learning_rate : 이전 tree의 오차를 얼마나 강하게 보정할 것인지 제어. 낮추면 비슷한 복잡도의 모델을 만들기 위해서 더 많은 tree를 추가한다. 따라서 n_estimator와 연관성이 크다

일반적으로 시간과 메모리 한도내에서 n_estimator를 맞추고, 적절한 learning_rate찾는다

max_depth : 일반적으로 작게 설정하며, tree의 깊이가 **5**보다 깊어지지 않게 한다

</br>

### 장점

</br>

random forest보다 성능이 좋다

random forest보다 메모리를 적게 사용하고 예측도 빠르다 

data scaling에 구애받지 않는다. 정규화나 표준화 같은 전처리 과정 필요없다

### 단점

random forest보다 hyperparameter tuning에 좀 더 민감하다

random forest보다 학습시간은 더 길다

희소한 고차원 데이터에는 잘 작동하지 않는다

</br>

</br>

## SVM(kernerlized Support Vector Machines)

각 support vector와의 거리를 측정해 margin을 넓히는 방식. support vector의 중요도는 훈련 과정에서 학습한다

</br>

* regression : SVR()
* classification : SVC()

### 주요 매개변수

gamma : 가우시안 커널 폭의 역수. 훈련 샘플이 미치는 영향의 범위를 결정(분류되는 영역). 작은 값은 넓은 영역을 뜻하며 큰 값이라면 영향이 미치는 범위가 제한적이기 때문에 값이 커지면 model이 그만큼 복잡해짐

C : 각 포인트의 중요도를 제한. 값이 클수록 model이 복잡해짐

</br>

### 장점

feature가 적어도 복잡한 결정 경계를 만들 수 있다

저차원, 고차원 dataSet 모두 잘 작동한다

</br>

### 단점

hyperparameter tuning에 민감하다

data scaling에 민감하다. 특히 입력 feature의 범위가 비슷해야 한다. 따라서 0과 1사이의 값으로 맞추는 방법을 주로 사용한다(MinMaxScaler())

sample이 많을 경우 잘 작동하지 않는다. 시간과 메모리 관점에서도 힘들다

</br>

</br>

## Deep Learning

충분히 overfintting되어 문제를 해결할 수 있는 큰 model을 설계한다. 이후 다음 훈련 data가 충분히 학습될 수 있다고 생각될 때 신경망 구조를 줄이거나 규제 강화를 위해 alpha 값을 증가시켜 일반화 성능을 향상시킨다

매끄러운 결정 경계를 원한다면 hidden layer output을 늘리거나, hidden layer를 늘리거나, tanh 함수를 사용할 수 있다

weight 초기값에 따라 model이 크게 달라질 수 있다

</br>

* regression : MLPRegressor(), Dense()
* classification : MLPClassifier(), Dense()

</br>

### 주요 매개변수

hidden_layer_sizes : hidden layer의 개수와 각 layer의 output. 보통 입력 feature의 수와 비슷하게 설정. 수천 초중반을 넘는 일은 거의 없다

activation : hidden layer의 활성화 함수

alpha : 클 수록 regularization(L2 페널티) 심하다. 기본값은 매우 낮다

random_state : weight 초기값 설정

max_iter : warning 뜨면 늘려줘야한다. default는 200

solver : 매개변수 학습에 사용하는 알고리즘 지정. default는 adam

</br>

### 장점

대량의 data에 내재된 정보를 잡아내고 매우 복잡한 model을 만들 수 있다

<br>

### 단점

hyperparameter tuning에 민감하다

data scaling 영향 크다. 평균은 0, 분산은 1이 되도록 변형하는 것이 좋다

큰 model은 학습시간이 오래 걸린다

