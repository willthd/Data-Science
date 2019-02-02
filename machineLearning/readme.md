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

범주형 변수가 정수로 표기되어 있는 경우 연속형 변수로 해석한다. 따라서 이 경우 one-hot-encodig을 사용한다

구간 분할이나 다항식과 상호작용 특성을 새로 추가해 큰 이득을 볼 수 있다

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

tree model은 feature의 순서에도 영향을 받는다

tree model은 훈련 세트에 있는 특성의 범위 밖으로 외삽(extrapolation)할 능력이 없다. 이런 경우 테스트 세트와 가자아 가까이 있는 마지막 훈련 세트 데이터의 타깃값을 예측으로 사용한다(예를 들어 시계열 data처럼 훈련 세트는 yy/mm/dd까지의 data이고, 테스트 세트는 그 이후의 경우)

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

max_features : feature 개수 제한. 작을 수록 overfitting 줄어든다. 기본값을 사용하는 것이 좋다(regressiong의 경우 max_features = n_features, classification의 경우 max_features = sqrt(n_features). decision tree나 gradient boosting tree의 경우 기본값 "None")

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

</br>

</br>

# 전처리

**StandardScaler()** : 각 feature의 평균을 0, 분산을 1로 변경해 모든 feature가 같은 크기를 가지게 한다

**RobustScaler()** : StandardScaler()와 비슷하지만 평균을 중간값, 분산을 사분위 값을 사용한다. outlier에 영향을 받지 않게 한다

**MinMaxScaler()** : 모든 feature가 정확하게 0과 1 사이에 위치하도록 데이터를 변경한다. X_test의 경우 X_train으로 fit된 scaler를 이용해 transform하기 때문에 0과 1 사이에 위치하지 않을 수 있다

**Normalizer()** : feature 벡터의 유클리디안 길이가 1이 되도록 데이터 포인트를 조정한다. 다른 scaler는 feature의 통계치를 이용하지만 normalizer는 sample마다 각기 정규화 된다

</br>

</br>

# 비지도 학습

> 출력값이나 정보 없이 학습 알고리즘을 가르쳐야 하는 모든 종류의 머신러닝

</br>

## 비지도 변환

>  데이터를 새롭게 표현해 사람이나 다른 머신러닝 알고리즘이 원래 데이터보다 쉽게 해석할 수 있도록 만드는 알고리즘. 시각화, data 압축, 지도 학습 등을 위해 정보가 더 잘 드러나는 feature를 찾기 위한 것

</br>

### PCA(주성분 분석) - 차원 축소

데이터의 분산이 가장 큰 방향을 찾는다

feature들의 선형 결합을 통해 feature들이 가지고 있는 전체 정보를 최대한 설명할 수 있는 서로 독립적인 새로운 feature(주성분)를 유도하여 해석하는 방법

산점도로 시각화 할 수 있다

feature의 scale값이 서로 다르면 올바른 주성분 방향을 찾을 수 없다. 따라서 PCA를 사용할 때는 StandardScaler()를 feature의 분산이 1이 되도록 data의 스케일을 조정한다

일반적으로 원본 특성 개수만큼의 주성분이 있다

PCA의 주성분의 특성 개수는 항상 입력 데이터의 특성 개수(차원)와 같다

```python
from sklearn.decomposition import PCA
# 데이터의 처음 두 개 주성분만 유지시킵니다
pca = PCA(n_components=2)
# 유방암 데이터로 PCA 모델을 만듭니다
pca.fit(X_scaled)

# 처음 두 개의 주성분을 사용해 데이터를 변환합니다
X_pca = pca.transform(X_scaled)
print("원본 데이터 형태: {}".format(str(X_scaled.shape)))
# (596, 30)
print("축소된 데이터 형태: {}".format(str(X_pca.shape)))
# (569, 2)

# 주성분은 아래에 저장되어 있다
pca.components_
print("축소된 데이터 형태: {}".format(str(pca.components_.shape)))
# (2, 30), 원본 데이터의 feature수와 주성분의 feature 수는 동일하다
```

</br>

### NMF(non-negative matrix factorization, 비음수 행렬 분해) - 차원 축소, 특성 추출

PCA에서는 데이터의 분산이 가장 크고 수직인 성분을 찾았다면 NMF에서는 음수가 아닌 성분과 계수 값을 찾는다. 즉, 주성분과 계수가 모두 0보다 크거나 같아야 한다. 따라서 음수가 아닌 feature를 가진 데이터에만 적용할 수 있다

데이터를 인코딩하거나 재구성하는 용도로 사용하기보다는 주로 데이터에 있는 유용한 패턴을 추출하는데 사용된다(특히 소리, 유전자 표현, 텍스트처럼 덧붙이는 구조를 가진 데이터)

</br>

### t-SNE(t-distributed stochastic neighbor embedding) - manifold learning

데이터 포인트 사이의 거리를 가장 잘 보존하는 2차원 표현을 찾는 것이다

시각화가 목적이기 때문에 3개 이상의 특성을 뽑는 경우는 거의 없다

테스트 세트에는 적용할 수 없고, 단지 훈련했던 데이터만 변환할 수 있다. 따라서 manifold learning은 탐색적 데이터 분석에 유용하지만 지도 학습용으로는 거의 사용하지 않는다

trainsform() 메소드가 따로 없으므로 fit_transform()을 사용한다

</br>

## 군집

> 데이터를 비슷한 것끼리 그룹으로 묶는 것

</br>

### k-means 군집

데이터 포인트를 가장 가까운 클러스터 중심에 할당하고, 그런 다음 클러스터에 할당된 데이터 포인트의 평균으로 클러스터 중심을 다시 지정한다. 클러스터에 할당되는 데이터 포인트에 변화가 없을 때 알고리즘이 종료된다

n_clusters 꼭 지정한다(몇 개의 cluster로 구분할 것인지)

k-means 알고리즘은 클러스터에서 모든 방향이 똑같이 중요하다고 가정한다. 따라서 그룹들이 대각선으로 늘어서 있는 경우, 가장 가까운 클러스터 중심까지의 거리만 고려하기 때문에 이런 데이터를 잘 처리하지 못한다

k-means 알고리즘을 사용한 벡터 양자화의 흥미로운 면은 입력 데이터의 차원보다 더 많은 클러스터를 사용해 데이터를 인코딩 할 수 있다는 점이다

단점으로는 무작위 초기화를 사용해 알고리즘의 출력이 난수 초깃값에 따라 달라진다는 점이다. 또한 클러스터의 모양을 가정하고 있어서 활용 범위가 비교적 제한적이며, 또 찾으려 하는 클러스터의 개수를 지정해야만 한다

</br>

### 병합 군집(agglomerative clustering)

병합 군집 알고리즘은 시자갛ㄹ 때 각 포인트를 하나의 클러스터로 지정하고, 그 다음 어떤 종료 조건을 만족할 때까지 가장 비슷한 두 클러스터를 합쳐나간다. sklearn에서 사용하는 종료 조건은 클러스터 개수로, 지정된 개수의 클러스터가 남을 때까지 비슷한 클러스터를 합친다

</br>

### DBSCAN

군집 알고리즘은 전체적으로 자세히 정리하지 않음

</br>

</br>

# 데이터 표현과 feature engineering

선형 모델은 구간 분하링나 다항식과 상호작용 특성을 새로 추가해 큰 이득을 볼수 있다. 반면에 랜덤 포레스트나 SVM같은 비선형 모델은 특성을 늘지 않고서도 복잡한 문제를 학습할 수 있다. 실제로 feature와 모델의 궁합이 가장 중요하다

### one-hot-encoding

pandas의 get_dummies()를 이용해 만들 수 있다

주의 할 점은, train과 test의 feature column순서가 동일해야 한다는 점이다. 따라서 train과 test가 합쳐진 전체 dataSet에서 get_dummies()를 이용해 one-hot-encoding을 처리하던지, 따로 할 경우 feature column의 순서가 동일한지 확인해봐야 한다

column값이 숫자인 범주형 feature는 그 값을 str으로 변환 후, get_dummies()를 활용하거나, get_dummies()함수에 컬럼명을 따로 명시해줘야 한다

</br>

### 구간 분할(binding, 이산화)

용량이 매우 크고 고차원 데이터셋이라 **선형 모델**을 사용해야 한다면 연속형 데이터를 구간 분할하는 것이 모델 성능을 높이는 데 아주 좋은 방법이다

</br>

### 상호작용과 다항식

특별히 feature를 풍부하게 나타내는 또 하나의 방법은 원본 데이터에 상호작용(interaction)과 다항식을 추가하는 것이다

ex)

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    random_state=0)

# 데이터 스케일 조정
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled, y_train)
print("상호작용 특성이 없을 때 점수: {:.3f}".format(ridge.score(X_test_scaled, y_test)))
# 0.621
ridge = Ridge().fit(X_train_poly, y_train)
print("상호작용 특성이 있을 때 점수: {:.3f}".format(ridge.score(X_test_poly, y_test)))
# 0.753

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled, y_train)
print("상호작용 특성이 없을 때 점수: {:.3f}".format(rf.score(X_test_scaled, y_test)))
# 0.795
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_poly, y_train)
print("상호작용 특성이 있을 때 점수: {:.3f}".format(rf.score(X_test_poly, y_test)))
# 0.773
```

random forest 같이 더 복잡한 model을 사용하면 결과는 다르다. feature를 추가하지 않아도 Ridge보다 성능이 좋다. 오히려 feature를 추가한 경우 성능이 조금 줄어든다. 따라서 항상 feature추가가 더 좋은 결론을 만드는 것은 아니다

</br>

### 일변량 비선형 변환

**tree기반 모델은 feature의 순서에만 영향을 받지만**, 선형 모델과 신경망은 각 특성의 스케일과 분포에 밀접하게 연관되어 있기 때문에 비선형 변환이 유용하다

대부분의 모델은 각 feature가(회귀에서는 target도) 정규분포와 비슷할 때 최고의 성능을 낸다. 확률적 요소를 가진 많은 알고리즘의 이론이 정규분포를 근간으로 하고 있기 때문이다

tree기반 model은 스스로 중요한 상호작용을 찾아낼 수 있고 대부분의 경우 데이터를 명시적으로 변환하지 않아도 된다

sin(), cos() 함수는 주기적인 패턴이 들어 있는 데이터를 다룰 때 편리하다

카운트(예를 들어 주문 횟수)를 예측하는 경우가 전형적인 예로 log(y + 1)를 사용해 변환하면 도움이 된다

</br>

### 특성 자동 선택

보통 새로운 특성을 추가할 때나 고차원 데이터셋을 사용할 때, 가장 유용한 특성만 선택하고 나머지는 무시해서 특성의 수를 줄이는 것이 좋다. 이렇게 하면 모델이 간단해지고 일반화 성능이 올라간다

feature 중 가장 좋다고 판단되는 feature만 선택한다

결과는 보통 항상 더 좋다고 할 수는 없다. 하지만 feature가 너무 많을 경우 고려해볼 필요 있다

선택된 feature 그래프로 확인할 수 있다

```python
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("faeture_number")
```

#### 일변량 통계(univariate statistics)

개개의 feature와 target 사이에 중요한 통계적 관계가 있는지 계산한다

```python
# %로 선택
from sklearn.feature_selection import SelectPercentile
# 개수로 선택
from sklearn.feature_selection import SeleckKBest

select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_seleceted = select.transform(X_test)

score = LogisticRegression()fit(X_train_selected, y_train).score(X_test_l1, y_test)
```

</br>

#### 모델 기반 선택(model-based selection)

모델을 사용해 feature의 중요도를 평가해서 가장 중요한 feature들만 선택한다

```python
from sklearn.feature_selection import SelectFromModel

# median이니까 feature 절반 가량 선택한다. "1.2*median" 이런식으로 사용할 수 있다
select = SelectFromModel(RandomForestClassfier(n_estimators=100, random_state=42, threshold="median"))
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
X_test_l1 = select.transform(X_test)

score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
```

</br>

#### 반복적 선택(iterative selection)

두 가지 방법 있다. 첫번째는 feature를 하나도 선택하지 않은 상태로 시작해서 어떤 종료 조건에 도달할 때까지 하나씩 추가하는 방법이고, 두번째는 모든 feature를 가지고 시작해서 어떤 종료 조건이 될 때까지 feature를 하나씩 제거해가는 방법이다(RFE, recursive feature elimination, 재귀적 특성 제거). 따라서 앞서 소개한 방법들보다 계산 비용이 훨씬 많이 든다

```python
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
             n_features_to_select=40)
select.fit(X_train, y_train)
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

score = LogisticRegression(solver='liblinear').fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
```

</br>

</br>

# 모델의 평가와 성능 향상

### 교차 검증(cross validation)

sklearn에 있는 cross_cal_score() 함수에서 fold를 나눌 때 기본적으로 분류에는 StratifiedKFold(계층 별 교차 검증)를 사용해 훈련 세트와 테스트 세트를 나누고, 회귀에는 단순한 KFold를 적용한다

</br>

#### k-fold

순서대로 fold 나눠준다. shuffle=true 할 경우 무작위로 섞고, 섞을 때 random_state를 고정해서 똑같은 작업을 재현할 수 있다

```python
from sklearn.model_selection import KFold

kfold = KFold()
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
```

</br>

### stratified k-fold

분류에 사용된다. 데이터셋에 담겨 있는 클래스별 비율에 맞게 fold를 설정해준다.  dataSet이 클래스 0, 1, 2 이렇게 순서대로 있는 경우 학습이 안되는 것을 막기 위함이다. 예를 들어 샘플의 90%가 클래스 A이고, 10%가 클래스 B에 속한다면, 계층별 교차 검증은 각 fold에 클래스 A 샘플이 90%, 클래스 B샘플이 10%가 되도록 만든다

```python
skf = StratifiedKFold()
scores = corss_val_score(logreg, iris.data, iris.target, cv=skf)
```

</br>

#### LOOCV

fold 하나에 샘플 하나만 들어 있는 k-fold cross validation. 각 반복에서 하나의 데이터 포인트를 선택해 테스트 세트로 사용한다. 데이터셋이 작을 경우 더 좋은 결과를 만들기도 한다

```python
loo = LeaveOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo) v
```

</br>

#### 임의 분할 교차 검증

전체 dataSet에서 비율 또는 개수 만큼 train과 test를 임의로 split해 만들고(random_state 고정) n_splits 값 만큼 반복 검증한다. 따라서 특정 data sample이 여러번 포함 될 수 있다(train, test 둘다). 대규모 데이터셋에서 작업할 때 유용

```python
suffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)

# 분류의 경우
StratifiedShuffleSplit() 사용한다
```

</br>

### Group CV

한 그룹 전체가 훈련 세트 아니면 테스트 세트에 있게 분할 하는 방법이다

```python
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
```

