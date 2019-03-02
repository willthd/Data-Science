#  and others

> 자주 쓰는 code 또는 필요한 개념들. 잡다하게 있음

</br>

</br>

대부분의 model은 각 feature가(회귀에서는 target도) 정규분포와 비슷할 때 최고의 성능을 낸다. 확률적 요소를 가진 많은 알고리즘의 이론이 정규분포를 근간으로 하고 있기 때문이다

</br>

### 대회 초반에는 cs score이 leader board score와 어느정도 linear하게 움직이는지를 잘 살펴보아야 한다. lbs에 overfit되지 않게 조심

</br>

### DataSet이 크다?

sample이 크거나, feature가 많다

feature가 3개면 3차원 dataSet

</br>

### dataframe에서 한 컬럼만 부르면 그것의 type은 series

</br>

### series -> dataframe

```python
df = trn_series.to_frame(trn_series.name)
```

</br>

### Feature Importance

이 값은 0과 1사이의 숫자로, 각 feature에 대해 0은 전혀 사용되지 않았다는 뜻이며, 1은 완벽하게 target class를 예측했다는 뜻이다. 값이 낮다고 해서 이 특성이 유용하지 않다는 뜻은 아니다. 단지 트리가 그 특성을 선택하지 않았을 뿐이며 다른 특성이 동일한 정보를 지니고 있어서일 수 있다.

</br>

### 높은 확률로로 test, train validation score 나왔지만 비슷하다

둘의 score값이 비슷하다면, 높은 확률이어도 underfitting되었을 수 있다

</br>

### Boosting Algorithms

GBM, adaboost, XGboost, light GBM, catboost (순서)

XGboost는 categorical features를 따로 one-hot-encoding 해야 하지만, light BGM, catboost는 그럴 필요 없다

**catboost, lightGBM, XGboost 비교**

https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

</br>

</br>

### Histogram, Distplot, Barplot, Countplot

**Histogram, Distplot은 coutinuous feature 일때**

* Histogram은 y축이 count

```python
# seaborn
# null있으면 안된다
sns.distplot(train["target"], kde=False, bins=200)
# null있어도 된다
sns.kdeplot(train["taget"])

# pyplot
# bins = 막대 개수
plt.hist(train["target"], bins=200)
# train["tageret"].hist(bins=200) 가능
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
```

* displot은 y축이 비율



**Barplot, Countplot은 categorical feature 일 때**

* Barplot은 y축 설정 해줘야 한다. 편차 표시 있음
* Countplot은 y축이 count

</br>

### factorplot과 pointplot의 차이

f는 hue describe가 그래프 밖

p는 hue describe가 그래프 안

f가 더 낫다

</br>

### warning message "ignore"일 경우, 없애기

```python
warnings.filterwarnings("ignore")
```

</br>

### method 설명보기

```python
# helf(method)
help(method)
```

</br>

### jupyter notebook에서 라이브러리내 method 확인하기

```python
# method??
mglearn.plots.plot_knn_classification??
```

</br>

### np.random

```python
# np.random.seed(x)
# seed를 사람이 수동으로 설정한다면 그 다음에 만들어지는 난수들은 예측할 수 있다.
# np.random.rand(x)
# 0부터 1사이의 균일 분포에서 x개 난수 생성
# np.random.shuffle(x)
# 배열 값 shuffle
np.random.seed(9)
np.random.rand(5)
# array([0.01037415, 0.50187459, 0.49577329, 0.13382953, 0.14211109])
np.random.rand(5)
# array([0.21855868, 0.41850818, 0.24810117, 0.08405965, 0.34549864])
np.random.seed(10)
np.random.rand(5)
# array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701])
np.random.seed(9)
np.random.rand(5)
# array([0.01037415, 0.50187459, 0.49577329, 0.13382953, 0.14211109])
```

</br>

### 임의 데이터 프레임 만들기

```python
# 방법 1
df = pd.DataFrame([["11","2", "6"], ["12","4", "2"], ["13","3", "4"]], columns=["a","b", "c"])

# 방법 2
df = pd.DataFrame({
          'A':['a','b','a'],
          'B':['b','a','c']
        })
```

</br>

### Index column 설정

```python
# 방법 1
train = pd.read_csv("./train.csv", index_col="PassengerId")

# 방법 2
# ()안에 inplace = True, 설정하면 train = 없어도 된다
train = train.set_index("col_name")
```

</br>

### 특정 column "Y"->1, "N"->0 으로 변경

```python
df["col_name"] = df["col_name"].map({'Y':1, 'N':0})
```

</br>

### column 이름 변경

```python
df.rename(columns={"a" : "c", "b" : "d"}, inplace=True)
```

</br>

### dataFrame, 컬럼 기준으로 정렬하기

```python
count_list = count_list.sort_values(by="a", ascending=False)
```

</br>

### null값 채우기

```python
train["Age"].fillna(0, inplace=True)
```

</br>

### 백분위 수를 이용해 data 찾기

```python
# 50%에 위치하는 data, 중앙값
np.percentile(train["col_name"], 50)
```

<br>

### 컬럼 지우기

```python
del train["col_name"]
```

</br>

### list -> dataFrame

```python
count_list = pd.DataFrame.from_dict(count_list)
```

</br>

### dataFrame에 index 입력하기

```python
count_list.index.name = "store_id"
```

</br>

### 시계열 data 읽어오면서 date 인식. 연도, 월, 날짜, 시간, 분, 초 컬럼 생성

```python
# 방법 1
train = pd.read_csv("data/bike/test.csv", parse_dates=["datetime"])

# 방법 2
train['first_active_month'] = pd.to_datetime(train['first_active_month'])

train["datetime-year"] = train["datetime"].dt.year
train["datetime-month"] = train["datetime"].dt.month
train["datetime-day"] = train["datetime"].dt.day
train["datetime-hour"] = train["datetime"].dt.hour
train["datetime-minute"] = train["datetime"].dt.minute
train["datetime-second"] = train["datetime"].dt.second
```

</br>

### Lambda 에서 elif 사용하고 싶을 때, else 여러번

```python
train["Pclass"] = train["Pclass"].apply(lambda x: "A" if x == 1 else ("B" if x == 2 else "C"))
```

</br>

### one-hot-encoding 한번에

```python
# 특정 column만
train = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])

# 범주형 컬럼 중 값이 문자열로 되어진 것만 전부 바꿔준다. 값이 숫자로 되어 있는 범주형 컬럼은 위와 같이 컬럼명을 따로 명시해줘야하거나, 그 값을 str으로 변환한 후 get_dummies()함수를 사용해야 한다
train = pd.get_dummies(train)
```

</br>

### 시계열 data를 split해서 연도, 월, 날짜 관련 컬럼을 생성

```python
def split_date(date):
    return date.split("-")
```

```python
train["year"], train["month"], train["day"] = zip(*train['date'].apply(lambda x: split_date(x)))

# zip([1, 1, 1], [2, 4, 8], [3, 9, 27]) -> (1, 2, 3), (1, 4, 9), (1, 8, 27)
# zip(*[[1, 2, 3], [2, 3, 4], [3, 4, 5]]) -> (1, 2, 3), (1, 4, 9), (1, 8, 27)
# *은 unpack을 의미한다
# .apply(lambda x : 함수)
```

</br>

### string 형태의 숫자를 int로 바꿀 때

```python
# train["year"] = int(train["year"])로 하면 안된다
# error message : cannot convert the series to <class 'int'>

train["year"] = train["year"].astype(int)
```

</br>

### 그래프 모아서 보기

```python
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 18))
# fplt.figure(figsize=(10, 7))
sns.barplot(data=train, x="year", y="total", ax=ax1)
sns.barplot(data=train, x="month", y="total", ax=ax2)
```

</br>

### 그래프 사이즈 정하기

```python
plt.figure(figsize=(10, 7))
```

</br>

### train["column"] 값 array로 만들기

```python
my_array = train["column"].values
```

</br>

### array 내에 중복되는 값 없애기

```python
# list에는 .unique() 함수 없다
# 아래 처럼 하면 값 순서대로 정렬되서 나온다
uniqueVals = np.unique(my_array)

# 바로 컬럼에 적용해서 사용할 수도 있다
train["column"].unique()
```

</br>

### array -> list

```python
my_list = uniqueVals.tolist()
```

</br>

### case : "nextday_date"컬럼 값이 holiday_date_list에 포함 되어 있는 경우 "nextday_holiday"를 1로 채워넣기

```python
# 방법 1
idx = [idx for idx, value in enumerate(train["nextday_date"]) if value in holiday_date_list]

# enumerate는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다
# idx 리스트는 결국 해당 조건이 맞는 경우의 idx들의 모음 의미한다

train.loc[idx, "nextday_holiday"] = 1
```

```python
# 방법 2
train.loc[train["nextday_date"].isin(holiday_date_list), "nextday_holiday"] = 1
train.loc[train["nextday_holiday"].isnull(), "nextday_holiday"] = 0

# 방법 3
train["nextday_holiday"] = train["nextday_date"].isin(holidat_date_list).astype(int)

###############
# 방법 4, 가장 좋다
train["nextday_holiday"] = train["nextday_date"].
(lambda x : 1 if x in holiday_date_list else 0)
```

</br>

### "Name" 컬럼 내 해당 문자열 존재하는 경우, "Title" 컬럼을 "Mr"로 채워 두기

```python
train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"
```

</br>

### column내 문자열 변경

```python
train["Name"] = train["Name"].str.replace("Mr", "아저씨")

train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
```

</br>

### 컬럼에서 숫자만 추출할 경우

```python
# 컬럼 값이 문자와 숫자로 혼합되어 있는 경우 통일 필요할 때
# "(\d+)" 정규표현식 사용
# ex) 31세, 31 세, 31 -> 31로 통일, NaN -> 0

################
# 방법 1, 가장 좋다
train.loc[train["Age"].isnull(), "Age"] = 0
# 미리 str으로 전부 바꿔줘야한다. 아주 중요 !
train["Age"] = train["Age"].astype(str)
train["Age"] = train["Age"].str.extract("(\d+)").astype(int)

# 방법 2
train.loc[train["Age"].isnull(), "Age"] = "0"
train.loc[train["Age"].str.contains(" 세"), "Age"] = train["Age"].str.replace(" 세", "")
train.loc[train["Age"].str.contains("세"), "Age"] = train["Age"].str.replace("세", "")
train["train"] = train.["Age"].astype(float)
train["train"] = train.["Age"].astype(int)


# 방법 3
# 22.0은 220으로 바뀐다는 점 주의. 정수 표현일 때 사용
# 1과의 차이는 1의 경우 앞의 숫자만 나옴. 예를 들어 7,000,000 이 str으로 저장되어 있을 경우 7만 추출
# 3의 경우, 7000000 모두 추출
train['Age'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
```

</br>

### 컬럼 중 max, min, mean

```python
train["Age"].max()
train["Age"].min()
train["Age"].mean()
```

</br>

### python, lambda

https://wikidocs.net/64 참고

map(), reduce(), filter()에 어떻게 적용하는지

</br>

### groupby()

그룹별 통계

https://datascienceschool.net/view-notebook/76dcd63bba2c4959af15bec41b197e7c/

아주 잘 설명되어 있다

</br>

### index 없애서 일반 colum으로 변경

```python
train.reset_index(inplace=True)
```

</br>

### 전체 data에서 train, test data 분류

```python
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0, test_size=0.2)
```

</br>

### partial_fit()

전체 data를 메모리에 모두 적재할 수 없을 때난 fit() 메서드 대신에 학습된 것을 유지하면서 반복하여 학습할 수 있는 partial_fit() 메서드를 사용한다. 이는 mini-batch를 사용해 model을 점진적으로 학습시킨 경우와 유사하다

</br>

### 범주형 데이터 문자열 확인하기

```python
train["column"].value_counts()
```

</br>

### column indexing

```python
# numpy는 마지막 원소를 포함하지 않지만, pandas는 포함한다
# a컬럼에서 b컬럼까지
X_train = train.loc[:, "a":"b"]
```

</br>

### 구간분할(binding, 이산화)

```python
# -3부터 3까지 11개의 point로 분할. 구간은 10
bins = np.linspace(-3, 3, 11)

# X를 각 구간에 맞게 1~10으로 표시
which_bin = np.digitize(X, bins=bins)
```

</br>

### reshape()

원소의 개수가 n개 일 때(차원은 중요하지 않다), 이를 원하는 shape으로 변형한다

reshape(-1, 1) : 열의 개수를 1로 유지하면서 행을 그에 맞게 변형한다 -> shape은 (n, 1)

reshape(1, -1) : 행의 개수를 1로 유지하면서 열을 그에 맞게 변형한다 -> shape은 (1, n)

reshape(-1) : 하나의 행렬로 존재한다 -> shape : (n, )

</br>

### Feature의 종류

**categorical**

ex) Sex - Male, Female

-> ont hot encoding, frequency encoding, mean encoding

barplot

**ordinal**

ex) Height - Tall, Medium, Short

-> label encoding

**continuous**

ex) Age

histogram, scatterplot, distplot

</br>

### crossTab

```python
pd.crosstab([train["Sex"], train["Survived"]], train["Pclass"], margins=True).style.background_gradient(cmap='summer_r')
```

</br>

### subplot에서 그래프 불필요하게 더 나올 때 제거하기

```python
# 그래프 번호. 0부터 시작
plt.close(2)
```

</br>

### pandas.qcut()

```python
# 샘플 수를 비슷하게 맞춰준다
train["Fare_Range"] = pd.qcut(train["Fare"], 4)
```

</br>

### pandas.cut()

```python
# range를 일정하게 해서 4등분
train["Fare_Range"] = pd.qcut(train["Fare"], 4)
```

</br>

### drop()

```python
train.drop(["Name", "Age", "Ticket", "Fare", "Cabin", "Fare_Range", "PassengerId"], axis=1, inplace=True)
```

</br>

### LabelEncoder()

```python
# String -> 숫자로 변경
from sklearn.preprocessing import LabelEncoder

for col in ["Sex", "Embarked", "Initial"]:
    train[col] = LabelEncoder.fit_transform(train[col])
```

```python
# factorize도 label encoding 하는 것. label encoder보다 빠르다
indexer = {}

for col in ["Sex", "Embarked", "Initial"]:
    _, indexer[col] = pd.factorize(train[col])
    train[col] = indexer[col].get_indexer(train[col])
```

</br>

### sns font scale

```python
sns.set(font_scale=4)
```

</br>

### voting : soft vs hard

![voting](./voting.jpg)

</br>

### variance & bias

![vb](./vb.jpg)

</br>

### feature importance

일반적으로 RandomForest를 이용해서 확인한다. 많은 subset과 model을 이용하기 때문?

아님 여러 model을 사용해서 평균을 낼 수도 있다

</br>

### imputer

```python
# null data 처리
from sklearn.preprocessing import Imputer
```

```python
# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
```

```python
# ravel()은 shape 변경해주는 것 (8000, 1) -> (8000,), reshape(-1)과 비슷하다고 보면 된다
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
```

</br>

### debug때는 data set 전부 읽을 필요 없다

**방법 1**

```python
DEBUG = True

if DEBUG:
	NROWS=10000
else:
	NROWS=None
```

```python
%%time
train = pd.read_csv("../input/train.csv", nrows=NROWS)
test = pd.read_csv("../input/test.csv", nrows=NROWS)
```

잘 되면 DEBUG 값만 False로 변경 후 전체 돌린다

```python
DEBUG = False
```

</br>

**방법 2**

```python
DEBUG = True

if DEBUG:
	FRAC=0.2
else:
	FRAC=None
```

```python
train = pd.read_csv("../input/train.csv")
train = pd.read_csv("../input/train.csv")
# local 환경에 맞게 비율 설정한다. random으로 추출
train = train.sample(frac=FRAC)
test = test.sample(frac=FRAC)
```

마찬가지로 잘 되면 DEBUG 값만 False로 변경 후 전체 돌린다

```python
DEBUG = False
```

</br>

**방법 3**

imbalanced data set의 경우, frac도 괜찮지만 더 정확하게 하고 싶다면

```python
from sklearn.model_selection import StratifiedKFold
```

```python
fold = StratifiedKFold(n_splits=10, random_state=666)
```

```python
# val_idx는 target idx말하는 것
for train_idx, val_idx in fold.split(train, train["target"]):
    # 딱 한번만 하겠다
	break
```

```python
train = train.loc[train_idx]
```

</br>

### categorical 컬럼 값 몇 종류 있는지 확인

```python
col_list = categorical 컬럼 리스트
for col in col_list:
	print(train[col].nunique())
```

</br>

### 중복된 row 없애기

```python
train.drop_duplicates()
```

</br>

### 정보 보기

꼭 시작 전에 해봐라

```python
train.info()
```

</br>

### How to sample Imbalanced DataSet

1. Oversampling
2. Undersampling
3. SMOTE

library 존재 -> imblearn

![imb](./imbalanced.jpg)

</br>

### assert

```python
# assert 이후가 맞으면 넘어가고, 틀리면 error 발생
assert len(trn_series) == len(target)
```

</br>

### mean encoding

특정 컬럼의 각 속성값과 대응하는 타겟값의 곱을 전부 더한 뒤 속성값과 그 속성값의 개수를 곱해 나눠준다

**ex)**

![mean1](./mean1.jpg) ————> ![mean2](./mean2.jpg)



overfitting 일어날 가능성 높다. 이를 막기 위해 noise를 추가하기도 한다. 일종의 Regularization

참조 링크

https://medium.com/datadriveninvestor/improve-your-classification-models-using-mean-target-encoding-a3d573df31e8

https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study



</br>

### fequency encoding

https://www.kaggle.com/youhanlee/which-encoding-is-good-for-time-validation-1-4417

```python
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')
```