#  and others

> 자주 쓰는 code 또는 필요한 개념들. 잡다하게 있음

</br>

</br>

대부분의 model은 각 feature가(회귀에서는 target도) 정규분포와 비슷할 때 최고의 성능을 낸다. 확률적 요소를 가진 많은 알고리즘의 이론이 정규분포를 근간으로 하고 있기 때문이다. 그래서 전처리 과정에서 scaling 하기도 한다. (non-linear. ex- log, exp, squred….)

</br>

### 대회 초반에는 cv score가 leader board score와 어느정도 linear하게 움직이는지를 잘 살펴보면서 cv 시스템을 개선시킨다. 이 과정은 좋은 feature를 선별하는 목적이 가장 크다. lbs에 overfit되지 않게 주의

</br>

### Check data set

```python
# count, mean, std, min, max, 25%, 50%, 75%
train.describe()

# null개수, dtype(dtype은 한 열에 여러개 섞여 있어도 다 나타내지 않고 하나만 나타낸다)
train.info()

# 전체 feature의 type
train.dtypes.value_counts()
```

</br>

### null을 입력하고 싶을 때는 None을 입력한다

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

### 배열 -> series

```python
coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns)
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
import warnings
warnings.filterwarnings("ignore")
```

</br>

### method 설명보기

```python
# helf(method), parameter 설명 상세하게 나온다
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

pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2, 'A', 4], columns=[48, 49, 50])

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

# 각 컬럼 별로 mean 구해서 null을 mean으로 채움
train.fillna(train.mean(), inplace=True)
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

# or
trian['datetime-year'] = train['datetime'].apply(lambda x: x.year)
```

**datetime은 비교 가능하다 <, >, ==**

https://inma.tistory.com/96

</br>

### Lambda 에서 elif 사용하고 싶을 때, else 여러번

row 별로 적용할 때는?

```python
train["Pclass"] = train["Pclass"].apply(lambda x: "A" if x == 1 else ("B" if x == 2 else "C"))
```

</br>

### lambda 에서  여러개 컬럼 사용하고 싶을 때

```python
# 보통 하나의 컬럼만 인자로 뽑아서 사용할 때는 X_mammo['column'].apply(lambda x : )를 진행하지만, 여러개 컬럼 이용할 때는 X_mammo만 적용한다. 하나의 로우를 다 인자로 사용하는 것. 이 때 axis=1 잊지 말기
X_mammo['target'] = X_mammo.apply(lambda x : find_year_after_last(x.id, x.YYYY, df_target), axis=1)
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
    try: return date.split("-")
    except: (None, None, None)
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
# cplt.figure(figsize=(10, 7))
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
train["nextday_holiday"] = train["nextday_date"].isin(holiday_date_list).astype(int)

###############
# 방법 4, 가장 좋다
train["nextday_holiday"] = train["nextday_date"].apply(lambda x : 1 if x in holiday_date_list else 0)
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

### index 설정

```python
train.set_index('A', inplace=True)
```

</br>

### 전체 data에서 train, test data 분류

불균형 데이터셋을 각 클래스별 비율에 맞게 분류한다. StratifiedKFold와의 차이는 train_test_spilit의 경우 1회 시행하는데 반해 StratifiedKFold는 cv 횟수 만큼 train과 test를 분리 한다. 그리고 StratifiedShuffleSplit은 stratified하게 train과 test를 분리하지만, test에 선택된 인덱스가 cv마다 겹칠 수 있다

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

![kindesOfData](./KindsOfData.jpg)

**categorical**

ex) Sex - Male, Female

-> ont hot encoding, frequency encoding, mean encoding

카테고리 종류가 적으면 one hot encoding, 많으면 frequenct, mean encoding을 쓰지만 더 좋은 점수를 만들기 위해서는 하나의 feature에 대해 one hot, frequency, mean 모두 사용하기도 한다

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

### Error : variance + bias

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
# missing_values에 none 입력 불가
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

http://freesearch.pe.kr/archives/4506

일반적으로 Oversampling이 Undersampling이 좋다고 한다.

1. Oversampling
2. Undersampling
3. SMOTE, 일반적으로 재현율은 높아지나, 정밀도는 낮아진다

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



overfitting 일어날 가능성 높다. 이를 막기 위해 noise를 추가하기도 한다(descrete한 걸 부드럽게 바꿔줌). 일종의 Regularization

참조 링크

https://medium.com/datadriveninvestor/improve-your-classification-models-using-mean-target-encoding-a3d573df31e8

https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study

</br>

### fequency encoding

https://www.kaggle.com/youhanlee/which-encoding-is-good-for-time-validation-1-4417

```python
def frequency_encoding(frame, col):
    # 각 클래스를 index로 하여 클래스별로 몇 개 씩 있는지
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    # 위의 것에서 index를 없애고 개수가 있는 컬럼 이름(0)을 변경
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')
```

</br>

### 계층이 여러개인 컬럼에서 sort_values() 적용

```python
# ["1번 계층 컬럼", "2번 계층 컬럼"]
# 오름차순
cat_perc.sort_values(by=["target", "mean"], ascending=False, inplace=True)
```

![sort1](./sort_values1.jpg) —————> ![sort2](./sort_values2.jpg)

</br>

### heatmap

Pearson Correlation Coefficient 값 나오는데, 이것만 맹신할 순 없다. 꼭 두 feature간의 시각화 해봐야 한다.

```python
sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
```

![heatmap](./heatmap_0.jpg)

위와 같은 경우는 heatmap 결과 값이 0.67이 나왔지만, 시각화 결과 한 영역에 집중되어 있는 것을 확인할 수 있다. 이럴 때에는 그 선형성을 의심해야한다.

</br>

### Feature 많을 때

https://scikit-learn.org/stable/modules/feature_selection.html

우선 위의 링크를 참조한다

추가적으로 팁은 1000개의 feature가 있을 경우, 1개씩 feature를 빼서 모델 성능을 확인하면 시간이 너무 오래 걸린다. 따라서 이럴 경우 block을 사용한다. block = 20으로 설정하면 총 1000/20 = 50번의 경우만 확인해보면 된다(위 링크에서 reculsive feature selection을 block화 시킨 방법)

**flow**

1. 20개 선정
2. 20개로 모델 학습 및 결과 산출
3. 20개 랜덤하게 선정해서 추가
4. 40개로 모델 학습 및 결과 산출
5. if 성능 향상되면
   1. feature importance 상위 10% 에 새로운 feature가 생겼는지 확인하고 있으면 그것을 남긴다
   2. 아니면 20개 다시 랜덤하게 선정해서 추가

6. 아니면 다시 랜덤하게 선정해서 추가

</br>

### early stopping

[Early stopping](https://en.wikipedia.org/wiki/Early_stopping) is an approach to training complex machine learning models to avoid overfitting

```python
xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=gini_xgb, maximize=True, verbose_eval=100)
```

nrounds 많큼 weak learner 만드는데, early_stopping_rounds 이후 성능 개선 되지 않으면 학습 멈춘다

</br>

### DMatrix

주로 넘파이 입력 파라미터를 받아서 만들어지는 XGboost만의 전용 데이터셋. 판다스의 dataFrame으로 데이터 인터페이스를 하기 위해서는 DataFrame.values를 이용해 넘파이로 일차 변환한 뒤에 이를 이용해 DMatrix 변환을 적용한다. 사용 이유는 메모리와 속도 향상. 파이썬 래퍼 사용시에는 DMatrix 사용한다. 사이킷런 래퍼는 사용 안해도 된다

```python
data_dmatrix = xgb.DMatrix(data=X,label=y)
```

</br>

### 두 개의 데이터 프레임을 합칠 때

https://datascienceschool.net/view-notebook/7002e92653434bc88c8c026c3449d27b/

#### merge

두 데이터 프레임의 공통 열 혹은 인덱스를 기준으로 두 테이블을 합친다.  이 때 기준이 되는 열, 행의 데이터를 키(key)라고 한다

key를 multi로 할 경우

```python
pd.merge(df_1, df_2, on=['key1', 'key2'])
```

### concat

key를 사용하지 않고 단순히 데이터를 세로 또는 가로 방식으로 연결한다. **따라서 인덱스 값이 중복될 수 있다**.

```python
df = pd.concat([df_1, df_2])
# 중복된 인덱스 없애기. 예를 들어 df_1이 0, 1, 2, 3 인덱스 있고, df_2가 0, 1, 2 있으면 df의 인덱스는 0, 1, 2, 3, 0, 1, 2 이런식으로 생성되어 버림
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
```

</br>

### append

concat과 동일한 기능. 마찬가지로 인덱스 값이 중복될 수 있기 때문에 reset_index 필요하다

```python
df = df.append(df1)
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
```

</br>

### PCA와 FA의 차이

PCA는 선형 독립, FA는 그렇지 않다

PCA는 numerical data에 사용가능하지만, categorical data에는 넌센스다. 이럴 경우 FA를 진행한다

https://dogmas.tistory.com/entry/%EC%9D%B8%EC%9E%90%EB%B6%84%EC%84%9DFactor-analysis%EA%B3%BC-%EC%A3%BC%EC%84%B1%EB%B6%84%EB%B6%84%EC%84%9DPrincipal-component-analysis%EC%9D%98-%EC%B0%A8%EC%9D%B4%EC%99%80-%EB%B9%84%EC%8A%B7%ED%95%9C-%EC%A0%90-%EB%B9%84%EA%B5%90-SPSS-%EC%82%AC%EC%9A%A9%EC%84%A4%EB%AA%85%EC%84%9C-25

</br>

### correlation

pearson corr : 두 continuous 변수 사이의 상관 관계

spearman 상관 관계 : 두 continuous 또는 ordinal 변수 사이의 상관 관계. 이상점이 있거나 표본 크기가 작을 때 유용하다. Spearman 상관 계수는 원시 데이터가 아니라 각 변수에 대해 순위를 매긴 값을 기반으로 한다

</br>

### Stacking

https://medium.com/@gurucharan_33981/stacking-a-super-learning-technique-dbed06b1156d

</br>

### Data Leakage

원호님이 보내주신 [링크](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/84614)에는 몇 가지 견해로 나오네요
첫째는 데이터에 대한 예상치 못한 info 유입
ex) Training data의 ordering이 예상치 못하게 target과 연관이 있음
사진 데이터, 사진을 찍으면 사진파일에 헤더로 사진을 찍은 정보가 나오기도 함
그런데, 이 첫째는 예측에 도움을 줄 수 있기 때문에 leak이긴 하나  good leak라고 하네요.

둘째는, Target Variable이 feature engineering 과정에서 적절하지 못하게 유입이 되는 경우
ex) Target encoding 과정에서 잘못하다가 새로운 feature로 target info가 누출될 수 있다고 합니다.

세번째는, A real machine learning problem, 그 feature가 실제로 사용이 가능할까?
ex) 고객이 물건을 살지 사지 않을지를 예측해야 하는데, 여러가지 feature(나이, 성별, 물건 가격, 등등)를 사용할 수 있습니다. 그런데 만약 '고객센터 전화시간'을 feature로 만들면 예측하는데는 아주 유용하게 쓸수는 있습니다만, 고객센터에 전화를 할지 말지는, 물건을 산 다음에 이루어지는 것이므로, 예측 후에 일어나는 사안에 관한 feature는 사용하지 않아야 한다.
-> 이건 약간 실제로 현업에서 데이터셋을 가지고 모델을 만들어야 하는 경우에 조심해야할것 같습니다.

종수님이 보내주신 링크는 아마 2번째와 연관이 있거나 비슷한 내용 같습니다.

https://ishuca.tistory.com/419

</br>

### 수행 시간 측정

```python
import time
start_time = time.time()
# ...
print("수행시간 : {%.1f}초".format(time.time() - start_time))
```

</br>

### get_clf_eval(y_test, predict)

accuracy, 정밀도, 재현율, F1, AUC 값 한 번에 나타낸다

</br>

### 부스팅 알고리즘 하이퍼파라미터 튜닝

Learning_rate를 작게 하면서 n_estimators를 크게 하는 것은 부스팅 계열 튜닝에서 가장 기본적인 튜닝 방안이다

</br>

### 일반적인 하이퍼파라미터 튜닝

2~3개 정도의 파라미터를 결합해 최적 파라미터를 찾아낸 뒤에 순차적으로 이 최적 파라미터를 기반으로 다시 1~2개 정도의 파라미터를 결합해 튜닝

</br>

### string 일치는 == 으로, dataframe 일치는 df_1.equals(df_2)

</br>

### groupby 좀 잘쓰자...

**상황**

dataframe에 id컬럼이 있고, id값은 중복되는 경우가 있다. 그리고 year 컬럼이 있다. 여기서 각 id마다 year가 무엇 무엇이 있는지 확인하고자 한다

```python
df_val = df.groupby('id')['year'].unique().to_frame()
df_val.reset_index(inplace=True)
```

만약 개수를 확인하고 싶다면

```python
df_val = df.groupby('id')['year'].nunique().to_frame()
df_val.reset_index(inplace=True)

df_val = df.groupby('id')['year'].count().to_frame()
df_val.reset_index(inplace=True)

# frequency_encoding도 있다
```

이렇게 볼 수도 있다

```python
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()})
```

</br>

### c for c in

```python
temp_list = [c for c in a_list if ('density' in c or 'caregory' in c)]
```

</br>

### dataframe to dictionary

id가 중복되는 경우 없을 때

```python
a = pd.DataFrame({'id' : [1, 1, 3, 5, 4],
									'YYYY' : [2001, 2002, 2003, 2004, 2004]})
a_dict = a.set_index('id')['YYYY'].to_dict()

# 결과물
# 중복되는 것 중 마지막 것만 입력됨
#{1: 2002, 3: 2003, 5: 2004, 4: 2004}
```

id가 중복되는 경우 있을 때

```python
a = pd.DaraFrame({'id' : [1, 1, 3, 5, 4],
									'YYYY' : [2001, 2002, 2003, 2004, 2004]})

a_dict = {id_: yyyy['YYYY'].tolist() for id_, yyyy i ttt.groupby('id')}
# 결과물
# 중복되는 것 중 마지막 것만 입력됨
# {1: [2001, 2002], 3: [2003], 5: [2004], 4: [2004]}
```

</br>

### hyperparemeters tuning, optimization

* grid search

* random search, coarse to fine search

* bayesian optimization

  * 두 가지 라이브러리 있다

    ```python
    from bayes_opt import BayesianOptimization
    from skopt import BayesSearchCV
    ```

</br>

### 소수점 자릿수

```python
# 39.54484700000000 -> 39.54
print("{:.2f}".format(39.54484700000000))
```

</br>

### col로 sort

```python
ttt.sort_values(by=['id', 'YYYY'])
```

</br>

### column type 확인

```python
if mammo[col].dtypes == np.int64
if mammo[col].dtypes == np.float64
# 둘 다 아니면 object
```

</br>

### 다중공선성

피쳐간의 상관관계가 매우 높은 경우 분산이 매우 커져서 오류에 매우 민감해진다

</br>

### regplot

산점도와 함께 선형 회귀 직선을 그림

```python
sns.regplot(x=feature, y='PRICE', data=bostonDF, ax=axs[row][col])
```

</br>

### 다항 회귀

회귀가 독립 변수의 단항식이 아닌 2차, 3차 방정식과 같은 다항식으로 표현되는 것을 다항 회귀라고 한다. 다항 회귀는 선형 회귀다. 회귀에서 선형/비선형 회귀를 나누는 기준은 회귀 계수가 선형/비선형인지에 따른 것이지 독립 변수의 선형/비선형 여부와는 무관하다. 사이킷런에서는 다항 회귀를 위한 클래스를 따로 마련해두지 않았기 때문에 PolynomialFeatures()로 feature를 변형하고, 이를 LinearRegression 클래스로 다항 회귀 구현한다. degree가 높을 수록 overfitting 위험성 높다. 일반적으로 차수는 2를 넘기지 않는다. 다항식 변환은 히처의 개수가 많을 경우 적용하기 힘들며, 또한 데이터 건수가 많아지면 계산에 많은 시간이 소모되어 적용에 한계가 있다

</br>

### 문자열 있는지 확인

```python
mysite = 'Site name is webisfree.'
if 'webisfree' in myname:
  print('Okay')
```

</br>

### 로그 변환

일반적으로 log() 함수를 적용하면 언더 플로우가 발생하기 쉬워서 1 + log() 함수를 적용하는데 이를 구현한 것이 np.log1p()

* under flow : 메모리가 표현할 수 있는 수 보다 작은 수를 저장하는 경우

```python
np.log1p()
```

log1p()로 변환된 값을 다시 원래 상태로 복구

```python
np.expm1()
```

</br>

### 데이터 프레임 동일한지 비교

```python
df1.equals(df2)
```

</br>

### 회귀 트리 regressor는 feature importance를, 선형 회귀는 coef_ 회귀 계수를 확인

</br>

### 데이터 복사

```python
house_df_org = pd.read_csv('./house_price.csv')
house_df = house_df_org.copy()
```

</br>

### 모든 컬럼에서 null 확인

```python
df.isnull().sum()
print(isnull_series[isnull_series > 0].sort_values(ascending=False))

# 특정 컬럼만 확인
isnull_series = df[col].isnull().sum()
```

</br>

### 모델 이름 파악

```python
# model._class_.name를 사용하는 예시

models = [lr_reg, ridge_reg, lasso_reg]

def get_rmse(model):
	pred = model.predict(X_test)
	mse = mean_squared_error(y_test, pred)
	rmse = np.sqrt(mse)
	print(model._class_.name, ' 로그 변환된 RMSE : ', np.round(rmse, 3))
	return rmse
	
get_rmses(models):
	rmses=[]
	for model in models:
		rmse = get_rmse(model)
		rmses.append(rmse)
	return rmses

```

</br>

### skew()

일반적으로 skew() 함수의 반환 값이 1 이상인 경우를 왜곡 정도가 높다고 판단한다. 하지만, 상황에 따라 편차는 있다. 1이상의 값을 반환하는 feature만 추출해 왜곡 정도를 완화하기 위해 변환(ex- log)을 적용한다. 이 때 주의할 점은 모든 숫자형 feature에 적용하는 것이 아니라는 점이다. one-hot-encoding된 카테고리 숫자형 feature는 제외한다

* 파이썬 머신러닝 완벽 가이드 p.361 참조

```python
from scipy.stats import skew

# object가 아닌 숫자형 피처의 컬럼 index 객체 추출
features_index = house_df.dtypes[house_df.dtypes != object].index
# int, float, object로 분류 할 수 있다

# house_df에 컬럼 index를 []로 입력하면 해당하는 컬럼 데이터 세트 반환.
skew_features = house_df[features_index].apply(lambda x: skew(x))
# skew 정도가 1이상인 컬럼만 추출
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))
                                 
# log변환(항상 log 변환은 아닐 수 있다)                                 
house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])                                 
```

</br>

### list 원소 unique하게 변경

```python
my_set = set(my_list)
my_list = list(my_set)
```

</br>

### dataframe 행 변경 및 추가

```python
# 행
df.loc[index번호] = [1, 2, 3]

# 열
df['A'] = [10, 20, 30]
```

</br>

### (list a) - (list b) 없다. (set a) - (set b) 있다

</br>

### nan은 isnull()로 안잡힌다. NaN, None은 isnull()로 안잡힌다

### nan, NaN, None의 타입은 각각 str, float, NoneType

</br>

### row별 각 컬럼값의 합, 최소값...을 sum이라는 컬럼에 넣음

```python
df['sum'] = df[numerical_fs].sum(axis=1)  
```

</br>