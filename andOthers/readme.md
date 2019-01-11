#  and others

> 카테고리 무시한 tips



### Boosting Algorithms

GBM, adaboost, XGboost, light BGM, catboost (순서)

XGboost는 categorical features를 따로 one-hot-encoding 해야 하지만, light BGM, catboost는 그럴 필요 없다

</br>

</br>

### python, 기타 라이브러리들 중 자주 사용하는 code

#### 컬럼 지우기

```python
del train["col_name"]
```

</br>

#### dictionary로 구성된 list -> dataFrame

```python
count_list = pd.DataFrame.from_dict(count_list)
```

</br>

#### dataFrame, 컬럼 기준으로 정렬하기

```python
count_list = count_list.sort_values(by="a", ascending=False)
```

</br>

#### dataFrame에 index 입력하기

```python
count_list.index.name = "store_id"
```

</br>

#### 시계열 data 읽어오면서 date 인식. 연도, 월, 날짜, 시간, 분, 초 컬럼 생성

```python
test = pd.read_csv("data/bike/test.csv", parse_dates=["datetime"])

train["datetime-year"] = train["datetime"].dt.year
train["datetime-month"] = train["datetime"].dt.month
train["datetime-day"] = train["datetime"].dt.day
train["datetime-hour"] = train["datetime"].dt.hour
train["datetime-minute"] = train["datetime"].dt.minute
train["datetime-second"] = train["datetime"].dt.second
```

</br>

#### 시계열 data를 split해서 연도, 월, 날짜 관련 컬럼을 생성

```python
def split_date(date):
    return date.split("-")
```

```python
train["year"], train["month"], train["day"] = zip(*train['date'].apply(lambda x: split_date(x)))

# zip([1, 2, 3], [2, 3, 4], [3, 4, 5]) -> (1, 2, 3), (2, 3, 4), (3, 4, 5)
# zip(*[[1, 2, 3], [2, 3, 4], [3, 4, 5]]) -> (1, 2, 3), (2, 3, 4), (3, 4, 5)
# *은 unpack을 의미한다
# .apply(lambda x : 함수)
```

</br>

#### string 형태의 숫자를 int로 바꿀 때

```python
# train["year"] = int(train["year"])로 하면 안된다
# error message : cannot convert the series to <class 'int'>

train["year"] = train["year"].astype(int)
```

</br>

#### 그래프 모아서 보기

```python
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18, 4)
sns.barplot(data=train, x="year", y="total", ax=ax1)
sns.barplot(data=train, x="month", y="total", ax=ax2)
```

</br>

#### 그래프 사이즈 정하기

```python
plt.figure(figsize=(10, 7))
```

</br>

#### train["column"] 값 array로 만들기

```python
my_array = train["column"].values
```

</br>

#### array 내에 중복되는 값 없애기

```python
uniqueVals = np.unique(my_array)
```

</br>

#### array -> list

```python
my_list = uniqueVals.tolist()
```

</br>

#### case : "nextday_date"컬럼 값이 holiday_date_list에 포함 되어 있는 경우 "nextday_holiday"를 1로 채워넣기

```python
idx = [idx for idx, value in enumerate(train["nextday_date"]) if value in holiday_date_list]

# enumerate는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다
# idx 리스트는 결국 해당 조건이 맞는 경우의 idx들의 모음 의미한다

train.loc[idx, "nextday_holiday"].apply(lambda x : 1)
```

