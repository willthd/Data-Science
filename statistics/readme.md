# Statistics

> 관심 또는 연구의 대상인 모집단의 특성을 파악하기 위해 모집단의 일부의 **자료(표본)를 수집하고**, 수집된 표본을 정리, 요약, 분석 하여 **표본의 특성을 파악한 후**, 표본의 특성을 이용해 **모집단의 특성**에 대해 추론하는 것

</br>

</br>

### Histogram vs Bar Plot

히스토그램은 수치자료 특히 연속자료의 분포형태를 표

히스토그램 vs 막대그래프 

* 구간이 같다면 차이가 없고, 구간이 다르다면 차이가 존재

계급의 상대도수를 사각형의 면적으로 표시  ⇒ 전체 면적 = 1

* 높이 = 상대도수/계급구간길이 = 밀도(density) 

</br>

### 수치자료의 대체중심위치

* 평균값
* 중앙값 : 순서통계량
* 절사평균값 : 오름 차순으로 data 정렬 후, 아래에서부터 k개 위에서부터 k개 제거 한 다음 남은 data로 평균
* 최빈값

</br>

### Box Plot(Viloin Plot)

**자료의 주요 위치 파악과 outlier 검출**

![box_plot](./box_plot.jpg)

자료를 ascending하게 나열 한 뒤,

Q1 : 25%에 위치하는 data

Q2 : 50%에 위치하는 data

Q3 : 75%에 위치하는 data

IQR(Inter Quartile Range) = Q3 - Q1

* Inner Fence

* * I1 = Q1 - 1.5 * IQR
  * I2 = Q3 + 1.5 * IQR

Outlier : I1 ~ I2 바깥에 위치하는 data

* Outer Fence

  * O1 = Q1 - 3 * IQR

  * O2 = Q3 + 3 * IQR

Outlier : I1 ~ I2 바깥에 위치하는 data. 경우에 따라 Outer Fence를 기준으로 Outlier를 설정하기도 한다

**추가**

Violin plot

![violin](./violin.png)

</br>

### 표준화(Standardization)

![stand](./stand.jpg)

표준화를 하면 절대 비교 가능

ex) 국어 원점수, 수학 원점수는 과목별 난이도가 다르기 때문에 절대비교 불가능하지만, 국어 표준점수, 수학 표준점수는 가능

표준화된 값(zi)의 평균은 0, 분산은 1, 표준편차는 1

</br>

### 변동계수 (Coefficient of Vaiation, CV)

변동계수 = 표준편차/평균

서로 다른 평균값을 가지는 경우 산포를 표준편차를 기준으로 하지 않고, 변동계수를 기준으로 하기도 한다

예를 들어 (3, 5, 12, 5, 8)라는 데이터 계열과 (15, 25, 60, 25, 40)라는 데이터 계열은 각각 3.14, 15.68이라는 표준편차를 가진다. 따라서 후자 쪽의 데이터가 편차가 큰 것 같이 보이지만 변동계수를 계산하면 양자는 0.475라는 같은 값을 갖는다

</br>

### 분포의 형태

https://rucrazia.tistory.com/65

https://support.minitab.com/ko-kr/minitab/18/help-and-how-to/statistics/basic-statistics/supporting-topics/data-concepts/how-skewness-and-kurtosis-affect-your-distribution/

Skewed 데이터를 변환하는 이유는 꼬리에 있는 값을 모델에 제대로 학습시키기 위함이다. Skewed 되어있는 값을 그대로 학습 시키면 꼬리 부분이 상대적으로 적고 멀어서 모델에 영향이 거의 없이 학습된다. 만약 꼬리부분도 노이즈가 아닌 정말 유의미한 데이터이면 꼬리부분에 해당하는 test 데이터는 예측력이 낮아진다.
변환을 해서 들어가게 되면 그만큼 데이터의 중간값(or 평균값)가 tail하고 가까워져서 모델에 보다 크게 들어간다. 그렇게 되면 꼬리쪽에 해당하는 test 데이터가 들어와도 예측력이 높아진다.

</br>

#### 왜도(Skewness)

자료가 대칭적으로 분포되어 있는지, 한쪽으로 기울어져 있는지

기울어져 있을 수록 outlier 존재 가능성 높다

피어슨(Karl Pearson) 제안

![skew](./skew.jpg)

루트 b1이 0에 가까울 수록 대칭

</br>

#### 첨도(Kurtosis)

양쪽 꼬리가 얼마나 두터운지

양쪽 꼬리가 두터울 수록 outlier 존재 가능성 높다

피어슨(Karl Pearson) 제안

![kurto](./kurto1.jpg)

b2가 3에 가까울 수록 꼬리 적당

</br>

#### 정규성 검정

왜도 = 0, 첨도 = 3에 가까울 수록 정규분포에 가까움

Jqcque-Bera 검정

![jb](./jb.jpg)

</br>

### Gasussian



### Laplasian



### 정규분포, 표준정규분포



### p-value

p-value(유의 확률)는 간단히 귀무가설을 지지하는 확률. 귀무가설은 내가 증명하고자 하는 대립가설에 반대되는 가설이다. 즉, 유의확률이 낮을 수록 귀무가설이 거짓이 될 확률이 높아지며, 이는 내가 증명하고자 하는 대립가설이 참이 될 확률이 높아지는 것이다. 이 때, 기준은 유의 수준에 의해 결정된다. 통계에서 98%의 신뢰구간을 갖는다면 신뢰 수준은 0.98이며 유의 수준(1-신뢰 수준)은 0.02 이다. 내가 세운 가설(대립 가설)이 참이 되게 하기 위해서는 유의 확률이 유의 수준보다 낮으면 관례적으로 대립 가설을 지지한다고 볼 수 있다.

</br>

#### p-value는 귀무가설이 참일 확률이 아니다.

> p-value는 귀무가설을 지지하는 확률이다. (귀무 가설 하에 표본의 분포를 그리고 그 분포와 모분포(실제 이상적인 분포)를 비교할 수 있는 확률이다.)
>
> 귀무가설이 참일 확률은 구할 수 없다.
>
> p-value가 낮아도 귀무가설이 참일 수 있고, p-value가 (아주) 높아도 귀무가설은 틀릴 수 있다.

#### p-value와 유의수준(alpha)을 통해 귀무가설을 기각했는데, 이게 올바른 판단인 것인가?

> 안타깝게도 **귀무가설을 잘못 기각했는지, 아니면 맞게 기각했는지는 확인할 수 없다.**
>
> 즉, 귀무 가설을 잘못 기각했다는 것은 확률값으로 판단할 수가 없다.
>
> 귀무가설이 맞다는 전제 하에 나온 분포에서 확률을 보는 것이다.
>
> - 만약 이 분포에서 p-value = 0.001의 값을 얻었고, 낮은 유의확률을 갖기 때문에 귀무가설을 기각했다.
> - 하지만, 우리는 귀무가설이 맞았음에도 불구하고 p-value가 낮았기 때문에 기각했다고 말할 수는 없다.
> - 왜냐하면, 0.1%의 경우에 대해서 귀무 가설이 참인지 거짓인지 확인할 방법이 없기 때문이다.
>
> 그리고 정의상 p-value는 그런 개념이 아니다. 만약 애초에 귀무가설이 틀렸다고 해보자.
>
> - 그런 상황에서도 여전히 p-value는 구해지는데, 그런 p-value가 과연 귀무가설을 잘못 기각한 확률이 될 수 있을까?
> - 애초부터 귀무가설이 틀릴 수도 있고, 귀무가설이 거짓이라도 p-value는 여전히 구해지기 때문에 귀무 가설이 참인지 거짓인지는 확인할 수가 없다.
> - 그래서 우리는 귀무가설이 맞다고 가정하고 문제를 풀기 시작하는 것이다.

#### 일반적으로 p-value는 어떤 가설을 전제로, 그 가설이 맞는다는 가정 하에, 내가 현재 구한 통계값이 얼마나 자주 나올 것인가를 의미한다고 할 수 있다.

> 가설검정은 전체 데이터를 갖고 하는 것이 아닌 sampling 된 데이터를 갖고 하는 것이다.
>
> 그래서 p-value가 필요하다.

#### 1-(p-value)는 대립가설이 맞을 확률이 아니다

> p-value와 대립가설은 관련이 없다.
>
> 순전히 '귀무가설이 맞다는 전제 하에' 나온 값이 p-value이고, p-value를 구함에 있어 대립가설은 그 어디에서도 작용하지 않는다.

#### 유의수준 significance level(alpha)은 p-value에 의해 결정되는 것이 아님

> alpha는 연구자의 주관이며, 관례적으로 0.05, 0.01 을 사용

</br>

https://namyoungkim.github.io/statistics/2017/09/13/probability_1/

http://www.databaser.net/moniwiki/wiki.php/%EC%9C%A0%EC%9D%98%ED%99%95%EB%A5%A0

https://adnoctum.tistory.com/332

http://statistics4everyone.blogspot.com/2016/06/p-story-iii-p.html

</br>

### Bayesian Theorem, MLE, MAP

https://darkpgmr.tistory.com/62?category=761008

https://darkpgmr.tistory.com/119?category=761008

</br>

### 강의

http://www.kmooc.kr/courses/course-v1:SookmyungK+SM_sta_004k+2018_02SM_02/about

</br>

https://www.edwith.org/harvardprobability/joinLectures/17924









