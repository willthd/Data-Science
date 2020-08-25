# Seaborn



### 참고

https://seaborn.pydata.org/index.html

https://datascienceschool.net/view-notebook/4c2d5ff1caab4b21a708cc662137bc65/



**seaborn에서 param 지정할 때 set 사용**

```python
sns.kdeplot(kpx_1836['nE_1836'], ax=axes[0][0]).set(xlim=(-0.75, 0.75), ylim=(0, 7), title=f'1836 nE range, std={std_1836:.2f}')
```

