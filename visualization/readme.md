# Seaborn



### 참고

https://seaborn.pydata.org/index.html

https://datascienceschool.net/view-notebook/4c2d5ff1caab4b21a708cc662137bc65/



**seaborn에서 param 지정할 때 set 사용**

```python
sns.kdeplot(kpx_1836['nE_1836'], ax=axes[0][0]).set(xlim=(-0.75, 0.75), ylim=(0, 7), title=f'1836 nE range, std={std_1836:.2f}')
```



**pyplot**

```python
plt.figure(figsize=(20, 10))
plt.plot(df_13['prediction'], marker="o", label='13 prediction')
plt.plot(df_13['real'], marker="*", label='13 real')
plt.ylabel('Power Generation', fontsize=16)
plt.xlabel('Hour', fontsize=16)
plt.tick_params(labelsize=13)
plt.legend(fotsize=15)
plt.grid()
plt.title("2021-02-13", fontsize=20)
```

![스크린샷 2021-03-02 오후 5.37.07](/Users/jongsoo/Desktop/screenshot/스크린샷 2021-03-02 오후 5.37.07.png)