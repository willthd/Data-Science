# Colab

> How to use colab

</br>

### 단축키

기존 jupyter notebook에서 shift 대신 command + m + ?

</br>

### local 에서 파일 업로드

```python
from google.colab import files
uploaded = files.upload()
```

</br>

### terminal 명령어

기존 terminal 명령어를 셀에서 동작 시킬 수 있으며, 이 때 앞에다가 !+기존 명령어로 진행

</br>

### sample_data 경로

content/sample_data

</br>

### kaggle 연동

```python
import os, shutil
```

```python
!pip install kaggle
from google.colab import files
# kaggle.json 연동. workingdir에 존재
files.upload()
```

```python
ls -1ha kaggle.json
```

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
# Permission Warning 이 일어나지 않도록 
!chmod 600 ~/.kaggle/kaggle.json
# 본인이 참가한 모든 대회 보기 
!kaggle competitions list
```

```python
! kaggle competitions download -c dogs-vs-cats
```

```python
!ls
```

```python
!unzip train.zip
```

</br>

### GitHub 노트북 파일을 Colab으로 가져와 실행하는 방법

**github주소** : Https://github.com/~

Https://colab.research.google.com/github/~

#### Github에서 dataset 가져오기

```python
!wget https://github.com/~
```

</br>

### Google Drive에서 파일 사용

```python
from google.colab import drive
drive.mount('/content/drive')

# Google colab: 데이터 경로 설정
path = 'drive/MyDrive/data/'
os.listdir(path)
```

