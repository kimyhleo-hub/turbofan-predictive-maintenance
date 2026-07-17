# 데이터 안내 — NASA C-MAPSS

원본 데이터는 용량 때문에 레포에 커밋하지 않는다(`.gitignore`로 제외). **각자 아래에서 받아 `data/raw/`에 넣는다.**

## 다운로드

두 곳 중 하나에서 받으면 된다 (내용 동일).

- **Kaggle**: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **GitHub 미러**: https://github.com/edwardzjl/CMAPSSData

## 배치 위치

받은 `.txt` 파일들을 `data/raw/`에 넣는다.

```
data/raw/
├── train_FD001.txt   test_FD001.txt   RUL_FD001.txt
├── train_FD002.txt   test_FD002.txt   RUL_FD002.txt
├── train_FD003.txt   test_FD003.txt   RUL_FD003.txt
└── train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

> **FD001부터 시작** (가장 쉬운 세트). FD002~004는 나중에 일반화 확인용.

## 파일 형식

컬럼명 없는 공백 구분 텍스트, 26개 열:

```
unit  cycle  setting1  setting2  setting3  sensor1 ... sensor21
```

- `unit`: 엔진 번호, `cycle`: 사이클(1부터 고장까지)
- `train`: 고장까지 전체 기록 / `test`: 고장 전 중단 → `RUL_FDxxx.txt`가 각 test 엔진의 실제 잔여수명(정답) 제공
- FD001 기준 유효 센서 15개 (s1·s5·s10·s16·s18·s19는 상수 → 제거)

## 빠른 확인 (Python)

```python
import pandas as pd
cols = ["unit","cycle","set1","set2","set3"] + [f"s{i}" for i in range(1,22)]
df = pd.read_csv("data/raw/train_FD001.txt", sep=r"\s+", header=None, names=cols)
print(df.shape, "| 엔진 수:", df.unit.nunique())
```
