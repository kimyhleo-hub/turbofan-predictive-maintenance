# Predictive Maintenance Scheduling for Turbofan Engines
**RUL Prediction + Resource-Constrained Metaheuristic Scheduling on NASA C-MAPSS**

Purdue 팀 프로젝트 · 4인 · 2026년 7월

NASA C-MAPSS 터보팬 엔진 데이터로 엔진별 잔여수명(RUL)을 예측하고, 그 예측을 정비 마감으로 삼아 제한된 정비 자원 하에서 정비 스케줄을 최적화한다. 만들어진 스케줄은 실제 고장 시점에 재현하여 성과를 검증한다.

---

## 개요

예지정비에서 잔여수명 예측은 그 자체가 목적이 아니라 정비 의사결정의 입력이다. 그러나 다수의 연구는 예측 정확도에 집중하고, 예측을 정비 계획으로 옮기는 단계는 "RUL이 임계값 이하이면 정비"라는 단순 규칙에 머문다. 정비 인력과 예비부품이 한정된 현실에서는 어느 엔진을 언제 정비할지가 곧 비용과 안전을 좌우하는 조합 최적화 문제다.

본 프로젝트는 세 단계로 구성된다.

1. **예측** — 센서 시계열로 엔진별 RUL을 회귀 예측한다.
2. **최적화** — 예측 RUL을 각 엔진의 정비 마감으로 두고, 정비팀 용량·부품 제약 하에서 스케줄을 최적화한다.
3. **검증** — 스케줄을 실제 고장 시점에 재현하여 미정비 고장 수·다운타임으로 성과를 측정한다.

**기여**

- RUL 예측을 자원 제약 정비 스케줄링에 연결하는 예측–최적화 파이프라인.
- 임계값 정책 대비, 자원 제약과 메타휴리스틱을 도입해 스케줄 품질을 실측 성과로 비교.
- 실제 고장 시점이 알려진 데이터로 스케줄을 검증하므로, 가정한 비용에 의존하지 않는 물리 지표(고장 수·다운타임)로 평가할 수 있다.

## 문제 정의

- **ML** — 엔진 센서 시계열로부터 잔여수명 RUL을 예측한다 (회귀).
- **Optimization** — 예측 RUL과 정비팀 용량·부품 제약 하에서, 미정비 고장과 다운타임을 최소화하는 정비 스케줄을 결정한다.

## 배경

C-MAPSS는 터보팬 엔진을 고장까지 반복 운전시키며 사이클별 센서값을 기록한 벤치마크로, 각 엔진의 실제 수명이 알려져 있다. 정비 정책은 통상 사후정비(고장 후), 정기정비(주기 기반), 예지정비(상태·예측 기반)로 나뉜다. 본 프로젝트는 예지정비 중에서도 RUL 예측과 자원 제약 스케줄링을 결합한 형태를 다룬다.

RUL 예측과 정비 스케줄링을 결합하는 연구는 확립된 흐름이나(예: IISE Transactions 2026, Annals of OR 2026), 공개 코드가 있는 대부분의 작업은 임계값 기반 정책에 머물고 정비팀 용량·부품 같은 자원 제약이나 메타휴리스틱 스케줄링을 다루지 않는다. 본 프로젝트의 초점은 그 공백, 즉 자원 제약 하 스케줄 최적화에 있다.

## 데이터

**NASA C-MAPSS Turbofan Engine Degradation** (Saxena et al., 2008)

컬럼은 `unit`, `cycle`, 운전조건 3개(`setting1~3`), 센서 21개(`sensor1~21`)로 구성된 공백 구분 텍스트다. 타깃 RUL은 원본에 없으며 `RUL = 총수명 − 현재 cycle`으로 계산한다(학습 시 상한 clip).

| 세트 | 엔진 수 | 수명(사이클) | 운전조건 | 고장모드 |
|------|--------:|-------------|---------:|---------:|
| FD001 | 100 | 128–362 | 1 | 1 |
| FD002 | 260 | 128–378 | 6 | 1 |
| FD003 | 100 | 145–525 | 1 | 2 |
| FD004 | 249 | 128–543 | 6 | 2 |

FD001을 기본으로 하고 FD002~004로 일반화를 확인한다. `train`은 고장까지 전 구간, `test`는 고장 전 중단된 구간이며 각 test 엔진의 실제 잔여수명이 정답으로 제공된다. FD001 기준 6개 센서(s1·s5·s10·s16·s18·s19)는 상수여서 제거하고 15개를 사용한다. 다운로드·배치는 [`data/README.md`](data/README.md) 참고.

## 방법

### RUL 예측

센서 시계열에서 최근 구간의 통계 특징(평균·기울기 등)을 추출하거나 시계열을 직접 입력해 RUL을 회귀한다.

| 구분 | 내용 |
|------|------|
| 입력 | 유효 센서 시계열(윈도우 특징 또는 원시 시퀀스) |
| 출력 | 엔진별 RUL(사이클) |
| 모델 | 선형회귀·RandomForest(baseline) → 트리 앙상블(LightGBM/XGBoost) → LSTM(선택) |
| 지표 | RMSE, MAE, PHM08 score(늦은 예측에 더 큰 벌점) |

평가는 엔진 단위로 train/test를 분리한다. 같은 엔진의 서로 다른 사이클을 무작위로 섞으면 미래 정보가 학습에 유입되어 성능이 과대평가된다. 표준 프로토콜에서 FD001의 현실적 RMSE는 약 12–14 수준이다.

### 정비 스케줄링 최적화

계획 기간을 `T`개 정비 창으로 나누고, 창당 정비팀이 `K`대까지 처리한다. 예측 RUL은 각 엔진의 정비 마감 창 `d_i`가 되며, 이 창까지 정비하지 못하면 고장으로 간주한다.

**결정변수**

```
x[i,t] = 1  엔진 i를 창 t에 정비, 아니면 0
prevented_i = 1  if  정비 시점 m_i ≤ 마감 d_i,  else 0
```

**목적함수** — 두 층으로 둔다.

```
(주)  minimize  Σ (1 − prevented_i)          미정비 고장 수
      또는      Σ wasted_life_i               낭비된 잔여수명(다운타임)

(부)  minimize  C_pm·(정비수) + C_fail·(고장수) + w·(조기정비 낭비),   C_fail ≫ C_pm
```

주 지표는 실제 고장 데이터로 직접 셀 수 있어 비용 가정에 의존하지 않는다. 비용비 `C_fail : C_pm`은 문헌 범위(통상 3–10배)를 참고해 설정하고 민감도로 확인한다.

**제약**

```
Σ_t x[i,t] ≤ 1              엔진당 정비 1회
Σ_i x[i,t] ≤ K       ∀t    창당 정비팀 용량
Σ x[i,t]·part_i ≤ Stock    예비부품 재고
m_i ≤ d_i − s_i            안전마진(RUL 불확실성 반영)
```

마감과 창 용량이 있는 배정·스케줄링 문제로, 용량·부품만 있는 경우는 EDF(마감 우선) 또는 정확법으로 풀 수 있다. 그러나 같은 기종을 같은 창에 모아 셋업 비용을 절감하는 묶음정비를 도입하면 목적함수가 비분리·비선형이 되어 NP-hard 스케줄링이 된다. 이 영역을 메타휴리스틱으로 다룬다.

**알고리즘**

- baseline — Random, EDF(마감 우선), 비용/위험 비율 greedy
- 메인 — Simulated Annealing. 해는 엔진별 정비 창 배정 벡터로 표현하고, 한 엔진의 창 변경 또는 두 엔진의 창 교환을 이웃 연산으로 둔다. 악화 해는 `exp(−Δ/T)` 확률로 수용하며 온도를 지수적으로 냉각한다. 초기 온도·냉각률·반복 수를 튜닝하고 수렴 곡선을 기록한다.

## 평가

예측 RUL로 만든 스케줄을 실제 고장 시점에 재현하여 미정비 고장 수, 다운타임, 총비용을 측정한다. Random·EDF·greedy·SA를 비교하고, 실제 RUL을 안다고 가정한 Oracle을 상한으로 두어 예측 오차가 성과에 미치는 영향을 분리한다.

민감도 분석으로 정비팀 용량 `K`, 비용비 `C_fail:C_pm`, 부품 재고, RUL 예측 불확실성을 변화시키며 결론의 강건성을 확인한다.

## 핵심 결과

전 707대 엔진 기준, 정적(1회 계획) 스케줄을 실제 고장 시점에 재현했을 때의 미정비 고장 수:

| 조건 | 미정비 고장 수 | 비고 |
|---|---:|---|
| 기본 예측(RMSE 15.0) | 85대 | 초기 특징셋 |
| 개선된 예측(RMSE 10.90, v3) | **48대** | 예측 개선만으로 -44% |
| **P10 분위수 마감**(같은 특징셋) | **39대** (-19%) | 점추정 대신 예측 불확실성을 마감에 직접 반영 |
| Oracle(실제 RUL을 안다고 가정, 상한) | 4대 | 스케줄러 성능의 이론적 한계 |

Oracle과의 격차(48→4)를 분해하면 9대는 불확실성 정량화 미활용분(P10으로 회수), 35대는 예측 오차 자체의 환원 불가능분이다 — 즉 **이 시점부터는 스케줄링 알고리즘이 아니라 예측 정확도가 병목**이라는 뜻이다.

**공정성 검증에서 나온 자기 정정**: 초기에는 "담금질 기법(SA)이 EDF(마감 우선)보다 낫다"고 봤으나, 서로 다른 마진 조건을 비교한 결과였다. 같은 조건에서 재비교하니 EDF@마진0 = SA(707대 중 배정 변경 0건) — 성능 차이는 SA가 아니라 마진 정책에서 왔다. SA의 실질 기여는 24,000회 탐색으로 그 해가 실제로 최적임을 검증한 것이었다. (이 발견 과정은 `reports/final_report.pdf`에 그대로 기록되어 있다.)

**확장: 정적 계획 대비 롤링(매 창 재예측·재계획)**

| 정책 | 미정비 고장 수 |
|---|---:|
| 정적(1회 계획), P10 마감 | 24.0 ± 5.1대 |
| **롤링(매 창 재예측), P10 마감** | **1.7 ± 0.9대 (-93%)** |
| Oracle(참조) | 0대 |

열화 말기로 갈수록 센서 신호가 강해져 예측이 정확해지는데, 롤링 재계획은 정확도가 필요한 시점과 정확도가 확보되는 시점을 일치시켜 이 효과를 극대화한다.

## 코드 구조

```
notebooks/               통합 노트북 (전부 실행 완료, 출력 포함)
  01_eda.ipynb           EDA — 2x2 구조, 조건 가림, 고장모드 분기, 센서 4분류
  02_preprocessing.ipynb 조건별 z정규화·RUL 라벨·특징·엔진 단위 분리 + 검증(V1-V15)
  03_rul_model.ipynb     RUL 예측 — ablation·튜닝·LSTM 사다리·루프 v2/v3 -> test 10.90
  04_optimization.ipynb  스케줄링 — SA·공정비교 감사·P10 분위수 마감·롤링 재계획
  05_evaluation.ipynb    최종 평가 — 독립 재계산·문헌 비교·검증 감사·결론
reports/                 보고서 5부작 (노트북과 1:1 대응, 각 .pdf + LaTeX 소스 .tex)
  eda_report            01 노트북 대응 — 데이터 탐색
  preprocessing_report  02 노트북 대응 — 전처리·라벨링
  rul_model_report      03 노트북 대응 — RUL 예측 모델
  scheduling_report     04 노트북 대응 — 정비 스케줄링 최적화
  final_report           전체 파이프라인 통합 결과 보고서
scripts/                 체크포인트형 실험 스크립트 (LSTM·튜닝·SA·롤링·감사·검증 등 17개)
  run_03b_experiments.py / run_v2_features.py / run_loop2.py   RUL 모델 실험(튜닝·LSTM·루프)
  run_04_scheduling.py / run_04b_sa_tuning.py / run_04d_quantile.py / run_04f_rolling.py
                          스케줄링 실험(전략 비교·SA 튜닝·P10 분위수·롤링)
  run_04c_stats.py / run_04e_audit.py / verify_v2.py            통계 검정·공정성 감사·재검증
  generate_*_artifacts.py / generate_eda_supplement.py          보고서용 그림·표 생성
  algo_template.py / validate_submission.py                     팀원 알고리즘 제출 어댑터·규격 검증
  run_notebooks.py / run_experiment.py                          노트북 일괄 실행·전략 비교 재현
models/                  학습된 LightGBM 모델 덤프 4종 (레포 포함, 04가 사용)
artifacts/               노트북이 로드하는 실험 결과 체크포인트 (레포 포함)
  03_model/              03 노트북용 (튜닝·LSTM·루프)
  04_scheduling/         04 노트북용 (SA·P10·롤링)
src/pipeline.py          공용 함수 (RUL 예측·스케줄링) — 03·04·05가 import
data/                    C-MAPSS 원본은 미포함(data/README.md 참고); processed/의 최종 예측 CSV는 레포 포함
reports/figures/         결과 그림
```

## 재현 순서 (중요)

노트북은 앞 단계의 산출물에 의존한다. **아래 순서대로** 실행한다.

| 순서 | 노트북 | 필요한 것 | 생성하는 것 | 소요 |
|---|---|---|---|---|
| 1 | `01_eda` | `data/raw/` | 그림 | ~5분 |
| 2 | `02_preprocessing` | `data/raw/` | **`data/processed/*.csv`** (03의 전제) | ~10분 |
| 3 | `03_rul_model` | 2단계 산출물 + `artifacts/03_model/` | 모델·예측 | ~20분 |
| 4 | `04_optimization` | `artifacts/04_scheduling/` + fleet CSV (**둘 다 레포 포함**) | 스케줄·그림 | ~5분 |
| 5 | `05_evaluation` | 위 + `data/raw/` (원본과 독립 대조용) | 최종 지표 | ~1분 |

> **04는 clone 직후 바로 실행된다** (원본 데이터도 필요 없음 — 검증 완료).
> 03은 02를, 05는 `data/raw/`를 먼저 준비해야 한다.

**`data/raw/` 원본은 커밋되지 않는다** — [data/README.md](data/README.md)를 보고 C-MAPSS를 받는다.

### 체크포인트에 대해

`03`은 무거운 실험(LSTM 3변형 학습, 24-trial 튜닝)의 **결과 체크포인트를 로드**한다.
이 파일들은 크기가 작아 **레포에 포함**되어 있으므로 `git pull`만 하면 된다:

```
artifacts/03_model/tuning_lstm_summary.json  # 튜닝·LSTM 사다리 결과
artifacts/03_model/loop2_summary.json  # 루프 v3 결과
artifacts/03_model/s2_lstmD.npz        # LSTM-D 예측
artifacts/04_scheduling/*.npy, *.json       # P10·롤링 결과
data/processed/fleet_rul_predictions_unified.csv         # 최종 예측 (04·05 입력)
```

체크포인트를 **직접 다시 만들려면** (수 시간 소요):

```bash
python scripts/run_03b_experiments.py   # 튜닝 + LSTM 사다리
python scripts/run_v2_features.py       # 루프 v2
python scripts/run_loop2.py             # 루프 v3 + LSTM-D
python scripts/run_04_scheduling.py     # 전략 비교·민감도
python scripts/run_04d_quantile.py      # P10 분위수
python scripts/run_04f_rolling.py       # 롤링 재계획
```

각 노트북 맨 위에 **의존성 체크 셀**이 있어, 파일이 없으면 무엇이 없는지 알려준다.

## 시작하기

```bash
git clone https://github.com/kimyhleo-hub/turbofan-predictive-maintenance.git
cd turbofan-predictive-maintenance
pip install -r requirements.txt
```

`data/README.md`를 참고해 C-MAPSS를 `data/raw/`에 배치한 뒤 노트북을 `01 → 05` 순으로 실행한다.

원본 노트북을 덮어쓰지 않고 전체 실행 여부를 검증하려면 아래 명령을 사용한다. 실행본은
`runs/notebooks/`에 저장되며 Git에는 올라가지 않는다.

```bash
python scripts/run_notebooks.py
```

전략 비교 표와 민감도 분석 CSV를 만들려면 아래 명령을 사용한다. 결과는
`artifacts/`에 저장되며 Git에는 올라가지 않는다.

```bash
python scripts/run_experiment.py
```

## 참고문헌

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *IEEE PHM*.
- Özcan, H. (2025). Interpretable ensemble remaining useful life prediction enables dynamic maintenance scheduling for aircraft engines. *Scientific Reports*, 15:39795.
- Wang, L., Zhao, X., & Pham, H. (2026). Bi-objective predictive maintenance optimization for aero-engines: Mathematical models and metaheuristic algorithms. *IISE Transactions*.
- Saha, D. (2026). Remaining useful life prediction and maintenance schedule designing for turbofan engines considering multistage change points and uncertainty awareness. *Annals of Operations Research*.
