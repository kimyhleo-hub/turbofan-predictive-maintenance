"""v2 결과 적대적 검증 — 극적 개선(valid -4.9)이 진짜인지.
A) 스파이크 누수테스트 확장(onset/cum/coup 포함)
B) 물리 설명: onset 이후 고장까지 소요시간의 산포 (작으면 onset_age가 정당한 강신호)
C) split 시드 3개 재현성 (split 운 배제)
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import importlib.util, sys

def log(*a): print(*a, flush=True)
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"data"/"processed").exists(): ROOT=cand; break
PROC=ROOT/"data"/"processed"
spec=importlib.util.spec_from_file_location("v2", ROOT/"scripts"/"run_v2_features.py")
# run_v2 스크립트는 임포트 시 전체 실행되므로 함수만 복사해 재정의 (독립 검증)
SEED=42; CLIP=125; FD_LIST=["FD001","FD002","FD003","FD004"]
BEST=dict(num_leaves=63, learning_rate=0.05, min_child_samples=60,
          colsample_bytree=0.7, subsample=0.7, reg_lambda=1.0)
FLIP=["s7","s12","s15","s20","s21"]
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))

def add_v2(df, union):
    df=df.sort_values(["engine_id","cycle"]).reset_index(drop=True)
    g=df.groupby("engine_id", sort=False)
    for s in union:
        z=df[f"z_{s}"]
        em=g[f"z_{s}"].expanding().mean().reset_index(level=0,drop=True)
        cnt=g.cumcount()+1
        bm=em.where(cnt<=10, np.nan).groupby(df["engine_id"]).ffill()
        df[f"b_{s}"]=z-bm
    babs=df[[f"b_{s}" for s in union]].abs()
    df["v2_H"]=babs.mean(axis=1, skipna=True)
    df["v2_H_ma20"]=df.groupby("engine_id",sort=False)["v2_H"].rolling(20,min_periods=1).mean().reset_index(level=0,drop=True)
    thr=0.3
    df["v2_cum"]=(df["v2_H"]-thr).clip(lower=0).groupby(df["engine_id"]).cumsum()
    on=(df["v2_H_ma20"]>thr).groupby(df["engine_id"]).cummax()
    df["v2_onset_age"]=on.groupby(df["engine_id"]).cumsum().astype(float)
    a=df["z_s9"]; b=df["z_s14"]; df["_ab"]=a*b; df["_a2"]=a*a; df["_b2"]=b*b
    def rm(c): return df.groupby("engine_id",sort=False)[c].rolling(20,min_periods=5).mean().reset_index(level=0,drop=True)
    ma,mb,mab,ma2,mb2=rm("z_s9"),rm("z_s14"),rm("_ab"),rm("_a2"),rm("_b2")
    cov=mab-ma*mb; va=(ma2-ma*ma).clip(lower=1e-12); vb=(mb2-mb*mb).clip(lower=1e-12)
    df["v2_coup914"]=(cov/np.sqrt(va*vb)).fillna(0.0).clip(-1,1)
    df.drop(columns=["_ab","_a2","_b2"],inplace=True)
    for s in FLIP: df[f"v2_abs_b_{s}"]=df[f"b_{s}"].abs()
    return df

manifest=json.loads((PROC/"feature_manifest.json").read_text(encoding="utf-8"))
UNION=manifest["unified"]["union_sensors"]
uni=pd.read_csv(PROC/"train_features_unified.csv")
uni=add_v2(uni,UNION)
META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
BASE=[c for c in uni.columns if c not in META and not c.startswith(("b_","v2_"))]
V2ALL=[f"b_{s}" for s in UNION]+["v2_H","v2_H_ma20","v2_cum","v2_onset_age","v2_coup914"]+[f"v2_abs_b_{s}" for s in FLIP]
uni["dataset_id"]=uni["dataset_id"].astype("category"); uni["condition_id"]=uni["condition_id"].astype("category")
CATS=["dataset_id","condition_id"]

# ---------- A) 스파이크 누수테스트 (전 v2 특징) ----------
log("[A] 스파이크 누수테스트 — 마지막 cycle 오염이 과거 v2 특징을 바꾸는가")
eng="FD004_1"
sub=uni[uni.engine_id==eng].copy()
raw=sub[[c for c in sub.columns if not c.startswith(("b_","v2_"))]].copy()
spk=raw.copy(); spk.loc[spk.cycle==spk.cycle.max(), [f"z_{s}" for s in UNION]]+=100
A=add_v2(raw,UNION); B=add_v2(spk,UNION)
early=A["cycle"]<A["cycle"].max()          # 마지막 1 cycle 제외 전부
bad=[]
for c in V2ALL:
    ok=np.allclose(A.loc[early,c].fillna(-9e9), B.loc[early,c].fillna(-9e9))
    if not ok: bad.append(c)
log("  전 특징 과거값 불변:", "PASS" if not bad else f"FAIL {bad}")
assert not bad

# ---------- B) 물리 설명 — onset 이후 고장까지 소요시간 ----------
log("[B] onset -> 고장 소요시간 분포 (train 엔진)")
g=uni.groupby("engine_id")
life=g["cycle"].max()
onset_t=g.apply(lambda x: x.loc[x["v2_onset_age"]>0,"cycle"].min() if (x["v2_onset_age"]>0).any() else np.nan, include_groups=False)
dur=(life-onset_t).dropna()
cover=onset_t.notna().mean()
log(f"  onset 감지 엔진 비율: {cover:.0%}")
log(f"  onset->고장 소요: 평균 {dur.mean():.0f} / std {dur.std():.0f} / CV {dur.std()/dur.mean():.2f}")
log(f"  (참고) 총수명:      평균 {life.mean():.0f} / std {life.std():.0f} / CV {life.std()/life.mean():.2f}")
log("  => 소요시간 CV가 총수명 CV보다 작으면 onset_age의 예측력이 물리적으로 정당")
r=np.corrcoef(uni["v2_onset_age"], uni["RUL"])[0,1]
log(f"  corr(onset_age, RUL) = {r:+.3f}")

# ---------- C) split 시드 3개 재현성 ----------
log("[C] 다른 엔진 분리(seed 7/123/2026)에서도 개선이 재현되는가")
def run_seed(seed):
    tr_idx,va_idx=[],[]
    for _,gg in uni.groupby("dataset_id",observed=True):
        gss=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=seed)
        i,j=next(gss.split(gg,groups=gg["engine_id"]))
        tr_idx+=list(gg.index[i]); va_idx+=list(gg.index[j])
    tr_idx=sorted(tr_idx); va_idx=sorted(va_idx)
    TR,VA=uni.loc[tr_idx],uni.loc[va_idx]
    out={}
    for name,xc in [("v1",BASE+CATS),("v2",BASE+V2ALL+CATS)]:
        m=lgb.LGBMRegressor(n_estimators=2000,subsample_freq=1,random_state=SEED,verbose=-1,**BEST)
        m.fit(TR[xc],TR["RUL"],eval_set=[(VA[xc],VA["RUL"])],eval_metric="rmse",
              categorical_feature=CATS,callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(0)])
        out[name]=round(rmse(VA["RUL"],np.clip(m.predict(VA[xc]),0,CLIP)),3)
    return out
for sd in (7,123,2026):
    t0=time.time(); o=run_seed(sd)
    log(f"  seed {sd}: v1={o['v1']}  v2={o['v2']}  Δ={o['v2']-o['v1']:+.2f}  ({time.time()-t0:.0f}s)")
log("[완료]")
