"""전처리 v2 — 보완 EDA로 검증된 특징의 구현과 ablation (루프 1회차).

특징군 (전부 기존 unified CSV의 z_ 위에서 과거-전용으로 파생):
  G1 자기-기준선 b_s = z_s - own_baseline (초기 제조 오프셋 제거; EDA 보완①)
  G2 복합 건강지표: H=mean|b|, H_ma20, 누적손상 cum, onset_age (EDA piecewise)
  G3 s9-s14 결합 출현: rolling corr20 (EDA 보완⑥, 0.09→0.63)
  G4 모드반전 센서 크기: |b| for s7,s12,s15,s20,s21 (EDA §7)

Ablation: base(136) → +G1 → +G1G2 → +G1G2G3 → +ALL, 같은 split·BEST 파라미터.
개선 시 전체 train 재학습 → test 1회 (v2 라운드로 명시 보고).
체크포인트: artifacts/03_model/
"""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

def log(*a): print(*a, flush=True)

for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"data"/"processed").exists(): ROOT=cand; break
PROC=ROOT/"data"/"processed"; RAW=ROOT/"data"/"raw"
CK=ROOT/"artifacts"/"03_model"; CK.mkdir(parents=True, exist_ok=True)
SEED=42; CLIP=125; FD_LIST=["FD001","FD002","FD003","FD004"]
BEST=dict(num_leaves=63, learning_rate=0.05, min_child_samples=60,
          colsample_bytree=0.7, subsample=0.7, reg_lambda=1.0)   # 03b 튜닝 결과
FLIP=["s7","s12","s15","s20","s21"]

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def phm08(y,p):
    d=np.asarray(p,float)-np.asarray(y,float)
    return float(np.sum(np.where(d<0,np.exp(-d/13)-1,np.exp(d/10)-1)))

# ---------------- v2 특징 파생 (과거 전용) ----------------
def add_v2(df, union):
    df=df.sort_values(["engine_id","cycle"]).reset_index(drop=True)
    g=df.groupby("engine_id", sort=False)
    zc=[f"z_{s}" for s in union]
    # G1: 자기-기준선 — 첫 ≤10 cycle의 expanding mean으로 고정 (과거 전용)
    for s in union:
        z=df[f"z_{s}"]
        em=g[f"z_{s}"].expanding().mean().reset_index(level=0,drop=True)
        cnt=g.cumcount()+1
        bm=em.where(cnt<=10, np.nan)
        bm=bm.groupby(df["engine_id"]).ffill()          # 10cycle 이후엔 그때 값 고정
        df[f"b_{s}"]=z-bm
    babs=df[[f"b_{s}" for s in union]].abs()
    # G2: 복합 건강지표
    df["v2_H"]=babs.mean(axis=1, skipna=True)
    df["v2_H_ma20"]=g_apply_roll(df,"v2_H",20)
    thr=0.3
    inc=(df["v2_H"]-thr).clip(lower=0)
    df["v2_cum"]=inc.groupby(df["engine_id"]).cumsum()
    on=(df["v2_H_ma20"]>thr)
    onset_idx=on.groupby(df["engine_id"]).cummax()      # 한 번 넘으면 유지
    age=onset_idx.groupby(df["engine_id"]).cumsum()     # 넘은 이후 경과 cycle 수
    df["v2_onset_age"]=age.astype(float)
    # G3: s9-s14 결합 출현 — rolling corr = cov/sqrt(var·var), apply 없이 벡터화
    a=df["z_s9"]; b=df["z_s14"]; df["_ab"]=a*b; df["_a2"]=a*a; df["_b2"]=b*b
    def rm(col): return df.groupby("engine_id",sort=False)[col].rolling(20,min_periods=5).mean().reset_index(level=0,drop=True)
    ma,mb,mab,ma2,mb2=rm("z_s9"),rm("z_s14"),rm("_ab"),rm("_a2"),rm("_b2")
    cov=mab-ma*mb; va=(ma2-ma*ma).clip(lower=1e-12); vb=(mb2-mb*mb).clip(lower=1e-12)
    df["v2_coup914"]=(cov/np.sqrt(va*vb)).fillna(0.0).clip(-1,1)
    df.drop(columns=["_ab","_a2","_b2"],inplace=True)
    # G4: 모드반전 센서 크기
    for s in FLIP: df[f"v2_abs_b_{s}"]=df[f"b_{s}"].abs()
    return df

def g_apply_roll(df,col,w):
    return df.groupby("engine_id",sort=False)[col].rolling(w,min_periods=1).mean().reset_index(level=0,drop=True)

# ---------------- 데이터 ----------------
log("[load] unified ...")
manifest=json.loads((PROC/"feature_manifest.json").read_text(encoding="utf-8"))
UNION=manifest["unified"]["union_sensors"]
uni=pd.read_csv(PROC/"train_features_unified.csv")
te =pd.read_csv(PROC/"test_features_unified.csv")
t0=time.time(); uni=add_v2(uni,UNION); te=add_v2(te,UNION)
log(f"[v2] 특징 파생 완료 ({time.time()-t0:.0f}s)")

META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
BASE=[c for c in uni.columns if c not in META and not c.startswith(("b_","v2_"))]
G1=[f"b_{s}" for s in UNION]
G2=["v2_H","v2_H_ma20","v2_cum","v2_onset_age"]
G3=["v2_coup914"]; G4=[f"v2_abs_b_{s}" for s in FLIP]
assert len(BASE)==136, len(BASE)
for d_ in (uni,te):
    d_["dataset_id"]=d_["dataset_id"].astype("category"); d_["condition_id"]=d_["condition_id"].astype("category")
CATS=["dataset_id","condition_id"]

# V-v2: 누수 스팟체크 — 마지막 cycle에 스파이크를 넣어도 과거 v2 특징 불변
chk=uni[uni.engine_id=="FD001_1"].copy()
spk=chk.copy(); spk.loc[spk["cycle"]==spk["cycle"].max(), [f"z_{s}" for s in UNION]] += 100
a=add_v2(chk.drop(columns=[c for c in chk.columns if c.startswith(("b_","v2_"))]),UNION)
b=add_v2(spk.drop(columns=[c for c in spk.columns if c.startswith(("b_","v2_"))]),UNION)
early=a["cycle"]<a["cycle"].max()-20
assert np.allclose(a.loc[early,"v2_H"],b.loc[early,"v2_H"]) and np.allclose(a.loc[early,"v2_cum"],b.loc[early,"v2_cum"]), "[V-v2] 미래 누수!"
log("[V-v2] v2 특징 미래 미사용 OK")

tr_idx,va_idx=[],[]
for _,g in uni.groupby("dataset_id",observed=True):
    gss=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=SEED)
    i,j=next(gss.split(g,groups=g["engine_id"]))
    tr_idx+=list(g.index[i]); va_idx+=list(g.index[j])
tr_idx=sorted(tr_idx); va_idx=sorted(va_idx)
TR,VA=uni.loc[tr_idx],uni.loc[va_idx]
va_ds=VA["dataset_id"].astype(str).to_numpy(); y_va=VA["RUL"].to_numpy(float)

def run(name,xcols):
    m=lgb.LGBMRegressor(n_estimators=4000,subsample_freq=1,random_state=SEED,verbose=-1,**BEST)
    m.fit(TR[xcols],TR["RUL"],eval_set=[(VA[xcols],VA["RUL"])],eval_metric="rmse",
          categorical_feature=CATS,callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(0)])
    p=np.clip(m.predict(VA[xcols]),0,CLIP)
    r=rmse(y_va,p); perfd={fd: round(rmse(y_va[va_ds==fd],p[va_ds==fd]),2) for fd in FD_LIST}
    log(f"  {name:22s} valid={r:.3f}  {perfd}  (iter {m.best_iteration_})")
    return {"name":name,"valid":round(r,3),"perfd":perfd,"iter":int(m.best_iteration_),"xcols_n":len(xcols)}

log("[ablation] v2 특징군 누적 추가")
res=[]
res.append(run("base(v1 136)",           BASE+CATS))
res.append(run("+G1 자기기준선(17)",       BASE+G1+CATS))
res.append(run("+G1G2 +건강지표(4)",      BASE+G1+G2+CATS))
res.append(run("+G1G2G3 +결합출현(1)",    BASE+G1+G2+G3+CATS))
res.append(run("+ALL +모드크기(5)",       BASE+G1+G2+G3+G4+CATS))

best=min(res,key=lambda r:r["valid"])
log(f"[선택] {best['name']} valid={best['valid']} (base 대비 {best['valid']-res[0]['valid']:+.3f})")

out={"ablation":res,"best":best}
if best["valid"] < res[0]["valid"]-0.03:     # 유의 개선 시에만 test
    pick={"+G1 자기기준선(17)":BASE+G1,"+G1G2 +건강지표(4)":BASE+G1+G2,
          "+G1G2G3 +결합출현(1)":BASE+G1+G2+G3,"+ALL +모드크기(5)":BASE+G1+G2+G3+G4}[best["name"]]
    xcols=pick+CATS
    log("[test] v2 최적 구성 — 전체 train 재학습 후 test 1회 (v2 라운드)")
    fm=lgb.LGBMRegressor(n_estimators=best["iter"],subsample_freq=1,random_state=SEED,verbose=-1,**BEST)
    fm.fit(uni[xcols],uni["RUL"],categorical_feature=CATS)
    last=te.sort_values(["dataset_id","unit","cycle"]).groupby(["dataset_id","unit"],observed=True).tail(1)
    true=np.concatenate([pd.read_csv(RAW/f"RUL_{fd}.txt",sep=r"\s+",header=None)[0].to_numpy() for fd in FD_LIST]).astype(float)
    p=np.clip(fm.predict(last[xcols]),0,CLIP); ds=last["dataset_id"].astype(str).to_numpy()
    yc=np.minimum(true,CLIP)
    tst={"RMSE_cap":round(rmse(yc,p),2),"PHM08_cap":round(phm08(yc,p),0),
         **{fd: round(rmse(np.minimum(true[ds==fd],CLIP),p[ds==fd]),2) for fd in FD_LIST}}
    log(f"[test v2] {tst}")
    out["test_v2"]=tst
else:
    log("[판정] 유의 개선 없음(<0.03) — test 미실시, 루프 종료 근거로 기록")
(CK/"v2_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding="utf-8")
log("[완료] v2_summary.json 저장")
