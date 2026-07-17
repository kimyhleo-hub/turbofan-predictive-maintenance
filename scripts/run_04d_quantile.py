"""04d — P10 분위수 마감: 균일 버퍼(w<0) -> 엔진별 맞춤 버퍼(불확실성 정량화).

Q1 quantile LGBM(alpha=0.1, v3 특징셋) 학습 -> test P10 예측
Q2 검증: 커버리지 P(true>=P10) ~ 0.9, 서브셋별 버퍼 크기(pred-P10) — 맞춤성 증거
Q3 P10 마감으로 SA 3변형(효율 w=+0.4 / 중립 w=0 / 강건 w=-0.3) + EDF -> replay
Q4 현재 챔피언(점예측+강건 SA=48)과 비교, oracle 격차 분해 갱신
"""
from pathlib import Path
import json, time, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"src"/"pipeline.py").exists(): ROOT=cand; break
sys.path.insert(0,str(ROOT))
from src.pipeline import (make_deadlines, make_part_types, edf_schedule,
                          multistart_simulated_annealing, schedule_metrics, count_failures, wasted_life)
def log(*a): print(*a, flush=True)
PROC=ROOT/"data"/"processed"; RAW=ROOT/"data"/"raw"; CK=ROOT/"artifacts"/"04_scheduling"
src=Path("scripts/run_loop2.py").read_text(encoding="utf-8")
ns={}; exec(src.split('log("[load]')[0], ns)
add_v23=ns["add_v23"]; BEST=ns["BEST"]; FLIP=ns["FLIP"]
CPW=13; T=10; K=75; SEED=42; FD=["FD001","FD002","FD003","FD004"]

# ---- Q1: quantile 모델 ----
p10_path=CK/"pred_p10.npy"
manifest=json.loads((PROC/"feature_manifest.json").read_text(encoding="utf-8"))
UNION=manifest["unified"]["union_sensors"]
if not p10_path.exists():
    log("[Q1] quantile LGBM(alpha=0.1) 학습 — v3 특징셋")
    uni=pd.read_csv(PROC/"train_features_unified.csv"); te=pd.read_csv(PROC/"test_features_unified.csv")
    uni=add_v23(uni,UNION); te=add_v23(te,UNION)
    META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
    BASE=[c for c in uni.columns if c not in META and not c.startswith(("b_","v2_","v3_"))]
    V2=[f"b_{s}" for s in UNION]+["v2_H","v2_H_ma20","v2_cum","v2_onset_age","v2_coup914"]+[f"v2_abs_b_{s}" for s in FLIP]
    V3=["v3_onset02","v3_onset05","v3_onset_flag","v3_H_margin"]
    for d in (uni,te):
        d["dataset_id"]=d["dataset_id"].astype("category"); d["condition_id"]=d["condition_id"].astype("category")
    CATS=["dataset_id","condition_id"]; X=BASE+V2+V3+CATS
    ti,vi=[],[]
    for _,g in uni.groupby("dataset_id",observed=True):
        gss=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=SEED)
        a,b=next(gss.split(g,groups=g["engine_id"])); ti+=list(g.index[a]); vi+=list(g.index[b])
    ti=sorted(ti); vi=sorted(vi)
    m=lgb.LGBMRegressor(objective="quantile",alpha=0.1,n_estimators=4000,subsample_freq=1,
                        random_state=SEED,verbose=-1,**BEST)
    m.fit(uni.loc[ti,X],uni.loc[ti,"RUL"],eval_set=[(uni.loc[vi,X],uni.loc[vi,"RUL"])],
          eval_metric="quantile",categorical_feature=CATS,
          callbacks=[lgb.early_stopping(200,verbose=False)])
    it=m.best_iteration_
    fm=lgb.LGBMRegressor(objective="quantile",alpha=0.1,n_estimators=it,subsample_freq=1,
                         random_state=SEED,verbose=-1,**BEST)
    fm.fit(uni[X],uni["RUL"],categorical_feature=CATS)
    last=te.sort_values(["dataset_id","unit","cycle"]).groupby(["dataset_id","unit"],observed=True).tail(1)
    p10=np.clip(fm.predict(last[X]),0,125)
    np.save(p10_path,p10); log(f"  iter={it}, P10 저장")
p10=np.load(p10_path)

fleet=pd.read_csv(PROC/"fleet_rul_predictions_unified.csv")
pred=fleet.pred_RUL.to_numpy(float); true=fleet.true_RUL.to_numpy(float)
ds=fleet.dataset_id.to_numpy(); part=make_part_types(np.arange(1,len(fleet)+1),4)
d_true=make_deadlines(true,CPW,0); d_p10=make_deadlines(p10,CPW,0)

# ---- Q2: 커버리지·맞춤성 ----
cov=float((true>=p10).mean())
log(f"[Q2] 커버리지 P(true>=P10) = {cov:.1%} (목표 ~90%)")
buf=pred-p10
for f in FD:
    s=ds==f
    log(f"  {f}: 버퍼(pred-P10) 평균 {buf[s].mean():5.1f}cyc | P10 커버리지 {float((true[s]>=p10[s]).mean()):.0%}")
log(f"  d_P10 분포: {np.bincount(np.minimum(d_p10,T),minlength=T+1)[1:].tolist()} (총 {len(d_p10)}, 슬롯 {T*K})")

# ---- Q3: P10 마감 스케줄링 ----
res=[]
def ev(s,name):
    st=schedule_metrics(s,d_true,part,1.0,5.0,0.1,5.0)
    r={"구성":name,"고장":int(st["failures"]),"낭비":round(float(st["wasted_life"]),0),
       "setup":int(st["setups"]),"비용":round(float(st["total_cost"]),1)}
    log(f"  {name:28s} 고장={r['고장']:3d} 낭비={r['낭비']:6.0f} setup={r['setup']} 비용={r['비용']:7.1f}")
    res.append(r); return r
log("[Q3] P10 마감 replay")
ev(edf_schedule(d_p10,T,K),"EDF @ P10")
for name,w in [("SA-효율 (w=+0.4) @ P10",0.4),("SA-중립 (w=0) @ P10",0.0),("SA-강건 (w=-0.3) @ P10",-0.3)]:
    s,_=multistart_simulated_annealing(d_p10,T,K,n_starts=3,n_iter=8000,temp=10.0,cooling=0.997,
                                       seed=SEED,c_pm=1.0,c_fail=18.0,w=w,part_type=part,setup_cost=5.0)
    ev(s,name)
log("  (기준: SA-강건 @ 점예측 = 고장 48 낭비 1133 비용 1212 | Oracle-SA = 4)")
json.dump({"coverage":cov,"buffer_perfd":{f: round(float(buf[ds==f].mean()),1) for f in FD},
           "results":res},open(CK/"p10_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
log("[완료] p10_summary.json")
