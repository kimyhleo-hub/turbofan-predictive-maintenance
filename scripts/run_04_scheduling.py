"""04 스케줄링 최적화 — SA 메인, baseline 비교, replay 평가, 민감도, 예측품질 실험.

정식화 (README):
  d_i = ceil(pred_RUL/CPW) - margin,  x[i,t]: 창 배정,  용량 K/창,  part 4종 setup 배칭
  계획은 예측 마감으로, 평가는 실제 고장(d_true = ceil(true_RUL/CPW))에 replay.
지표: 미정비 고장 수(주), 낭비 수명(다운타임 proxy), 총비용, setup 수, 실행시간.
src/pipeline.py의 EDF/greedy/SA/multistart/replay 전부 재사용.
"""
from pathlib import Path
import json, time, sys
import numpy as np, pandas as pd
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"src"/"pipeline.py").exists(): ROOT=cand; break
sys.path.insert(0,str(ROOT))
from src.pipeline import (make_deadlines, make_part_types, edf_schedule, latest_deadline_schedule,
                          grouped_greedy_schedule, random_schedule, simulated_annealing,
                          multistart_simulated_annealing, schedule_metrics, count_failures, wasted_life)
def log(*a): print(*a, flush=True)
PROC=ROOT/"data"/"processed"; CK=ROOT/"artifacts"/"04_scheduling"; CK.mkdir(parents=True,exist_ok=True)
FIG=ROOT/"reports"/"figures"/"scheduling"; FIG.mkdir(parents=True,exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", font_scale=0.9)
try: plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass

# ---------------- 설정 ----------------
CPW=13                 # 창 하나 = 13 cycle -> pred<=125 => d<=10
T=10; K=75             # 슬롯 750 / 707엔진 (슬랙 1.06)
CPM, CFAIL, WW, SETUP = 1.0, 5.0, 0.1, 5.0
SEED=42

fleet=pd.read_csv(PROC/"fleet_rul_predictions_unified.csv")
pred=fleet["pred_RUL"].to_numpy(float); true=fleet["true_RUL"].to_numpy(float)
units=np.arange(1,len(fleet)+1)
part=make_part_types(units, 4)
d_true=make_deadlines(true, CPW, 0)      # 실제 고장 창 (마진 없음)
log(f"[설정] 엔진 {len(fleet)} | CPW={CPW} T={T} K={K} | d_pred 분포(마진0): "
    f"{np.bincount(make_deadlines(pred,CPW,0), minlength=T+1)[1:].tolist()}")

def evaluate(schedule, name, t0=None):
    """예측으로 만든 스케줄을 실제 고장에 replay."""
    m=schedule_metrics(schedule, d_true, part, c_pm=CPM, c_fail=CFAIL, w=WW, setup_cost=SETUP)
    m={"전략":name,"미정비고장":m["failures"],"낭비수명(창)":round(m["wasted_life"],0),
       "setup":m["setups"],"총비용":round(m["total_cost"],1),
       "미배정":int((schedule==0).sum()),
       "시간(s)":round(time.time()-t0,2) if t0 else 0.0}
    log(f"  {name:22s} 고장={m['미정비고장']:3d} 낭비={m['낭비수명(창)']:6.0f} setup={m['setup']:3d} 비용={m['총비용']:7.1f} ({m['시간(s)']}s)")
    return m

def run_strategies(d_plan, tag, sa_record=False):
    rows=[]; extra={}
    t0=time.time(); s=np.zeros(len(d_plan),int)
    for sd in range(5):
        s_=random_schedule(d_plan,T,K,seed=sd)
        rows.append(("Random",s_)) if sd==0 else None
    # Random은 5시드 평균 지표로
    rs=[schedule_metrics(random_schedule(d_plan,T,K,seed=sd), d_true, part, CPM,CFAIL,WW,SETUP) for sd in range(5)]
    m={"전략":"Random(5seed평균)","미정비고장":round(np.mean([r["failures"] for r in rs]),1),
       "낭비수명(창)":round(np.mean([r["wasted_life"] for r in rs]),0),
       "setup":round(np.mean([r["setups"] for r in rs]),0),
       "총비용":round(np.mean([r["total_cost"] for r in rs]),1),"미배정":"-","시간(s)":round(time.time()-t0,2)}
    log(f"  {'Random(5seed)':22s} 고장={m['미정비고장']:5} 낭비={m['낭비수명(창)']:6} 비용={m['총비용']:7}")
    out=[m]
    for name,fn in [("EDF",lambda: edf_schedule(d_plan,T,K)),
                    ("Latest",lambda: latest_deadline_schedule(d_plan,T,K)),
                    ("Grouped-greedy",lambda: grouped_greedy_schedule(d_plan,T,K,part))]:
        t0=time.time(); s_=fn(); out.append(evaluate(s_,name,t0))
    t0=time.time()
    res=multistart_simulated_annealing(d_plan,T,K,n_starts=3,n_iter=8000,temp=10.0,cooling=0.997,
                                       seed=SEED,c_pm=CPM,c_fail=CFAIL*3.6,w=WW*4,part_type=part,
                                       setup_cost=SETUP,record=sa_record)
    if sa_record: sa_s,sa_c,hist=res; extra["hist"]=hist
    else: sa_s,sa_c=res
    out.append(evaluate(sa_s,"SA (multistart 3)",t0))
    extra["sa_schedule"]=sa_s
    return out,extra

summ={}
# ---------------- S1: 메인 비교 (v3 예측, 마진 1창) ----------------
p1=CK/"s1.json"
log("[S1] 전략 비교 — v3 예측, 마진 1창, K=75")
d_plan=make_deadlines(pred,CPW,1)
rows,extra=run_strategies(d_plan,"v3",sa_record=True)
summ["main"]=rows; hist=extra["hist"]
np.save(CK/"sa_main.npy",extra["sa_schedule"])
plt.figure(figsize=(7,4))
plt.plot(hist,color="#4C72B0"); plt.xlabel("SA iteration"); plt.ylabel("best cost (예측 마감 기준)")
plt.title("SA 수렴 곡선 (multistart best, K=75)"); plt.tight_layout()
plt.savefig(FIG/"O2_sa_convergence.png",dpi=110,bbox_inches="tight"); plt.close()

# Oracle (실제 RUL로 계획) — 상한선
log("[S1b] Oracle — 실제 RUL로 SA")
t0=time.time()
o_s,_=multistart_simulated_annealing(d_true,T,K,n_starts=3,n_iter=8000,temp=10.0,cooling=0.997,
                                     seed=SEED,c_pm=CPM,c_fail=CFAIL*3.6,w=WW*4,part_type=part,setup_cost=SETUP)
summ["oracle"]=evaluate(o_s,"Oracle-SA",t0)

# ---------------- S2: 예측 품질 -> 결정 품질 ----------------
log("[S2] 예측 품질 실험 — 03 기본(15.02) vs v3(10.90) vs Oracle")
base_pred_path=CK/"pred_base.npy"
if not base_pred_path.exists():
    import lightgbm as lgb
    te=pd.read_csv(PROC/"test_features_unified.csv")
    for c in ("dataset_id","condition_id"): te[c]=te[c].astype("category")
    META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
    feats=[c for c in te.columns if c not in META]
    b=lgb.Booster(model_file=str(ROOT/"models"/"lgbm_unified.txt"))
    last=te.sort_values(["dataset_id","unit","cycle"]).groupby(["dataset_id","unit"],observed=True).tail(1)
    pb=np.clip(b.predict(last[feats+["dataset_id","condition_id"]]),0,125)
    np.save(base_pred_path,pb)
pb=np.load(base_pred_path)
rmse=lambda a,c: float(np.sqrt(np.mean((np.minimum(a,125)-c)**2)))
log(f"  기본예측 RMSE={rmse(true,pb):.2f} | v3 RMSE={rmse(true,pred):.2f}")
qual=[]
for name,pr in [("기본모델(15.0)",pb),("v3(10.9)",pred),("Oracle(0)",np.minimum(true,125))]:
    dq=make_deadlines(pr,CPW,1) if name!="Oracle(0)" else d_true
    t0=time.time()
    s_,_=multistart_simulated_annealing(dq,T,K,n_starts=3,n_iter=8000,temp=10.0,cooling=0.997,
                                        seed=SEED,c_pm=CPM,c_fail=CFAIL*3.6,w=WW*4,part_type=part,setup_cost=SETUP)
    r=evaluate(s_,f"SA @ {name}",t0); r["예측"]=name; qual.append(r)
summ["quality"]=qual

# ---------------- S3: 민감도 (K x margin, 비용비) ----------------
log("[S3] 민감도 — K x 안전마진 (SA, replay 고장 수)")
sens=[]
for Kk in (72,75,80,90):
    for mg in (0,1,2):
        dq=make_deadlines(pred,CPW,mg)
        s_,_=multistart_simulated_annealing(dq,T,Kk,n_starts=2,n_iter=6000,temp=10.0,cooling=0.997,
                                            seed=SEED,c_pm=CPM,c_fail=CFAIL*3.6,w=WW*4,part_type=part,setup_cost=SETUP)
        f=count_failures(s_,d_true); wlife=wasted_life(s_,d_true)
        sens.append({"K":Kk,"margin":mg,"고장":int(f),"낭비":round(wlife,0)})
        log(f"  K={Kk} margin={mg}: 고장={f} 낭비={wlife:.0f}")
summ["sens_km"]=sens
piv=pd.DataFrame(sens).pivot(index="margin",columns="K",values="고장")
plt.figure(figsize=(6.2,3.8))
sns.heatmap(piv,annot=True,fmt="d",cmap="RdYlGn_r",cbar_kws={"label":"미정비 고장 수"})
plt.title("민감도: 용량 K × 안전마진 (SA, replay)"); plt.ylabel("안전마진(창)")
plt.tight_layout(); plt.savefig(FIG/"O3_sensitivity_K_margin.png",dpi=110,bbox_inches="tight"); plt.close()

log("[S3b] 비용비 민감도 — C_fail/C_pm ∈ {3,5,10} (SA 목적함수 가중, K=75 margin=1)")
cost_sens=[]
for cf in (3,5,10):
    s_,_=multistart_simulated_annealing(make_deadlines(pred,CPW,1),T,K,n_starts=2,n_iter=6000,
                                        temp=10.0,cooling=0.997,seed=SEED,c_pm=CPM,c_fail=float(cf)*3.6,
                                        w=WW*4,part_type=part,setup_cost=SETUP)
    m=schedule_metrics(s_,d_true,part,CPM,float(cf),WW,SETUP)
    cost_sens.append({"C_fail":cf,"고장":m["failures"],"낭비":round(m["wasted_life"],0),
                      "setup":m["setups"],"총비용":round(m["total_cost"],1)})
    log(f"  C_fail={cf}: {cost_sens[-1]}")
summ["sens_cost"]=cost_sens

# ---------------- 그림 O1/O4 ----------------
df=pd.DataFrame(summ["main"]); df=df[df["전략"]!="Random(5seed평균)"]
rnd=summ["main"][0]
fig,ax=plt.subplots(1,2,figsize=(12.5,4.4))
names=["Random"]+df["전략"].tolist()+["Oracle-SA"]
fails=[rnd["미정비고장"]]+df["미정비고장"].tolist()+[summ["oracle"]["미정비고장"]]
costs=[rnd["총비용"]]+df["총비용"].tolist()+[summ["oracle"]["총비용"]]
colors=["#8E9A9A","#DD8452","#B5A46B","#8172B3","#C44E52","#55A868"]
ax[0].bar(names,fails,color=colors); ax[0].set_ylabel("미정비 고장 수 (replay)"); ax[0].tick_params(axis="x",rotation=20)
for i,v in enumerate(fails): ax[0].text(i,v+0.5,str(v),ha="center",fontsize=9)
ax[0].set_title("주 지표 — 실제 고장으로 재현한 미정비 고장")
ax[1].bar(names,costs,color=colors); ax[1].set_ylabel("총비용"); ax[1].tick_params(axis="x",rotation=20)
ax[1].set_title(f"총비용 (C_pm=1, C_fail={CFAIL:.0f}, w={WW}, setup={SETUP:.0f})")
plt.tight_layout(); plt.savefig(FIG/"O1_strategy_comparison.png",dpi=110,bbox_inches="tight"); plt.close()

qd=pd.DataFrame(summ["quality"])
plt.figure(figsize=(6.6,4))
bars=plt.bar(qd["예측"],qd["미정비고장"],color=["#8E9A9A","#4C72B0","#55A868"])
for b,v in zip(bars,qd["미정비고장"]): plt.text(b.get_x()+b.get_width()/2,v+0.3,str(v),ha="center")
plt.ylabel("미정비 고장 수 (SA, replay)"); plt.title("예측 품질 → 정비 결정 품질")
plt.tight_layout(); plt.savefig(FIG/"O4_pred_quality_to_decision.png",dpi=110,bbox_inches="tight"); plt.close()

json.dump(summ,open(CK/"sched_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)
log("[완료] sched_summary.json + O1~O4 저장")
