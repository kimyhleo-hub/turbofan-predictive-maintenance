"""04c — 통계적 검증 + 성능 평가 지표 (Lecture 12·13 방법론).

L13 워크플로: 단일 실행 함정 회피(N=10 시드) -> 게이트키퍼(Shapiro-Wilk 정규성,
Levene 등분산) -> 통과 시 t-test / 실패 시 Mann-Whitney U.
  비교 1 (핵심 주장): SA-강건(w=-0.3) vs SA-효율(w=+0.4) — replay 미정비 고장 수
  비교 2: SA-강건 분포 vs EDF(결정론적, 78) — Wilcoxon signed-rank + 승률
L12 지표: Success Rate(최적 대비 5% 이내), Feasible Rate(Repair로 100% 구조 보장),
Best/Worst ratio, mean±std, FEV(SA=반복당 1콜), 수렴 밴드(단일 곡선 금지).
"""
from pathlib import Path
import json, time, sys
import numpy as np, pandas as pd
from scipy import stats
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"src"/"pipeline.py").exists(): ROOT=cand; break
sys.path.insert(0,str(ROOT))
from src.pipeline import (make_deadlines, make_part_types, edf_schedule,
                          multistart_simulated_annealing, count_failures, wasted_life)
def log(*a): print(*a, flush=True)
PROC=ROOT/"data"/"processed"; CK=ROOT/"artifacts"/"04_scheduling"
FIG=ROOT/"reports"/"figures"/"scheduling"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", font_scale=0.9)
try: plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
CPW=13; T=10; K=75; N=10
fleet=pd.read_csv(PROC/"fleet_rul_predictions_unified.csv")
pred=fleet.pred_RUL.to_numpy(float); true=fleet.true_RUL.to_numpy(float)
part=make_part_types(np.arange(1,len(fleet)+1),4)
d_plan=make_deadlines(pred,CPW,0); d_plan_m1=make_deadlines(pred,CPW,1); d_true=make_deadlines(true,CPW,0)

ckp=CK/"stats_runs.json"
if not ckp.exists():
    rows=[]
    for i in range(N):
        for name,dq,w in [("robust",d_plan,-0.3),("efficiency",d_plan_m1,+0.4)]:
            t0=time.time()
            s,_=multistart_simulated_annealing(dq,T,K,n_starts=3,n_iter=8000,temp=10.0,cooling=0.997,
                                               seed=1000+37*i,c_pm=1.0,c_fail=18.0,w=w,
                                               part_type=part,setup_cost=5.0)
            rows.append({"grp":name,"seed":1000+37*i,"fail":int(count_failures(s,d_true)),
                         "waste":float(wasted_life(s,d_true)),"sec":round(time.time()-t0,1),
                         "fev":3*8000})
            log(f"  [{name:10s}] seed{i}: 고장={rows[-1]['fail']:3d} 낭비={rows[-1]['waste']:6.0f} ({rows[-1]['sec']}s)")
    json.dump(rows,open(ckp,"w"),indent=1)
runs=pd.DataFrame(json.load(open(ckp)))
rob=runs[runs.grp=="robust"]["fail"].to_numpy(float)
eff=runs[runs.grp=="efficiency"]["fail"].to_numpy(float)
EDF_FAIL=int(count_failures(edf_schedule(d_plan_m1,T,K),d_true))
log(f"\n[분포] SA-강건 {sorted(rob.astype(int))} | SA-효율 {sorted(eff.astype(int))} | EDF={EDF_FAIL}")

out={"N":N,"EDF":EDF_FAIL,
     "robust":{"mean":float(rob.mean()),"std":float(rob.std(ddof=1)),"best":int(rob.min()),"worst":int(rob.max())},
     "efficiency":{"mean":float(eff.mean()),"std":float(eff.std(ddof=1)),"best":int(eff.min()),"worst":int(eff.max())}}

# ---- L13: 게이트키퍼 -> 검정 ----
sw_r=stats.shapiro(rob); sw_e=stats.shapiro(eff); lev=stats.levene(rob,eff)
out["gatekeepers"]={"shapiro_robust_p":round(float(sw_r.pvalue),4),
                    "shapiro_eff_p":round(float(sw_e.pvalue),4),"levene_p":round(float(lev.pvalue),4)}
normal = sw_r.pvalue>0.05 and sw_e.pvalue>0.05
if normal:
    tt=stats.ttest_ind(rob,eff,equal_var=lev.pvalue>0.05)
    out["test"]={"name":"t-test" if lev.pvalue>0.05 else "Welch t-test","p":float(tt.pvalue)}
else:
    mw=stats.mannwhitneyu(rob,eff,alternative="less")
    out["test"]={"name":"Mann-Whitney U (단측: 강건<효율)","p":float(mw.pvalue)}
log(f"[게이트키퍼] Shapiro 강건 p={out['gatekeepers']['shapiro_robust_p']} / 효율 p={out['gatekeepers']['shapiro_eff_p']} / Levene p={out['gatekeepers']['levene_p']}")
log(f"[검정] {out['test']['name']}: p={out['test']['p']:.2e} -> {'H0 기각(차이 실재)' if out['test']['p']<0.05 else 'H0 기각 실패'}")

# vs EDF (결정론): Wilcoxon signed-rank (강건-78) + 승률
wx=stats.wilcoxon(rob-EDF_FAIL, alternative="less")
out["vs_EDF"]={"wilcoxon_p":float(wx.pvalue),"win_rate":float((rob<EDF_FAIL).mean())}
log(f"[vs EDF] 승률 {out['vs_EDF']['win_rate']:.0%}, Wilcoxon p={wx.pvalue:.4f}")

# ---- L12: Effectiveness 지표 ----
for g,arr in [("robust",rob),("efficiency",eff)]:
    thr=arr.min()*1.05
    out[g].update({"success_rate_5pct":float((arr<=thr).mean()),
                   "best_worst_ratio":round(float(arr.min()/arr.max()),3),
                   "feasible_rate":1.0})     # Repair 전략: 구조적으로 100%
log(f"[지표] 강건: mean {out['robust']['mean']:.1f}±{out['robust']['std']:.1f}, best/worst {out['robust']['best']}/{out['robust']['worst']}"
    f" (ratio {out['robust']['best_worst_ratio']}), 성공률(5%) {out['robust']['success_rate_5pct']:.0%}")

# ---- 그림 O8: 상자그림 + EDF 기준선 ----
plt.figure(figsize=(6.4,4.2))
sns.boxplot(data=runs,x="grp",y="fail",hue="grp",order=["robust","efficiency"],
            palette={"robust":"#C44E52","efficiency":"#8E9A9A"},width=0.45,legend=False)
sns.stripplot(data=runs,x="grp",y="fail",order=["robust","efficiency"],color="black",size=4,alpha=0.6)
plt.axhline(EDF_FAIL,color="#55A868",ls="--",label=f"EDF = {EDF_FAIL} (결정론)")
plt.xticks([0,1],[f"SA-강건 (w=-0.3)\nN={N}",f"SA-효율 (w=+0.4)\nN={N}"])
plt.ylabel("미정비 고장 수 (replay)"); plt.xlabel("")
plt.title(f"단일 실행이 아닌 분포로 — {out['test']['name']} p={out['test']['p']:.1e}")
plt.legend(); plt.tight_layout()
plt.savefig(FIG/"O8_stat_validation.png",dpi=110,bbox_inches="tight"); plt.close()
json.dump(out,open(CK/"stats_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
log("[완료] stats_summary.json + O8")
