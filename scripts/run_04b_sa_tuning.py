"""04b — SA 하이퍼파라미터 튜닝: 강의자료 방법론 (Day 6 SA + Hyper-Parameter Tuning Using SA).

강의 방법론 정렬:
  - T0 통계적 결정: 무작위 가용해 R개의 목적값 표준편차 -> T0 = -sigma/ln(p)   [Day6 p.23]
  - 지수 스케줄 T_{k+1} = alpha*T_k, 온도당 level length L회 반복               [Day6 p.18,21]
  - 자연 종료: T < t_end 또는 무개선 level 지속                                  [Day6 p.21]
  - 수용: dE<=0 즉시, 아니면 P = exp(-dE/T)                                     [Day6 p.15]
  - 완전요인 p x alpha, 셀당 N=10 독립 시드 반복, mean/std 히트맵 + 행/열 추세    [Tuning PDF]
고정 통제(공정 비교): 동일 문제(v3 예측 마감, margin0), 동일 초기해 방식(무작위 가용해),
동일 t_end, 동일 강건 목적(c_pm=1, c_fail=18, w=-0.3, setup=5).
응답: 주 = 최종 목적값 / 부 = level 수, 수용 이동 수, 실행시간.
최종: 승자 (p,alpha)로 재실행 -> replay -> 기존 SA-강건(고장 48)과 비교.
"""
from pathlib import Path
import json, time, sys
import numpy as np, pandas as pd
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"src"/"pipeline.py").exists(): ROOT=cand; break
sys.path.insert(0,str(ROOT))
from src.pipeline import (make_deadlines, make_part_types, random_schedule, edf_schedule,
                          total_cost, _neighbor, _repair, schedule_metrics, count_failures, wasted_life)
def log(*a): print(*a, flush=True)
PROC=ROOT/"data"/"processed"; CK=ROOT/"artifacts"/"04_scheduling"; CK.mkdir(parents=True,exist_ok=True)
FIG=ROOT/"reports"/"figures"/"scheduling"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", font_scale=0.9)
try: plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass

CPW=13; T=10; K=75
OBJ=dict(c_pm=1.0, c_fail=18.0, w=-0.3, setup_cost=5.0)   # S4 강건 목적 (조기배정=버퍼 보상)
L_LEVEL=700          # level length: 이웃공간(707x10=7070)의 ~10%              [Day6 p.21]
                     # (1차 시도 L=150은 예산 부족으로 미수렴 — cells_L150.json 보존)
T_END=0.05           # 자연 종료 온도 (dE=1 수용확률 ~ e^-20)
NO_IMP=8             # 무개선 level 정지
MAX_LEVEL=400
P_GRID=[0.6,0.7,0.8,0.9]; A_GRID=[0.7,0.8,0.9]; N_REP=10

fleet=pd.read_csv(PROC/"fleet_rul_predictions_unified.csv")
pred=fleet.pred_RUL.to_numpy(float); true=fleet.true_RUL.to_numpy(float)
part=make_part_types(np.arange(1,len(fleet)+1),4)
d_plan=make_deadlines(pred,CPW,0); d_true=make_deadlines(true,CPW,0)
cost=lambda s: total_cost(s,d_plan,OBJ["c_pm"],OBJ["c_fail"],OBJ["w"],part,OBJ["setup_cost"])

# ---- T0 통계적 결정: 무작위 가용해 R=30의 목적값 sigma ----
R=30
sig=float(np.std([cost(random_schedule(d_plan,T,K,seed=s)) for s in range(R)]))
log(f"[T0] 무작위 가용해 R={R}: sigma(C)={sig:.1f} -> T0(p)= " +
    ", ".join(f"p={p}: {-sig/np.log(p):.0f}" for p in P_GRID))

def sa_course(p0, alpha, seed, init="random"):
    """강의식 SA: 통계적 T0, 지수 스케줄+level, 자연 종료. 반환: 최종 best 목적값 + 부응답."""
    rng=np.random.default_rng(seed)
    if init=="heuristic":
        from src.pipeline import latest_deadline_schedule, grouped_greedy_schedule
        cands=[edf_schedule(d_plan,T,K),latest_deadline_schedule(d_plan,T,K),
               grouped_greedy_schedule(d_plan,T,K,part)]
        cur=_repair(min(cands,key=cost),T,K)
    else:
        cur=_repair(random_schedule(d_plan,T,K,seed=seed),T,K)   # 튜닝: 동일 초기해 방식(무작위)
    cur_c=cost(cur); best=cur.copy(); best_c=cur_c
    Tk=-sig/np.log(p0); lvl=0; no_imp=0; acc=0; t0=time.time(); hist=[best_c]
    while Tk>T_END and lvl<MAX_LEVEL and no_imp<NO_IMP:
        improved=False
        for _ in range(L_LEVEL):
            nb=_repair(_neighbor(cur,T,K,rng),T,K)
            nc=cost(nb); dE=nc-cur_c
            if dE<=0 or rng.random()<np.exp(-dE/Tk):
                cur,cur_c=nb,nc; acc+=1
                if cur_c<best_c: best,best_c=cur.copy(),cur_c; improved=True
        hist.append(best_c)
        no_imp=0 if improved else no_imp+1
        Tk*=alpha; lvl+=1
    return dict(best=best, best_c=float(best_c), levels=lvl, accepted=acc,
                sec=round(time.time()-t0,2), hist=hist)

# ---- 완전요인 p x alpha, N=10 시드 (셀 단위 체크포인트) ----
resp=CK/"sa_tuning_cells.json"
cells=json.loads(resp.read_text(encoding="utf-8")) if resp.exists() else {}
for p0 in P_GRID:
    for al in A_GRID:
        key=f"{p0}-{al}"
        if key in cells: continue
        runs=[sa_course(p0,al,seed=100+r) for r in range(N_REP)]
        cs=[r["best_c"] for r in runs]
        cells[key]={"p":p0,"alpha":al,"mean":float(np.mean(cs)),"std":float(np.std(cs)),
                    "min":float(np.min(cs)),"max":float(np.max(cs)),
                    "levels":float(np.mean([r["levels"] for r in runs])),
                    "accepted":float(np.mean([r["accepted"] for r in runs])),
                    "sec":float(np.mean([r["sec"] for r in runs]))}
        resp.write_text(json.dumps(cells,ensure_ascii=False,indent=1),encoding="utf-8")
        log(f"  (p={p0}, a={al}): mean={cells[key]['mean']:.1f} std={cells[key]['std']:.1f} "
            f"levels={cells[key]['levels']:.0f} acc={cells[key]['accepted']:.0f} {cells[key]['sec']:.1f}s")

df=pd.DataFrame(cells.values())
pm=df.pivot(index="p",columns="alpha",values="mean"); ps=df.pivot(index="p",columns="alpha",values="std")
fig,ax=plt.subplots(1,2,figsize=(11,4))
sns.heatmap(pm,annot=True,fmt=".1f",cmap="RdYlGn_r",ax=ax[0],cbar_kws={"label":"목적값"})
ax[0].set_title(f"최종 목적값 평균 (N={N_REP} 시드)"); ax[0].set_xlabel("cooling rate α"); ax[0].set_ylabel("초기 수용확률 p")
sns.heatmap(ps,annot=True,fmt=".1f",cmap="Blues",ax=ax[1],cbar_kws={"label":"std"})
ax[1].set_title("최종 목적값 표준편차 (신뢰성)"); ax[1].set_xlabel("cooling rate α"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig(FIG/"O6_sa_tuning_heatmaps.png",dpi=110,bbox_inches="tight"); plt.close()
log("\n[추세] 열(alpha) 평균:", {c: round(pm[c].mean(),1) for c in pm.columns},
    "| 행(p) 평균:", {i: round(pm.loc[i].mean(),1) for i in pm.index})
win=df.loc[df["mean"].idxmin()]
log(f"[승자] p={win['p']}, alpha={win['alpha']} (mean {win['mean']:.1f} ± {win['std']:.1f})")

# ---- 승자 설정 최종 실행 (배포: 휴리스틱 초기해 결합) -> replay ----
runs=[sa_course(win["p"],win["alpha"],seed=500+r,init="heuristic") for r in range(3)]
bi=int(np.argmin([r["best_c"] for r in runs])); fin=runs[bi]
f=int(count_failures(fin["best"],d_true)); wl=float(wasted_life(fin["best"],d_true))
st=schedule_metrics(fin["best"],d_true,part,1.0,5.0,0.1,5.0)
log(f"[최종 replay] 튜닝 SA(p={win['p']},a={win['alpha']}): 고장={f} 낭비={wl:.0f} setup={st['setups']} 비용={st['total_cost']:.1f}")
log("  (비교: 기존 SA-강건 고장 48 / EDF 78)")
plt.figure(figsize=(6.6,3.8))
plt.plot(fin["hist"],color="#4C72B0")
plt.xlabel("temperature level"); plt.ylabel("best 목적값")
plt.title(f"튜닝 SA 수렴 (p={win['p']}, α={win['alpha']}, level당 {L_LEVEL}회)")
plt.tight_layout(); plt.savefig(FIG/"O7_tuned_sa_convergence.png",dpi=110,bbox_inches="tight"); plt.close()
json.dump({"sigma":sig,"L":L_LEVEL,"t_end":T_END,"cells":cells,
           "col_trend":{str(c): float(pm[c].mean()) for c in pm.columns},
           "row_trend":{str(i): float(pm.loc[i].mean()) for i in pm.index},
           "winner":{"p":float(win["p"]),"alpha":float(win["alpha"]),"mean":float(win["mean"]),"std":float(win["std"])},
           "final_replay":{"고장":f,"낭비":round(wl,0),"setup":int(st["setups"]),"총비용":round(float(st["total_cost"]),1)}},
          open(CK/"sa_tuning_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
np.save(CK/"sa_tuned_schedule.npy",fin["best"])
log("[완료] sa_tuning_summary.json + O6/O7")
