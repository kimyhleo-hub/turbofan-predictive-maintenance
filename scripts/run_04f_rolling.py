"""04f — 롤링 재계획(rolling horizon) vs 정적 계획.

테스트베드: valid 엔진(분리-train 미학습, run-to-failure 전체 궤적 보유).
특징이 전부 인과적(past-only; 스파이크 테스트 검증)이므로 "절단 예측 = 전체 궤적의 해당 행".
시뮬레이션: 초기 나이 a0 = U(0.45,0.9)*수명(seed42). 창 w=1..10, 창=13cyc.
  롤링: 매 창 시작 시 생존·미정비 엔진을 현재 나이 행으로 재예측 -> EDF 재계획 -> 이번 창 배정분 정비.
  정적: w=1 예측으로 한 번 계획, 무수정 실행.
정답 규칙(본 replay와 동일): 죽음 창 D=ceil(잔여0/13), 정비 창 <= D면 on-time.
비교: {정적,롤링} x {점예측, P10} 4팔 + Oracle(실제 잔여) 참조.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"src"/"pipeline.py").exists(): ROOT=cand; break
sys.path.insert(0,str(ROOT))
def log(*a): print(*a, flush=True)
PROC=ROOT/"data"/"processed"; CK=ROOT/"artifacts"/"04_scheduling"
FIG=ROOT/"reports"/"figures"/"scheduling"
src=Path("scripts/run_loop2.py").read_text(encoding="utf-8")
ns={}; exec(src.split('log("[load]')[0], ns)
add_v23=ns["add_v23"]; BEST=ns["BEST"]; FLIP=ns["FLIP"]
CPW=13; T=10; SEED=42; FD=["FD001","FD002","FD003","FD004"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", font_scale=0.9)
try: plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass

# ---- 데이터·모델 (분리-train만 학습) ----
manifest=json.loads((PROC/"feature_manifest.json").read_text(encoding="utf-8"))
UNION=manifest["unified"]["union_sensors"]
uni=pd.read_csv(PROC/"train_features_unified.csv"); uni=add_v23(uni,UNION)
META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
BASE=[c for c in uni.columns if c not in META and not c.startswith(("b_","v2_","v3_"))]
V2=[f"b_{s}" for s in UNION]+["v2_H","v2_H_ma20","v2_cum","v2_onset_age","v2_coup914"]+[f"v2_abs_b_{s}" for s in FLIP]
V3=["v3_onset02","v3_onset05","v3_onset_flag","v3_H_margin"]
uni["dataset_id"]=uni["dataset_id"].astype("category"); uni["condition_id"]=uni["condition_id"].astype("category")
CATS=["dataset_id","condition_id"]; X=BASE+V2+V3+CATS
ti,vi=[],[]
for _,g in uni.groupby("dataset_id",observed=True):
    gss=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=SEED)
    a,b=next(gss.split(g,groups=g["engine_id"])); ti+=list(g.index[a]); vi+=list(g.index[b])
ti=sorted(ti); vi=sorted(vi)
mp_path=CK/"roll_models_done.flag"
import joblib
if not mp_path.exists():
    log("[모델] 분리-train 재학습 (v3 점예측 iter781 / P10 iter419)")
    m_pt=lgb.LGBMRegressor(n_estimators=781,subsample_freq=1,random_state=SEED,verbose=-1,**BEST)
    m_pt.fit(uni.loc[ti,X],uni.loc[ti,"RUL"],categorical_feature=CATS)
    m_q=lgb.LGBMRegressor(objective="quantile",alpha=0.1,n_estimators=419,subsample_freq=1,
                          random_state=SEED,verbose=-1,**BEST)
    m_q.fit(uni.loc[ti,X],uni.loc[ti,"RUL"],categorical_feature=CATS)
    joblib.dump(m_pt,CK/"roll_m_pt.pkl"); joblib.dump(m_q,CK/"roll_m_q.pkl"); mp_path.touch()
m_pt=joblib.load(CK/"roll_m_pt.pkl"); m_q=joblib.load(CK/"roll_m_q.pkl")

# ---- 시뮬레이션 준비 ----
VA=uni.loc[vi]
eng=sorted(VA["engine_id"].unique())
life={e:int(g.cycle.max()) for e,g in VA.groupby("engine_id")}
rng=np.random.default_rng(SEED)
a0={e:max(15,int(rng.uniform(0.45,0.90)*life[e])) for e in eng}
D={e:int(np.ceil((life[e]-a0[e])/CPW)) for e in eng}          # 죽음 창 (이 창까지 정비=on-time)
n=len(eng); K=int(np.ceil(n*1.06/T))
VA_idx=VA.set_index(["engine_id","cycle"])
log(f"[설정] valid 엔진 {n}대 | K={K}/창 (슬롯 {T*K}) | 죽음창<=10: {sum(1 for e in eng if D[e]<=10)}대")

def predict_rows(pairs, model):
    rows=VA_idx.loc[pairs]
    return np.clip(model.predict(rows[X]),0,125)

def edf_plan(active, dl_abs, w, used):
    """EDF: 마감 순으로 창 w..T의 잔여 용량에 배정. 반환 {engine: window}."""
    plan={}; cap={t:K-used.get(t,0) for t in range(w,T+1)}
    for e in sorted(active,key=lambda e:dl_abs[e]):
        d=min(max(dl_abs[e],w),T)
        for t in range(w,d+1):
            if cap[t]>0: plan[e]=t; cap[t]-=1; break
    return plan

def simulate(mode, kind):
    """mode: static|rolling / kind: point|p10|oracle -> 지표."""
    model=m_pt if kind=="point" else m_q
    maintained={}; used={}
    active=set(eng)
    if mode=="static":
        pairs=[(e,a0[e]) for e in eng]
        rul=(np.array([life[e]-a0[e] for e in eng],float) if kind=="oracle"
             else predict_rows(pairs,model))
        dl={e:int(np.ceil(max(r,1)/CPW)) for e,r in zip(eng,rul)}
        plan=edf_plan(eng,dl,1,{})
        for e,t in plan.items():
            if t<=D[e]: maintained[e]=t          # 죽기 전 정비 성공
        fail=[e for e in eng if D[e]<=T and e not in maintained]
        waste=sum(max(0,life[e]-(a0[e]+CPW*(maintained[e]-1))) for e in maintained)
        return dict(fail=len(fail),maint=len(maintained),
                    carried=n-len(maintained)-len(fail),waste=int(waste))
    # rolling
    for w in range(1,T+1):
        dead={e for e in active if D[e]<w}       # 이번 창 전에 사망
        active-=dead
        if not active: break
        pairs=[(e,a0[e]+CPW*(w-1)) for e in active]
        if kind=="oracle":
            rul=np.array([life[e]-(a0[e]+CPW*(w-1)) for e in active],float)
        else:
            rul=predict_rows(pairs,model)
        dl={e:w-1+int(np.ceil(max(r,1)/CPW)) for e,r in zip(active,rul)}
        plan=edf_plan(active,dl,w,used)
        now=[e for e,t in plan.items() if t==w]
        for e in now:
            maintained[e]=w; used[w]=used.get(w,0)+1
        active-=set(now)
    fail=[e for e in eng if D[e]<=T and e not in maintained]
    waste=sum(max(0,life[e]-(a0[e]+CPW*(maintained[e]-1))) for e in maintained)
    return dict(fail=len(fail),maint=len(maintained),carried=len(active),waste=int(waste))

log("[시뮬레이션] 정적 vs 롤링 x 점예측/P10 (+Oracle 참조)")
res={}
for mode in ("static","rolling"):
    for kind in ("point","p10","oracle"):
        r=simulate(mode,kind); res[f"{mode}-{kind}"]=r
        log(f"  {mode:8s} x {kind:6s}: 고장={r['fail']:3d} 정비={r['maint']} 이월={r['carried']} 낭비={r['waste']}cyc")
json.dump({"n":n,"K":K,"res":res},open(CK/"rolling_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

labels=["점예측","P10","Oracle"]
st=[res[f"static-{k}"]["fail"] for k in ("point","p10","oracle")]
ro=[res[f"rolling-{k}"]["fail"] for k in ("point","p10","oracle")]
x=np.arange(3); wd=0.35
plt.figure(figsize=(7,4.2))
b1=plt.bar(x-wd/2,st,wd,label="정적 (1회 계획)",color="#8E9A9A")
b2=plt.bar(x+wd/2,ro,wd,label="롤링 (매 창 재예측·재계획)",color="#4C72B0")
for b in list(b1)+list(b2): plt.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,int(b.get_height()),ha="center",fontsize=9)
plt.xticks(x,labels); plt.ylabel(f"미정비 고장 수 (valid {n}대, 10창)")
plt.title("롤링 재계획의 가치 — 재예측이 고장을 얼마나 줄이는가")
plt.legend(); plt.tight_layout()
plt.savefig(FIG/"O9_rolling_vs_static.png",dpi=110,bbox_inches="tight"); plt.close()
log("[완료] rolling_summary.json + O9")
