"""03b 실험 드라이버 — LGBM 튜닝 + LSTM A/B/C + 최종 test. 체크포인트 재개형.

중단되어도 재실행하면 완료된 단계(튜닝은 trial 단위)는 건너뛴다.
체크포인트: artifacts/03_model/
"""
from __future__ import annotations
from pathlib import Path
import json, time, gc, sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import GroupShuffleSplit

def log(*a): print(*a, flush=True)

for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand / "data" / "processed").exists():
        ROOT = cand; break
PROC = ROOT/"data"/"processed"; SEQ = PROC/"seq"; RAW = ROOT/"data"/"raw"
CKPT = ROOT/"artifacts"/"03_model"; CKPT.mkdir(parents=True, exist_ok=True)
FIG = ROOT/"reports"/"figures"/"rul_model"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT/"models"; MODELS.mkdir(exist_ok=True)

SEED=42; CLIP=125; L=30
FD_LIST=["FD001","FD002","FD003","FD004"]

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def phm08(y,p):
    d=np.asarray(p,float)-np.asarray(y,float)
    return float(np.sum(np.where(d<0,np.exp(-d/13)-1,np.exp(d/10)-1)))

# ---------------- 공통 데이터 ----------------
log("[load] unified CSV ...")
uni=pd.read_csv(PROC/"train_features_unified.csv")
uni_te=pd.read_csv(PROC/"test_features_unified.csv")
manifest=json.loads((PROC/"feature_manifest.json").read_text(encoding="utf-8"))
META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
FEATS=[c for c in uni.columns if c not in META]
for d_ in (uni,uni_te):
    d_["dataset_id"]=d_["dataset_id"].astype("category"); d_["condition_id"]=d_["condition_id"].astype("category")
XCOLS=FEATS+["dataset_id","condition_id"]; CATS=["dataset_id","condition_id"]

tr_idx,va_idx=[],[]
for _,g in uni.groupby("dataset_id",observed=True):
    gss=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=SEED)
    i,j=next(gss.split(g,groups=g["engine_id"]))
    tr_idx+=list(g.index[i]); va_idx+=list(g.index[j])
tr_idx=sorted(tr_idx); va_idx=sorted(va_idx)
TR,VA=uni.loc[tr_idx],uni.loc[va_idx]
va_ds=VA["dataset_id"].astype(str).to_numpy()
y_va=VA["RUL"].to_numpy(float)
va_mask_rows=np.zeros(len(uni),bool); va_mask_rows[va_idx]=True
log(f"[data] train {len(TR):,} / valid {len(VA):,} rows | valid 엔진 {VA.engine_id.nunique()}")

def fit_eval(cfg, seed=SEED):
    m=lgb.LGBMRegressor(n_estimators=4000, subsample_freq=1, random_state=seed, verbose=-1, **cfg)
    m.fit(TR[XCOLS],TR["RUL"],eval_set=[(VA[XCOLS],VA["RUL"])],eval_metric="rmse",
          categorical_feature=CATS,
          callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(0)])
    p=np.clip(m.predict(VA[XCOLS]),0,CLIP)
    return m,p,rmse(y_va,p)

# ---------------- Stage 1: LGBM 튜닝 (trial 단위 재개) ----------------
tune_path=CKPT/"tuning_trials.json"
tune=json.loads(tune_path.read_text(encoding="utf-8")) if tune_path.exists() else {"results":[]}
rng=np.random.default_rng(SEED)
SPACE=dict(num_leaves=[31,63,127,255], learning_rate=[0.03,0.05,0.08],
           min_child_samples=[10,30,60], colsample_bytree=[0.7,0.9],
           subsample=[0.7,0.9], reg_lambda=[0.0,1.0,10.0])
trials=[]; seen=set()
while len(trials)<24:                       # 시드 고정 → 재실행에도 동일 trial 목록
    cfg={k: v[rng.integers(len(v))] for k,v in SPACE.items()}
    cfg={k:(int(v) if k in ("num_leaves","min_child_samples") else float(v)) for k,v in cfg.items()}
    key=tuple(cfg.values())
    if key in seen: continue
    seen.add(key); trials.append(cfg)
done=len(tune["results"])
if done<24:
    log(f"[S1] LGBM 튜닝 — {done}/24 완료 상태에서 재개")
    for k in range(done,24):
        t0=time.time(); _,_,r=fit_eval(trials[k])
        tune["results"].append({**trials[k],"valid_RMSE":round(r,3)})
        tune_path.write_text(json.dumps(tune,ensure_ascii=False,indent=1),encoding="utf-8")
        log(f"  [S1 {k+1:2d}/24] RMSE={r:.3f} ({time.time()-t0:.0f}s) {trials[k]}")
tt=pd.DataFrame(tune["results"]).sort_values("valid_RMSE").reset_index(drop=True)
BEST={k:tt.iloc[0][k] for k in SPACE}
BEST={k:(int(v) if k in ("num_leaves","min_child_samples") else float(v)) for k,v in BEST.items()}
log(f"[S1] BEST valid={tt.iloc[0]['valid_RMSE']}: {BEST}")

# ---------------- Stage 2: 5-seed 앙상블 ----------------
seeds_path=CKPT/"seeds.npz"
if not seeds_path.exists():
    log("[S2] 5-seed 앙상블 학습")
    preds=[]; iters=[]; rs=[]
    for s in range(SEED,SEED+5):
        t0=time.time(); m,p,r=fit_eval(BEST,seed=s)
        preds.append(p); iters.append(int(m.best_iteration_)); rs.append(r)
        log(f"  [S2] seed {s}: valid={r:.3f} iter={m.best_iteration_} ({time.time()-t0:.0f}s)")
    np.savez_compressed(seeds_path, preds=np.array(preds), iters=np.array(iters), rmses=np.array(rs))
sz=np.load(seeds_path)
va_pred_lgbm=sz["preds"].mean(0); best_iters=sz["iters"].tolist()
lgbm_ens_valid=rmse(y_va,va_pred_lgbm)
perfd_ens={fd: round(rmse(y_va[va_ds==fd], va_pred_lgbm[va_ds==fd]),2) for fd in FD_LIST}
log(f"[S2] 앙상블 valid={lgbm_ens_valid:.3f} FD별={perfd_ens}")
assert perfd_ens["FD001"]>8, "[가드] 누수 의심"

# ---------------- LSTM 공통 ----------------
COLS=["unit","cycle","set1","set2","set3"]+[f"s{i}" for i in range(1,22)]
UNION=manifest["unified"]["union_sensors"]
GLOBAL=sorted(uni["condition_id"].astype(str).unique()); CI={c:i for i,c in enumerate(GLOBAL)}
DSI={fd:i for i,fd in enumerate(FD_LIST)}

def make_windows(feat,Lw):
    T,F=feat.shape
    p=np.vstack([np.zeros((Lw-1,F),feat.dtype),feat])
    return np.ascontiguousarray(sliding_window_view(p,(Lw,F))[:,0],np.float32)

def load_raw(fd,split):
    t=pd.read_csv(RAW/f"{split}_{fd}.txt",sep=r"\s+",header=None,names=COLS).sort_values(["unit","cycle"])
    r=t[["set1","set2","set3"]].round({"set1":0,"set2":2,"set3":0})+0.0
    t["cond"]=(r.set1.astype(int).astype(str)+"|"+r.set2.map(lambda v:f"{v:.2f}")+"|"+r.set3.astype(int).astype(str))
    return t

class RulLSTM(nn.Module):
    def __init__(self,F,hidden=100,layers=2,drop=0.2):
        super().__init__()
        self.lstm=nn.LSTM(F,hidden,num_layers=layers,batch_first=True,dropout=drop)
        self.head=nn.Sequential(nn.Linear(hidden,50),nn.ReLU(),nn.Dropout(drop),nn.Linear(50,1))
    def forward(self,x):
        h,_=self.lstm(x)
        return self.head(h[:,-1]).squeeze(-1)

def train_lstm(Xtr_np, Xte_np, y_all, tag, epochs=30, patience=5, bs=512, lr=1e-3):
    torch.manual_seed(SEED)
    Xtr=torch.from_numpy(Xtr_np); Y=torch.from_numpy(y_all)
    tr_i=np.where(~va_mask_rows)[0]; va_i=np.where(va_mask_rows)[0]
    model=RulLSTM(Xtr_np.shape[2])
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=2)
    best=(1e9,None,0)
    for ep in range(1,epochs+1):
        model.train(); perm=np.random.default_rng(SEED+ep).permutation(tr_i); t0=time.time()
        for k in range(0,len(perm),bs):
            idx=perm[k:k+bs]
            opt.zero_grad()
            loss=nn.functional.mse_loss(model(Xtr[idx]),Y[idx])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); ps=[]
        with torch.no_grad():
            for k in range(0,len(va_i),4096):
                ps.append(model(Xtr[va_i[k:k+4096]]).numpy())
        pv=np.clip(np.concatenate(ps),0,CLIP)
        r=rmse(y_all[va_i],pv); sch.step(r)
        mark=""
        if r<best[0]:
            best=(r,{k2:v.detach().clone() for k2,v in model.state_dict().items()},ep); mark=" *"
        log(f"  [{tag}] ep{ep:02d} valid={r:.3f} ({time.time()-t0:.0f}s){mark}")
        if ep-best[2]>=patience:
            log(f"  [{tag}] early stop (best ep{best[2]} {best[0]:.3f})"); break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        ps=[]
        for k in range(0,len(va_i),4096):
            ps.append(model(Xtr[va_i[k:k+4096]]).numpy())
        pva=np.clip(np.concatenate(ps),0,CLIP)
        pte=np.clip(model(torch.from_numpy(Xte_np)).numpy(),0,CLIP)
    return best[0],best[2],pva,pte

def build_variant(variant, trains, tests, stats):
    def frames(t,fd):
        n=len(t); Xr=np.zeros((n,len(UNION)),np.float32)
        for j,s in enumerate(UNION):
            if trains[fd][s].nunique()>1:
                mu,sd=stats[s]; Xr[:,j]=((t[s].to_numpy(float)-mu)/sd).astype(np.float32)
        oc=np.zeros((n,6),np.float32); oc[np.arange(n),t["cond"].map(CI).to_numpy()]=1.0
        od=np.zeros((n,4),np.float32); od[:,DSI[fd]]=1.0
        return np.hstack([Xr,od]) if variant=="A" else np.hstack([Xr,oc,od])
    Xtr=[]; Xte=[]
    for fd in FD_LIST:
        for _,g in trains[fd].groupby("unit"): Xtr.append(make_windows(frames(g,fd),L))
        for _,g in tests[fd].groupby("unit"):  Xte.append(make_windows(frames(g,fd),L)[-1])
    return np.concatenate(Xtr), np.stack(Xte)

# ---------------- Stage 3~5: LSTM A/B/C (변형 단위 재개) ----------------
zc=np.load(SEQ/"unified_train.npz"); zt=np.load(SEQ/"unified_test.npz")
y_all=zc["y"].astype(np.float32); true_te=zt["true_RUL"].astype(float)
need=[v for v in "ABC" if not (CKPT/f"lstm_{v}.npz").exists()]
if need:
    trains={fd: load_raw(fd,"train") for fd in FD_LIST}
    tests ={fd: load_raw(fd,"test")  for fd in FD_LIST}
    stats={}
    for s in UNION:
        vals=np.concatenate([trains[fd][s].to_numpy(float) for fd in FD_LIST if trains[fd][s].nunique()>1])
        stats[s]=(vals.mean(), vals.std() or 1.0)
for v in "ABC":
    p=CKPT/f"lstm_{v}.npz"
    if p.exists():
        log(f"[S3~5] LSTM-{v} 체크포인트 존재 — 건너뜀"); continue
    log(f"[S3~5] LSTM-{v} 시작")
    if v=="C":
        Xtr_np,Xte_np=zc["X"],zt["X"]
    else:
        t0=time.time(); Xtr_np,Xte_np=build_variant(v,trains,tests,stats)
        log(f"  [{v}] 텐서 {Xtr_np.shape} 생성 ({time.time()-t0:.0f}s)")
        assert Xtr_np.shape[0]==len(y_all) and Xte_np.shape[0]==len(true_te)
    val,ep,pva,pte=train_lstm(Xtr_np,Xte_np,y_all,v)
    perfd={fd: round(rmse(y_va[va_ds==fd], pva[va_ds==fd]),2) for fd in FD_LIST}
    np.savez_compressed(p, valid=val, best_ep=ep, pva=pva, pte=pte,
                        perfd=json.dumps(perfd))
    log(f"[S3~5] LSTM-{v} 완료 valid={val:.3f} FD별={perfd}")
    del Xtr_np,Xte_np; gc.collect()

# ---------------- Stage 6: 최종 test + fleet + digest ----------------
log("[S6] 최종 — LGBM full-train 재학습 + test 표 + 혼합")
last=uni_te.sort_values(["dataset_id","unit","cycle"]).groupby(["dataset_id","unit"],observed=True).tail(1).reset_index(drop=True)
true=np.concatenate([pd.read_csv(RAW/f"RUL_{fd}.txt",sep=r"\s+",header=None)[0].to_numpy() for fd in FD_LIST]).astype(float)
assert len(last)==707 and np.array_equal(true,true_te)
ds=last["dataset_id"].astype(str).to_numpy()

final_path=CKPT/"lgbm_test.npz"
if not final_path.exists():
    pte_seeds=[]
    for it,s in zip(best_iters,range(SEED,SEED+5)):
        t0=time.time()
        fm=lgb.LGBMRegressor(n_estimators=int(it),subsample_freq=1,random_state=s,verbose=-1,**BEST)
        fm.fit(uni[XCOLS],uni["RUL"],categorical_feature=CATS)
        pte_seeds.append(np.clip(fm.predict(last[XCOLS]),0,CLIP))
        log(f"  [S6] full-train seed {s} ({time.time()-t0:.0f}s)")
    fm.booster_.save_model(str(MODELS/"lgbm_unified_tuned.txt"))
    np.savez_compressed(final_path, pte=np.mean(pte_seeds,axis=0))
pte_lgbm=np.load(final_path)["pte"]

lstm={v: np.load(CKPT/f"lstm_{v}.npz") for v in "ABC"}
alphas=np.arange(0,1.001,0.05)
mix=[rmse(y_va, a*va_pred_lgbm+(1-a)*lstm["C"]["pva"]) for a in alphas]
alpha=float(alphas[int(np.argmin(mix))])
pte_mix=alpha*pte_lgbm+(1-alpha)*lstm["C"]["pte"]
log(f"[S6] 혼합 alpha={alpha:.2f} (valid RMSE {min(mix):.3f} | LGBM 단독 {lgbm_ens_valid:.3f} | LSTM-C {float(lstm['C']['valid']):.3f})")

def row(name,p):
    yc=np.minimum(true,CLIP)
    r={"모델":name,"RMSE_cap":round(rmse(yc,p),2),"PHM08_cap":round(phm08(yc,p),0)}
    for fd in FD_LIST: r[fd]=round(rmse(np.minimum(true[ds==fd],CLIP),p[ds==fd]),2)
    return r
rows=[{"모델":"LGBM 기본(03)","RMSE_cap":15.02,"PHM08_cap":3102,"FD001":13.74,"FD002":14.55,"FD003":14.21,"FD004":16.26},
      row("LGBM 튜닝+5seed",pte_lgbm),
      row("LSTM-A raw(전처리X)",lstm["A"]["pte"]),
      row("LSTM-B raw+cond",lstm["B"]["pte"]),
      row("LSTM-C 우리 전처리",lstm["C"]["pte"]),
      row(f"혼합 a={alpha:.2f}",pte_mix)]
test_tbl=pd.DataFrame(rows)

fleet_pred = pte_mix if min(mix)<=lgbm_ens_valid else pte_lgbm
pd.DataFrame({"dataset_id":ds,"unit":last["unit"].to_numpy(),
              "pred_RUL":np.round(fleet_pred,2),"true_RUL":true}).to_csv(
    PROC/"fleet_rul_predictions_unified.csv",index=False)

summary={"BEST":BEST,"tune_top5":tt.head(5).to_dict("records"),
         "lgbm_valid_single":float(sz["rmses"][0]),"lgbm_valid_ens":round(lgbm_ens_valid,3),
         "lgbm_perfd_ens":perfd_ens,
         "lstm_valid":{v: float(lstm[v]["valid"]) for v in "ABC"},
         "lstm_perfd":{v: json.loads(str(lstm[v]["perfd"])) for v in "ABC"},
         "lstm_best_ep":{v: int(lstm[v]["best_ep"]) for v in "ABC"},
         "alpha":alpha,"mix_valid":round(float(min(mix)),3),
         "test_table":rows}
(CKPT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding="utf-8")
log("\n=== DIGEST 03b ===")
log(json.dumps(summary,ensure_ascii=False,indent=1))
log("[완료] summary.json / fleet CSV 갱신")
