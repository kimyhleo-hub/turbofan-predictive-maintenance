"""루프 2회차 — v3 특징(다중 임계 onset) + LSTM-D(v2 채널) + 혼합. 체크포인트 재개형.

EDA 근거(2회차 mini-EDA): FD003 onset rel 0.45±0.17로 늦고 산포 큼 → 단일 θ=0.3 시계로는
경계 정보가 부족 → 다중 임계 onset 시계(θ=0.2/0.5) + 도달 플래그 + 경계 마진.
LSTM-D: 사다리 확장 — C(27ch) + v2 채널 21개 = 48ch. 시퀀스 모델에서도 v2 표현이 이득인지.
test 규율: 라운드 valid-best 1개 + (사전 등록) LSTM-D 사다리용 1개만 test.
"""
from __future__ import annotations
from pathlib import Path
import json, time, gc
import numpy as np, pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from numpy.lib.stride_tricks import sliding_window_view

def log(*a): print(*a, flush=True)
for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand/"data"/"processed").exists(): ROOT=cand; break
PROC=ROOT/"data"/"processed"; RAW=ROOT/"data"/"raw"; SEQ=PROC/"seq"
CK=ROOT/"artifacts"/"03_model"; CK.mkdir(parents=True, exist_ok=True)
SEED=42; CLIP=125; FD_LIST=["FD001","FD002","FD003","FD004"]
BEST=dict(num_leaves=63, learning_rate=0.05, min_child_samples=60,
          colsample_bytree=0.7, subsample=0.7, reg_lambda=1.0)
FLIP=["s7","s12","s15","s20","s21"]
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))
def phm08(y,p):
    d=np.asarray(p,float)-np.asarray(y,float)
    return float(np.sum(np.where(d<0,np.exp(-d/13)-1,np.exp(d/10)-1)))

# ---------- v2 + v3 특징 ----------
def add_v23(df, union):
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
    df["v2_cum"]=(df["v2_H"]-0.3).clip(lower=0).groupby(df["engine_id"]).cumsum()
    for th,tag in [(0.2,"02"),(0.3,"03"),(0.5,"05")]:          # v3: 다중 임계 시계
        on=(df["v2_H_ma20"]>th).groupby(df["engine_id"]).cummax()
        df[f"v3_onset{tag}"]=on.groupby(df["engine_id"]).cumsum().astype(float)
        if tag=="03": df["v2_onset_age"]=df[f"v3_onset{tag}"]
    df["v3_onset_flag"]=(df["v2_onset_age"]>0).astype(float)   # v3: 도달 플래그
    df["v3_H_margin"]=df["v2_H_ma20"]-0.3                       # v3: 경계 마진(부호)
    a=df["z_s9"]; b=df["z_s14"]; df["_ab"]=a*b; df["_a2"]=a*a; df["_b2"]=b*b
    def rm(c): return df.groupby("engine_id",sort=False)[c].rolling(20,min_periods=5).mean().reset_index(level=0,drop=True)
    ma,mb,mab,ma2,mb2=rm("z_s9"),rm("z_s14"),rm("_ab"),rm("_a2"),rm("_b2")
    cov=mab-ma*mb; va=(ma2-ma*ma).clip(lower=1e-12); vb=(mb2-mb*mb).clip(lower=1e-12)
    df["v2_coup914"]=(cov/np.sqrt(va*vb)).fillna(0.0).clip(-1,1)
    df.drop(columns=["_ab","_a2","_b2"],inplace=True)
    for s in FLIP: df[f"v2_abs_b_{s}"]=df[f"b_{s}"].abs()
    return df

log("[load] unified + v2/v3 파생 ...")
manifest=json.loads((PROC/"feature_manifest.json").read_text(encoding="utf-8"))
UNION=manifest["unified"]["union_sensors"]
uni=pd.read_csv(PROC/"train_features_unified.csv"); te=pd.read_csv(PROC/"test_features_unified.csv")
uni=add_v23(uni,UNION); te=add_v23(te,UNION)
META={"unit","cycle","dataset_id","condition_id","engine_id","RUL"}
BASE=[c for c in uni.columns if c not in META and not c.startswith(("b_","v2_","v3_"))]
V2=[f"b_{s}" for s in UNION]+["v2_H","v2_H_ma20","v2_cum","v2_onset_age","v2_coup914"]+[f"v2_abs_b_{s}" for s in FLIP]
V3=["v3_onset02","v3_onset05","v3_onset_flag","v3_H_margin"]
for d in (uni,te):
    d["dataset_id"]=d["dataset_id"].astype("category"); d["condition_id"]=d["condition_id"].astype("category")
CATS=["dataset_id","condition_id"]
tr_idx,va_idx=[],[]
for _,g in uni.groupby("dataset_id",observed=True):
    gss=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=SEED)
    i,j=next(gss.split(g,groups=g["engine_id"]))
    tr_idx+=list(g.index[i]); va_idx+=list(g.index[j])
tr_idx=sorted(tr_idx); va_idx=sorted(va_idx)
TR,VA=uni.loc[tr_idx],uni.loc[va_idx]
va_ds=VA["dataset_id"].astype(str).to_numpy(); y_va=VA["RUL"].to_numpy(float)
va_mask_rows=np.zeros(len(uni),bool); va_mask_rows[va_idx]=True

def fit(xcols,tag,n_est=4000,pat=200):
    m=lgb.LGBMRegressor(n_estimators=n_est,subsample_freq=1,random_state=SEED,verbose=-1,**BEST)
    m.fit(TR[xcols],TR["RUL"],eval_set=[(VA[xcols],VA["RUL"])],eval_metric="rmse",
          categorical_feature=CATS,callbacks=[lgb.early_stopping(pat,verbose=False),lgb.log_evaluation(0)])
    p=np.clip(m.predict(VA[xcols]),0,CLIP)
    r=rmse(y_va,p); pf={f: round(rmse(y_va[va_ds==f],p[va_ds==f]),2) for f in FD_LIST}
    log(f"  {tag:26s} valid={r:.3f}  {pf}  (iter {m.best_iteration_})")
    return m,p,r,pf

# ---------- S1: LGBM v2 vs v3 ----------
s1p=CK/"s1.json"
if not s1p.exists():
    log("[S1] LGBM — v2 기준 vs v3(다중 임계 onset)")
    _,p2,r2,pf2=fit(BASE+V2+CATS,"v2 (기준)")
    m3,p3,r3,pf3=fit(BASE+V2+V3+CATS,"v3 (+다중임계 4)")
    np.save(CK/"pva_lgbm.npy", p3 if r3<r2 else p2)
    json.dump({"v2":{"valid":r2,"perfd":pf2},"v3":{"valid":r3,"perfd":pf3},
               "pick":"v3" if r3<r2 else "v2",
               "iter": int(m3.best_iteration_) if r3<r2 else None},
              open(s1p,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
s1=json.load(open(s1p,encoding="utf-8")); log(f"[S1] 결과: {s1['pick']} 선택 (v2 {s1['v2']['valid']:.3f} vs v3 {s1['v3']['valid']:.3f})")
pva_lgbm=np.load(CK/"pva_lgbm.npy")

# ---------- S2: LSTM-D (C + v2 채널 21 = 48ch) ----------
def make_windows(feat,L=30):
    T,F=feat.shape
    p=np.vstack([np.zeros((L-1,F),feat.dtype),feat])
    return np.ascontiguousarray(sliding_window_view(p,(L,F))[:,0],np.float32)
GLOBAL=sorted(uni["condition_id"].astype(str).unique()); CI={c:i for i,c in enumerate(GLOBAL)}
DSI={f:i for i,f in enumerate(FD_LIST)}
D_EXTRA=[f"b_{s}" for s in UNION]+["v2_H","v2_cum","v2_onset_age","v2_coup914"]   # 21채널 (스케일: cum/onset 정규화)
def seq_D(df):
    n=len(df)
    Z=np.nan_to_num(df[[f"z_{s}" for s in UNION]].to_numpy(np.float32),nan=0.0)
    B=np.nan_to_num(df[[f"b_{s}" for s in UNION]].to_numpy(np.float32),nan=0.0)
    ex=np.stack([df["v2_H"].to_numpy(np.float32),
                 (df["v2_cum"]/50.0).to_numpy(np.float32),
                 (df["v2_onset_age"]/100.0).to_numpy(np.float32),
                 df["v2_coup914"].to_numpy(np.float32)],axis=1)
    oc=np.zeros((n,6),np.float32); oc[np.arange(n),df["condition_id"].astype(str).map(CI).to_numpy()]=1.0
    od=np.zeros((n,4),np.float32)
    od[np.arange(n),df["dataset_id"].astype(str).map(DSI).to_numpy()]=1.0
    return np.hstack([Z,B,ex,oc,od])           # 17+17+4+6+4 = 48

class RulLSTM(nn.Module):
    def __init__(self,F,hidden=100,layers=2,drop=0.2):
        super().__init__()
        self.lstm=nn.LSTM(F,hidden,num_layers=layers,batch_first=True,dropout=drop)
        self.head=nn.Sequential(nn.Linear(hidden,50),nn.ReLU(),nn.Dropout(drop),nn.Linear(50,1))
    def forward(self,x):
        h,_=self.lstm(x); return self.head(h[:,-1]).squeeze(-1)

s2p=CK/"s2_lstmD.npz"
if not s2p.exists():
    log("[S2] LSTM-D 텐서 생성 (48ch)")
    Xl=[]; Xt=[]
    for eid,g in uni.groupby("engine_id",sort=False): Xl.append(make_windows(seq_D(g)))
    # test 순서는 반드시 true_RUL 순서(FD 카테고리 순 × 숫자 unit 순)와 일치해야 함
    for (fd_,u_),g in te.sort_values(["dataset_id","unit","cycle"]).groupby(["dataset_id","unit"],observed=True):
        Xt.append(make_windows(seq_D(g))[-1])
    Xd=np.concatenate(Xl); Xl=None; Xdt=np.stack(Xt)
    y_all=uni["RUL"].to_numpy(np.float32)
    log(f"  X {Xd.shape} / test {Xdt.shape}")
    torch.manual_seed(SEED)
    Xtr=torch.from_numpy(Xd); Y=torch.from_numpy(y_all)
    tr_i=np.where(~va_mask_rows)[0]; va_i=np.where(va_mask_rows)[0]
    model=RulLSTM(Xd.shape[2]); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=2)
    best=(1e9,None,0)
    for ep in range(1,31):
        model.train(); perm=np.random.default_rng(SEED+ep).permutation(tr_i); t0=time.time()
        for k in range(0,len(perm),512):
            idx=perm[k:k+512]; opt.zero_grad()
            loss=nn.functional.mse_loss(model(Xtr[idx]),Y[idx])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); ps=[]
        with torch.no_grad():
            for k in range(0,len(va_i),4096): ps.append(model(Xtr[va_i[k:k+4096]]).numpy())
        pv=np.clip(np.concatenate(ps),0,CLIP); r=rmse(y_all[va_i],pv); sch.step(r)
        mark=""
        if r<best[0]: best=(r,{k2:v.detach().clone() for k2,v in model.state_dict().items()},ep); mark=" *"
        log(f"  [D] ep{ep:02d} valid={r:.3f} ({time.time()-t0:.0f}s){mark}")
        if ep-best[2]>=5: log(f"  [D] early stop (best ep{best[2]} {best[0]:.3f})"); break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        ps=[]
        for k in range(0,len(va_i),4096): ps.append(model(Xtr[va_i[k:k+4096]]).numpy())
        pva=np.clip(np.concatenate(ps),0,CLIP)
        pte=np.clip(model(torch.from_numpy(Xdt)).numpy(),0,CLIP)
    np.savez_compressed(s2p, valid=best[0], best_ep=best[2], pva=pva, pte=pte)
    del Xd,Xtr; gc.collect()
s2=np.load(s2p)
pf_D={f: round(rmse(y_va[va_ds==f], s2["pva"][va_ds==f]),2) for f in FD_LIST}
log(f"[S2] LSTM-D valid={float(s2['valid']):.3f} FD별={pf_D} (C 대비 {float(s2['valid'])-13.711:+.2f})")

# ---------- S3: 혼합 (라운드 valid-best 산정) ----------
alphas=np.arange(0,1.001,0.05)
mix=[rmse(y_va, a*pva_lgbm+(1-a)*s2["pva"]) for a in alphas]
alpha=float(alphas[int(np.argmin(mix))])
cands={"LGBM(라운드)":rmse(y_va,pva_lgbm),"LSTM-D":float(s2["valid"]),"혼합":float(min(mix))}
log(f"[S3] valid 후보: {cands} | alpha={alpha:.2f}")
best_name=min(cands,key=cands.get)

# ---------- S4: test — valid-best 1개 + (사전등록) LSTM-D 사다리 ----------
last=te.sort_values(["dataset_id","unit","cycle"]).groupby(["dataset_id","unit"],observed=True).tail(1)
true=np.concatenate([pd.read_csv(RAW/f"RUL_{f}.txt",sep=r"\s+",header=None)[0].to_numpy() for f in FD_LIST]).astype(float)
ds=last["dataset_id"].astype(str).to_numpy(); yc=np.minimum(true,CLIP)
def trow(p):
    return {"RMSE":round(rmse(yc,p),2),"PHM":round(phm08(yc,p),0),
            **{f: round(rmse(np.minimum(true[ds==f],CLIP),p[ds==f]),2) for f in FD_LIST}}
out={"s1":s1,"lstmD":{"valid":float(s2["valid"]),"perfd":pf_D,"best_ep":int(s2["best_ep"])},
     "mix_alpha":alpha,"valid_cands":cands,"valid_best":best_name}
# LGBM 라운드 구성 full-train 재학습
xcols=(BASE+V2+V3+CATS) if s1["pick"]=="v3" else (BASE+V2+CATS)
it=s1.get("iter") or 787
fm=lgb.LGBMRegressor(n_estimators=int(it),subsample_freq=1,random_state=SEED,verbose=-1,**BEST)
fm.fit(uni[xcols],uni["RUL"],categorical_feature=CATS)
pte_lgbm=np.clip(fm.predict(last[xcols]),0,CLIP)
pte_mix=alpha*pte_lgbm+(1-alpha)*s2["pte"]
pick_pred={"LGBM(라운드)":pte_lgbm,"LSTM-D":s2["pte"],"혼합":pte_mix}[best_name]
out["test_valid_best"]={"name":best_name,**trow(pick_pred)}
out["test_lstmD_ladder"]=trow(s2["pte"])
log(f"[S4] test(valid-best={best_name}): {out['test_valid_best']}")
log(f"[S4] test(LSTM-D 사다리): {out['test_lstmD_ladder']}")
if best_name!="LGBM(라운드)":
    pd.DataFrame({"dataset_id":ds,"unit":last["unit"].to_numpy(),
                  "pred_RUL":np.round(pick_pred,2),"true_RUL":true}).to_csv(PROC/"fleet_rul_predictions_unified.csv",index=False)
    log("  fleet CSV -> 라운드 best로 갱신")
json.dump(out,open(CK/"loop2_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
log("[완료] loop2_summary.json")
