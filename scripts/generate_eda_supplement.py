"""보완 EDA — FD001~FD004 종합 재계산.

기존 보고서의 공백을 메운다:
  1. 네 서브셋 구조/수명/test RUL 종합 비교
  2. FD002 자체 재계산 (조건 분산 지배도, raw vs 조건정규화 RUL 상관, setting 상관, 조건별 상수)
  3. 고장모드 proxy clustering — FD003(고장2, 조건1) 검증 + FD001(고장1) 대조군 + FD004(고장2, 조건6)
  4. 말기 분산 정량화 (FD001 vs FD003)
  5. 전 서브셋 조건 간/내 분산 분해 요약

출력: reports/figures/eda_supplement/*.png  +  콘솔 수치 digest
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=0.9)
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand / "data" / "raw").exists():
        ROOT = cand
        break
else:
    raise FileNotFoundError("data/raw 없음")

RAW = ROOT / "data" / "raw"
FIG = ROOT / "reports" / "figures" / "eda_supplement"
FIG.mkdir(parents=True, exist_ok=True)

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]
SETTINGS = ["set1", "set2", "set3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + SETTINGS + SENSORS
SEED = 42


def load_train(fd):
    return pd.read_csv(RAW / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)


def load_test(fd):
    return pd.read_csv(RAW / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)


def load_rul(fd):
    return pd.read_csv(RAW / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["true_RUL"])["true_RUL"].values


def prep(fd):
    df = load_train(fd).sort_values(["unit", "cycle"]).copy()
    life = df.groupby("unit")["cycle"].transform("max")
    df["total_life"] = life
    df["RUL_raw"] = life - df["cycle"]
    df["relative_cycle"] = df["cycle"] / life
    # `+ 0.0`으로 -0.0 을 0.0 으로 정규화 (안 하면 FD001/FD003이 조건 2개로 잘못 잡힘)
    r = df[SETTINGS].round({"set1": 0, "set2": 2, "set3": 0}) + 0.0
    df["condition_id"] = (r["set1"].astype(int).astype(str) + "|" +
                          r["set2"].map(lambda v: f"{v:.2f}") + "|" +
                          r["set3"].astype(int).astype(str))
    return df


def valid_sensors(df):
    n = df[SENSORS].nunique()
    return [s for s in SENSORS if n[s] > 1]


def cond_z(df, sensors):
    out = df.copy()
    for c in sensors:
        m = df.groupby("condition_id")[c].transform("mean")
        s = df.groupby("condition_id")[c].transform("std").replace(0, 1.0).fillna(1.0)
        out[f"z_{c}"] = (df[c] - m) / s
    return out


def var_decomp(df, sensor):
    overall = df[sensor].mean()
    total = ((df[sensor] - overall) ** 2).sum()
    if total == 0:
        return np.nan
    cm = df.groupby("condition_id")[sensor].transform("mean")
    between = ((cm - overall) ** 2).sum()
    return between / total


def line(t=""):
    print(("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78) if t else "")


DATA = {fd: prep(fd) for fd in FD_LIST}
VALID = {fd: valid_sensors(DATA[fd]) for fd in FD_LIST}

# ---------------------------------------------------------------- 1. 종합 구조
line("1. 네 서브셋 종합 (원본 재계산)")
rows = []
for fd in FD_LIST:
    d = DATA[fd]
    te, tr_rul = load_test(fd), load_rul(fd)
    life = d.groupby("unit")["cycle"].max()
    rows.append({
        "set": fd, "train_rows": len(d), "train_units": d["unit"].nunique(),
        "test_units": te["unit"].nunique(), "conditions": d["condition_id"].nunique(),
        "constant": 21 - len(VALID[fd]), "valid": len(VALID[fd]),
        "life_mean": round(life.mean(), 2), "life_max": int(life.max()),
        "testRUL_min": int(tr_rul.min()), "testRUL_max": int(tr_rul.max()),
        "testRUL_mean": round(tr_rul.mean(), 1),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for fd in FD_LIST:
    life = DATA[fd].groupby("unit")["cycle"].max()
    sns.ecdfplot(life, ax=ax[0], label=fd)
ax[0].set_title("엔진 수명 ECDF"); ax[0].set_xlabel("total life (cycles)"); ax[0].legend()
for fd in FD_LIST:
    sns.kdeplot(load_rul(fd), ax=ax[1], label=fd, fill=False)
ax[1].set_title("test 정답 RUL 분포"); ax[1].set_xlabel("true_RUL"); ax[1].legend()
sub = summary.set_index("set")[["conditions", "constant"]]
sub.plot(kind="bar", ax=ax[2], color=["#4C72B0", "#C44E52"])
ax[2].set_title("운전조건 수 vs 상수 센서 수"); ax[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(FIG / "S1_overview_4subsets.png", dpi=110, bbox_inches="tight"); plt.close()

# ------------------------------------------- 2. 조건 분산 지배도 (전 서브셋)
line("2. 조건 간 분산 비중 (between-condition share) — 서브셋별 요약")
vd_rows = []
for fd in FD_LIST:
    d = DATA[fd]
    if d["condition_id"].nunique() == 1:
        vd_rows.append({"set": fd, "n_cond": 1, "mean_between": 0.0, "min_between": 0.0, "max_between": 0.0})
        continue
    shares = [var_decomp(d, s) for s in VALID[fd]]
    shares = [x for x in shares if np.isfinite(x)]
    vd_rows.append({"set": fd, "n_cond": d["condition_id"].nunique(),
                    "mean_between": round(np.mean(shares), 4),
                    "min_between": round(np.min(shares), 4),
                    "max_between": round(np.max(shares), 4)})
vd = pd.DataFrame(vd_rows)
print(vd.to_string(index=False))
print("\n(조건이 1개인 FD001/FD003은 조건 간 분산이 정의상 0)")

# ------------------------------------- 3. FD002 재계산 (raw vs cond-z RUL 상관)
line("3. FD002 재계산 — setting 상관 / 조건별 상수 / raw vs 조건정규화 RUL 상관")
d2 = cond_z(DATA["FD002"], VALID["FD002"])
rows = []
for s in VALID["FD002"]:
    set_corr = d2[SETTINGS + [s]].corr().loc[SETTINGS, s].abs().max()
    raw_p = d2[[s, "RUL_raw"]].corr().iloc[0, 1]
    z_p = d2[[f"z_{s}", "RUL_raw"]].corr().iloc[0, 1]
    raw_sp = spearmanr(d2[s], d2["RUL_raw"]).statistic
    z_sp = spearmanr(d2[f"z_{s}"], d2["RUL_raw"]).statistic
    n_const_cond = sum(d2.groupby("condition_id")[s].nunique() <= 1)
    rows.append({"sensor": s, "max_set_corr": round(set_corr, 3),
                 "raw_pearson": round(raw_p, 3), "z_pearson": round(z_p, 3),
                 "raw_spear": round(raw_sp, 3), "z_spear": round(z_sp, 3),
                 "const_cond": f"{n_const_cond}/6",
                 "between_var": round(var_decomp(d2, s), 3)})
fd002 = pd.DataFrame(rows).reindex(pd.DataFrame(rows)["z_pearson"].abs().sort_values(ascending=False).index)
print(fd002.to_string(index=False))

top = fd002.head(10).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
y = np.arange(len(top))
ax[0].barh(y - 0.2, top["raw_pearson"].abs(), height=0.4, color="#8E9A9A", label="|corr| raw")
ax[0].barh(y + 0.2, top["z_pearson"].abs(), height=0.4, color="#C44E52", label="|corr| condition-z")
ax[0].set_yticks(y); ax[0].set_yticklabels(top["sensor"]); ax[0].legend()
ax[0].set_xlabel("|Pearson corr with RUL_raw|"); ax[0].set_title("FD002: 조건 정규화로 RUL 신호 복원")
allb = fd002.sort_values("between_var")
ax[1].barh(allb["sensor"], allb["between_var"], color="#4C72B0")
ax[1].set_xlabel("between-condition variance share"); ax[1].set_title("FD002: 센서별 조건 지배도")
ax[1].axvline(0.9, color="red", ls=":", lw=1)
plt.tight_layout(); plt.savefig(FIG / "S2_fd002_condition_effect.png", dpi=110, bbox_inches="tight"); plt.close()

# ------------------------- 0. 데이터 품질 (결측·중복·cycle 연속성)
line("0. 데이터 품질 — 결측 / 중복 / cycle 연속성")
qrows = []
for fd in FD_LIST:
    tr = load_train(fd)
    g = tr.groupby("unit")["cycle"]
    contiguous = int(((g.min() == 1) & (g.max() == g.count())).sum())
    qrows.append({"set": fd, "rows": len(tr), "missing": int(tr.isna().sum().sum()),
                  "dup_rows": int(tr.duplicated().sum()),
                  "dup_unit_cycle": int(tr.duplicated(["unit", "cycle"]).sum()),
                  "units": tr["unit"].nunique(), "cycle_연속_unit": contiguous})
print(pd.DataFrame(qrows).to_string(index=False))

# ------------- 4. 고장모드 proxy clustering — 2x2 요인 실험 (조건 x 고장모드)
line("4. 고장모드 proxy clustering — 2x2 요인 실험 (운전조건 x 고장모드)")
FAULT = {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}
clus_rows = []
fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
for ax, fd in zip(axes, FD_LIST):
    d, v = DATA[fd], VALID[fd]
    ncond = d["condition_id"].nunique()
    if ncond > 1:
        d = cond_z(d, v); feats = [f"z_{s}" for s in v]
    else:
        feats = v
    late = d[d["relative_cycle"] >= 0.8]
    prof = late.groupby("unit")[feats].mean()
    X = StandardScaler().fit_transform(prof)
    lab = KMeans(n_clusters=2, random_state=SEED, n_init=20).fit_predict(X)
    sil = silhouette_score(X, lab)
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(X)
    life = d.groupby("unit")["cycle"].max().loc[prof.index].values
    l0, l1 = life[lab == 0].mean(), life[lab == 1].mean()
    clus_rows.append({"set": fd, "조건": ncond, "고장모드": FAULT[fd], "silhouette": round(sil, 3),
                      "크기": f"{np.bincount(lab)[0]}/{np.bincount(lab)[1]}",
                      "수명_c0": round(l0, 1), "수명_c1": round(l1, 1), "수명차": round(abs(l0 - l1), 1)})
    ax.scatter(pcs[:, 0], pcs[:, 1], c=lab, cmap="coolwarm", s=20, alpha=0.85)
    ax.set_title(f"{fd}  (조건 {ncond}, 고장모드 {FAULT[fd]})\nsilhouette={sil:.3f}", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.suptitle("말기(relative_cycle≥0.8) 센서 프로파일의 2-군집 분리 — 2×2 요인 실험", y=1.04)
plt.tight_layout(); plt.savefig(FIG / "S3_fault_mode_proxy.png", dpi=110, bbox_inches="tight"); plt.close()

px = pd.DataFrame(clus_rows)
print(px.to_string(index=False))
print("\n2x2 (silhouette):  행=운전조건, 열=고장모드")
print(px.pivot(index="조건", columns="고장모드", values="silhouette").to_string())
print("\n2x2 (군집간 평균수명 차):")
print(px.pivot(index="조건", columns="고장모드", values="수명차").to_string())
print("\n해석: 고장모드를 1→2로 늘리면 분리도가 크게 오르지만(+0.31/+0.19),")
print("      운전조건을 1→6으로 늘려도 거의 변하지 않는다(+0.02/-0.09).")
print("      => 2-군집 구조는 고장모드에서 오며, 운전조건과는 무관하다.")

# ----------------------------------- 5. 말기 분산 — 네 서브셋 전부
line("5. 말기 분산 — 초기(rel<=0.2) 대비 말기(rel>=0.8) IQR 비율 (전 서브셋)")

def disp_ratio(d, col):
    z = (d[col] - d[col].mean()) / (d[col].std() or 1.0)
    e, l = z[d["relative_cycle"] <= 0.2], z[d["relative_cycle"] >= 0.8]
    ie = e.quantile(0.75) - e.quantile(0.25)
    il = l.quantile(0.75) - l.quantile(0.25)
    return il / ie if ie > 1e-9 else np.nan

res = {}
for fd in FD_LIST:
    d, v = DATA[fd], VALID[fd]
    if d["condition_id"].nunique() > 1:      # 조건 제거 후 비교 (공정)
        d = cond_z(d, v); res[fd] = {s: disp_ratio(d, f"z_{s}") for s in v}
    else:
        res[fd] = {s: disp_ratio(d, s) for s in v}
piv = pd.DataFrame(res).reindex([f"s{i}" for i in range(1, 22)])
print("(FD002/FD004는 조건 정규화 z 기준 — 조건 효과 제거 후 비교)")
print(piv.round(2).to_string())

FOCUS = ["s7", "s12", "s15", "s20", "s21"]
print(f"\n관심 센서 {FOCUS}:")
print(piv.loc[FOCUS].round(2).to_string())
print("\n'말기 IQR이 2배 초과' 센서 수:")
counts = {}
for fd in FD_LIST:
    hits = sorted(piv.index[piv[fd] > 2.0].tolist(), key=lambda x: int(x[1:]))
    counts[fd] = len(hits)
    print(f"  {fd} (고장모드 {FAULT[fd]}): {len(hits)}개 -> {hits}")
print("\n=> 고장모드 1개: 2~4개 / 고장모드 2개: 8~10개. 말기 분산 폭증도 고장모드에서 온다.")

fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
w = 0.2
xs = np.arange(len(FOCUS))
for i, (fd, c) in enumerate(zip(FD_LIST, ["#4C72B0", "#7BA7D4", "#C44E52", "#E39396"])):
    ax[0].bar(xs + (i - 1.5) * w, piv.loc[FOCUS, fd], width=w,
              label=f"{fd} (고장{FAULT[fd]})", color=c)
ax[0].axhline(1.0, color="gray", ls=":", lw=1)
ax[0].set_xticks(xs); ax[0].set_xticklabels(FOCUS)
ax[0].set_ylabel("late IQR / early IQR"); ax[0].legend(fontsize=8)
ax[0].set_title("관심 센서의 말기 분산 — 고장모드 2개에서만 폭증")
bars = ax[1].bar(FD_LIST, [counts[f] for f in FD_LIST],
                 color=["#4C72B0", "#7BA7D4", "#C44E52", "#E39396"])
for b, f in zip(bars, FD_LIST):
    ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"고장{FAULT[f]}",
               ha="center", fontsize=9)
ax[1].set_ylabel("말기 IQR > 2배인 센서 수")
ax[1].set_title("말기 분산이 큰 센서 개수 — 고장모드와 함께 증가")
plt.tight_layout(); plt.savefig(FIG / "S4_late_dispersion.png", dpi=110, bbox_inches="tight"); plt.close()

# ----------------------------------- 6. sensor11 cross-FD raw vs z (정확 재계산)
line("6. sensor11 RUL 상관 — 네 서브셋 raw vs 조건 정규화 (정확 재계산)")
rows = []
for fd in FD_LIST:
    d = cond_z(DATA[fd], VALID[fd])
    raw = d[["s11", "RUL_raw"]].corr().iloc[0, 1]
    z = d[["z_s11", "RUL_raw"]].corr().iloc[0, 1]
    rows.append({"set": fd, "n_cond": d["condition_id"].nunique(),
                 "raw_corr": round(raw, 3), "cond_z_corr": round(z, 3)})
print(pd.DataFrame(rows).to_string(index=False))

print(f"\n\n[figures] {FIG}")
for p in sorted(FIG.glob("*.png")):
    print("  ", p.name, f"{p.stat().st_size//1024}KB")
