from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NOTEBOOK_OUT = ROOT / "_build" / "final_eda.ipynb"
REPORT_OUT = ROOT / "reports" / "eda_final_report.md"
FIG_DIR = ROOT / "reports" / "figures" / "eda_final"

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]
SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"sensor{i}" for i in range(1, 22)]
FEATURES = SETTINGS + SENSORS
COLS = ["unit", "cycle"] + SETTINGS + SENSORS


def ensure_dirs() -> None:
    NOTEBOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_train(fd: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)


def load_test(fd: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)


def load_rul(fd: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["true_RUL"]).assign(
        unit=lambda x: np.arange(1, len(x) + 1)
    )


def add_rul(df: pd.DataFrame, clip: int = 125) -> pd.DataFrame:
    out = df.copy()
    out["total_life"] = out.groupby("unit")["cycle"].transform("max")
    out["RUL_raw"] = out["total_life"] - out["cycle"]
    out["RUL"] = out["RUL_raw"].clip(upper=clip)
    out["relative_cycle"] = out["cycle"] / out["total_life"]
    return out


def rounded_settings(df: pd.DataFrame) -> pd.DataFrame:
    return df[SETTINGS].round({"setting1": 0, "setting2": 2, "setting3": 0})


def add_condition_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rounded = rounded_settings(out)
    out["condition"] = rounded.apply(
        lambda r: f"s1={r['setting1']:.0f}, s2={r['setting2']:.2f}, s3={r['setting3']:.0f}",
        axis=1,
    )
    return out


def savefig(fig: plt.Figure, filename: str) -> Path:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    life_rows = []
    for fd in FD_LIST:
        tr = load_train(fd)
        te = load_test(fd)
        rul = load_rul(fd)
        life = tr.groupby("unit")["cycle"].max()
        nunique = tr[SENSORS].nunique()
        constant_sensors = nunique[nunique <= 1].index.tolist()
        rows.append(
            {
                "set": fd,
                "train_rows": len(tr),
                "train_units": tr["unit"].nunique(),
                "test_rows": len(te),
                "test_units": te["unit"].nunique(),
                "life_min": int(life.min()),
                "life_median": float(life.median()),
                "life_mean": float(life.mean()),
                "life_max": int(life.max()),
                "rounded_conditions": len(rounded_settings(tr).drop_duplicates()),
                "constant_sensor_count": len(constant_sensors),
                "constant_sensors": ", ".join(constant_sensors) if constant_sensors else "(none)",
                "true_RUL_min": int(rul["true_RUL"].min()),
                "true_RUL_max": int(rul["true_RUL"].max()),
            }
        )
        life_rows.extend({"set": fd, "unit": unit, "life": value} for unit, value in life.items())
    return pd.DataFrame(rows), pd.DataFrame(life_rows)


def feature_screening(fd: str = "FD001") -> pd.DataFrame:
    df = add_rul(load_train(fd))
    rows = []
    for feature in FEATURES:
        value_share = df[feature].value_counts(dropna=False, normalize=True)
        corr = df[[feature, "RUL_raw"]].corr().iloc[0, 1]
        rows.append(
            {
                "feature": feature,
                "nunique": df[feature].nunique(),
                "std": df[feature].std(),
                "dominant_ratio": value_share.iloc[0],
                "corr_RUL_raw": corr,
                "abs_corr_RUL_raw": abs(corr) if pd.notna(corr) else np.nan,
            }
        )
    out = pd.DataFrame(rows)

    def classify(row: pd.Series) -> str:
        if row["nunique"] <= 1 or row["std"] == 0:
            return "drop_constant"
        if row["dominant_ratio"] >= 0.95 or row["std"] < 0.01:
            return "review_near_constant"
        if row["abs_corr_RUL_raw"] >= 0.30:
            return "keep_rul_signal"
        return "review_weak_signal"

    out["decision"] = out.apply(classify, axis=1)
    order = ["keep_rul_signal", "review_near_constant", "review_weak_signal", "drop_constant"]
    out["decision"] = pd.Categorical(out["decision"], categories=order, ordered=True)
    return out.sort_values(["decision", "abs_corr_RUL_raw"], ascending=[True, False])


def trend_correlation(fd: str = "FD001") -> pd.DataFrame:
    df = add_rul(load_train(fd))
    bins = np.linspace(0, 1, 41)
    rel = df.copy()
    rel["rel_bin"] = pd.cut(rel["relative_cycle"], bins=bins, include_lowest=True, labels=False)
    rul_max = min(250, df["RUL_raw"].max())
    rul = df[df["RUL_raw"] <= rul_max].copy()
    rul["failure_progress"] = 1 - (rul["RUL_raw"] / rul_max)
    rul["progress_bin"] = pd.cut(rul["failure_progress"], bins=bins, include_lowest=True, labels=False)

    rows = []
    for feature in FEATURES:
        corr_rul = df[[feature, "RUL_raw"]].corr().iloc[0, 1]
        corr_rel = df[[feature, "relative_cycle"]].corr().iloc[0, 1]
        rel_curve = rel.groupby("rel_bin")[feature].mean().reindex(range(40)).to_numpy(dtype=float)
        rul_curve = rul.groupby("progress_bin")[feature].mean().reindex(range(40)).to_numpy(dtype=float)
        valid = np.isfinite(rel_curve) & np.isfinite(rul_curve)
        if valid.sum() > 2 and np.nanstd(rel_curve[valid]) > 0 and np.nanstd(rul_curve[valid]) > 0:
            curve_corr = np.corrcoef(rel_curve[valid], rul_curve[valid])[0, 1]
        else:
            curve_corr = np.nan
        rows.append(
            {
                "feature": feature,
                "corr_RUL_raw": corr_rul,
                "abs_corr_RUL_raw": abs(corr_rul) if pd.notna(corr_rul) else np.nan,
                "corr_relative_cycle": corr_rel,
                "curve_corr_rel_vs_rul": curve_corr,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_corr_RUL_raw", ascending=False)


def make_figures() -> dict[str, str]:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    summary, life_df = build_summary()
    figs: dict[str, str] = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.barplot(data=summary, x="set", y="train_rows", ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Train row count")
    sns.barplot(data=summary, x="set", y="train_units", ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("Train unit count")
    sns.barplot(data=summary, x="set", y="rounded_conditions", ax=axes[1, 0], color="#DD8452")
    axes[1, 0].set_title("Rounded operating condition count")
    sensor_counts = summary.melt(
        id_vars="set",
        value_vars=["constant_sensor_count"],
        var_name="type",
        value_name="count",
    )
    sns.barplot(data=sensor_counts, x="set", y="count", ax=axes[1, 1], color="#8172B2")
    axes[1, 1].set_title("Constant sensor count")
    fig.tight_layout()
    figs["dataset_overview"] = savefig(fig, "01_dataset_overview.png").name

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    sns.boxplot(data=life_df, x="set", y="life", ax=axes[0])
    axes[0].set_title("Total life by FD")
    sns.histplot(data=life_df, x="life", hue="set", bins=30, element="step", fill=False, stat="probability", common_norm=False, ax=axes[1])
    axes[1].set_title("Normalized life histogram")
    sns.ecdfplot(data=life_df, x="life", hue="set", ax=axes[2])
    axes[2].set_title("Life ECDF")
    fig.tight_layout()
    figs["life_distribution"] = savefig(fig, "02_life_distribution.png").name

    fig, axes = plt.subplots(2, 2, figsize=(18, 8.5), sharex=True)
    for ax, fd in zip(axes.ravel(), FD_LIST):
        tr = load_train(fd)
        corr = tr[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, linewidths=0.2, cbar=True)
        ax.set_title(f"{fd}: corr(setting, sensor)")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    figs["setting_sensor_corr"] = savefig(fig, "03_setting_sensor_corr.png").name

    corr_rows = []
    for fd in ["FD002", "FD004"]:
        tr = load_train(fd)
        corr = tr[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
        for sensor in SENSORS:
            corr_rows.append({"set": fd, "sensor": sensor, "max_abs_corr": corr[sensor].abs().max()})
    top_by_fd = pd.DataFrame(corr_rows).sort_values(["set", "max_abs_corr"], ascending=[True, False]).groupby("set").head(6)
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.ravel()
    idx = 0
    for fd in ["FD002", "FD004"]:
        tr = add_condition_label(load_train(fd))
        order = tr.groupby("condition").size().sort_values(ascending=False).index.tolist()
        for sensor in top_by_fd[top_by_fd["set"] == fd]["sensor"]:
            ax = axes[idx]
            sns.boxplot(data=tr, x="condition", y=sensor, order=order, ax=ax, color="#D6EAF8", fliersize=0.5)
            ax.set_title(f"{fd}: {sensor} by condition")
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.set_xlabel("")
            idx += 1
    fig.tight_layout()
    figs["condition_sensor_boxplots"] = savefig(fig, "04_condition_sensor_boxplots.png").name

    constant_union = sorted(
        set().union(
            *[
                set(load_train(fd)[SENSORS].nunique().loc[lambda s: s <= 1].index)
                for fd in FD_LIST
            ]
        ),
        key=lambda x: int(x.replace("sensor", "")),
    )
    rows = []
    for fd in FD_LIST:
        tr = load_train(fd)
        for sensor in constant_union:
            rows.append({"set": fd, "sensor": sensor, "nunique": tr[sensor].nunique(), "std": tr[sensor].std()})
    constant_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(constant_df.pivot(index="sensor", columns="set", values="nunique"), annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("nunique for sensors constant in at least one FD")
    std_mat = constant_df.pivot(index="sensor", columns="set", values="std").replace(0, np.nan)
    sns.heatmap(np.log10(std_mat), annot=True, fmt=".2f", cmap="mako", ax=axes[1])
    axes[1].set_title("log10(std); blank = std 0")
    fig.tight_layout()
    figs["constant_sensor_check"] = savefig(fig, "05_constant_sensor_check.png").name

    screen = feature_screening("FD001")
    palette = {
        "keep_rul_signal": "#4C72B0",
        "review_near_constant": "#DD8452",
        "review_weak_signal": "#8172B2",
        "drop_constant": "#8C8C8C",
    }
    plot_screen = screen.copy()
    plot_screen["std_log10"] = np.log10(plot_screen["std"].replace(0, np.nan))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.scatterplot(
        data=plot_screen,
        x="std_log10",
        y="abs_corr_RUL_raw",
        hue="decision",
        size="nunique",
        sizes=(35, 220),
        palette=palette,
        ax=axes[0],
    )
    for _, row in plot_screen.dropna(subset=["std_log10", "abs_corr_RUL_raw"]).iterrows():
        axes[0].text(row["std_log10"] + 0.02, row["abs_corr_RUL_raw"], row["feature"], fontsize=7)
    axes[0].axhline(0.30, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(np.log10(0.01), color="black", linestyle=":", linewidth=1)
    axes[0].set_title("Signal strength vs variance")
    bar = screen.sort_values("abs_corr_RUL_raw", ascending=True)
    axes[1].barh(bar["feature"], bar["abs_corr_RUL_raw"], color=bar["decision"].astype(str).map(palette))
    axes[1].axvline(0.30, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("|corr(feature, RUL)|")
    dom = screen.sort_values("dominant_ratio", ascending=True)
    axes[2].barh(dom["feature"], dom["dominant_ratio"], color=dom["decision"].astype(str).map(palette))
    axes[2].axvline(0.95, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Dominant value ratio")
    fig.tight_layout()
    figs["feature_screening"] = savefig(fig, "06_feature_screening_fd001.png").name

    fd = "FD001"
    df = add_rul(load_train(fd))
    selected_units = [1, 2, 3, 10, 20]
    ncols = 4
    nrows = int(np.ceil(len(FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.25))
    axes = axes.ravel()
    for ax, feature in zip(axes, FEATURES):
        for unit in selected_units:
            g = df[df["unit"] == unit]
            ax.plot(g["cycle"], g[feature], marker="o", markersize=1.8, linewidth=0.65, alpha=0.78, label=f"u{unit}")
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes[len(FEATURES) :]:
        ax.axis("off")
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle("FD001: all feature time series for selected units", y=1.002)
    fig.tight_layout()
    figs["all_feature_timeseries"] = savefig(fig, "07_all_feature_timeseries_fd001.png").name

    trend = df.copy()
    trend["rel_bin"] = pd.cut(trend["relative_cycle"], bins=np.linspace(0, 1, 41), include_lowest=True, labels=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.25), sharex=True)
    axes = axes.ravel()
    for ax, feature in zip(axes, FEATURES):
        agg = trend.groupby("rel_bin")[feature].agg(["mean", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]).reset_index()
        agg.columns = ["rel_bin", "mean", "q10", "q90"]
        agg["relative_cycle"] = (agg["rel_bin"] + 0.5) / 40
        ax.plot(agg["relative_cycle"], agg["mean"], marker="o", markersize=2.4, linewidth=1.1, color="#1f618d")
        ax.fill_between(agg["relative_cycle"], agg["q10"], agg["q90"], color="#1f618d", alpha=0.15)
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes[len(FEATURES) :]:
        ax.axis("off")
    fig.suptitle("FD001: mean feature trend over normalized life", y=1.002)
    fig.tight_layout()
    figs["mean_trends"] = savefig(fig, "08_mean_trends_fd001.png").name

    rul_max = min(250, df["RUL_raw"].max())
    rul_df = df[df["RUL_raw"] <= rul_max].copy()
    rul_df["rul_bin"] = pd.cut(rul_df["RUL_raw"], bins=np.linspace(0, rul_max, 51), include_lowest=True, labels=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.25), sharex=True)
    axes = axes.ravel()
    for ax, feature in zip(axes, FEATURES):
        agg = rul_df.groupby("rul_bin")[feature].mean().reset_index()
        agg["RUL_mid"] = (agg["rul_bin"] + 0.5) * (rul_max / 50)
        ax.plot(agg["RUL_mid"], agg[feature], marker="o", markersize=2.4, linewidth=1.1, color="#C44E52")
        ax.invert_xaxis()
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes[len(FEATURES) :]:
        ax.axis("off")
    fig.suptitle("FD001: feature trend by RUL_raw", y=1.002)
    fig.tight_layout()
    figs["rul_trends"] = savefig(fig, "09_rul_trends_fd001.png").name

    corr = trend_correlation("FD001")
    valid_corr = corr.dropna(subset=["corr_RUL_raw"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    bar = valid_corr.sort_values("corr_RUL_raw")
    axes[0].barh(bar["feature"], bar["corr_RUL_raw"], color=np.where(bar["corr_RUL_raw"] >= 0, "#4C72B0", "#C44E52"))
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Feature vs RUL_raw correlation")
    curve = corr.dropna(subset=["curve_corr_rel_vs_rul"]).sort_values("curve_corr_rel_vs_rul")
    axes[1].barh(curve["feature"], curve["curve_corr_rel_vs_rul"], color="#8172B2")
    axes[1].set_title("Mean trend curve similarity")
    fig.tight_layout()
    figs["trend_correlation"] = savefig(fig, "10_trend_correlation_fd001.png").name

    valid_sensors = [s for s in SENSORS if load_train("FD001")[s].nunique() > 1]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(load_train("FD001")[valid_sensors].corr(), cmap="vlag", center=0, square=True, linewidths=0.25, ax=ax)
    ax.set_title("FD001 valid sensor correlation")
    fig.tight_layout()
    figs["sensor_corr"] = savefig(fig, "11_sensor_corr_fd001.png").name

    return figs


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    data = data.reset_index(drop=True)
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in data.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(figs: dict[str, str]) -> None:
    summary, _ = build_summary()
    life_summary = summary[["set", "train_units", "life_min", "life_median", "life_mean", "life_max", "rounded_conditions", "constant_sensor_count"]]
    screen = feature_screening("FD001")
    decisions = screen.groupby("decision", observed=True)["feature"].apply(lambda x: ", ".join(x)).reset_index()
    corr = trend_correlation("FD001")[["feature", "corr_RUL_raw", "curve_corr_rel_vs_rul"]].head(12)

    def fig_md(key: str, title: str) -> str:
        return f"![{title}](figures/eda_final/{figs[key]})\n\n"

    report = f"""# Leo Final EDA Report - NASA C-MAPSS

## 목적

최종 EDA는 raw C-MAPSS 데이터의 구조, FD별 차이, setting-sensor 관계, FD001 기준 feature screening, 시계열/RUL trend를 모델 전처리 관점에서 정리한다.

## 핵심 결론

- `unit`은 엔진 기종이 아니라 개별 엔진 run-to-failure 시계열 ID다. raw 파일에는 엔진 기종/모델 컬럼이 없다.
- FD001/FD003은 rounded operating condition이 1개이고, FD002/FD004는 6개다.
- FD002/FD004는 unit 수와 row 수가 더 크므로 histogram의 count가 크게 보일 수 있다. 수명 histogram의 count는 raw row 수가 아니라 해당 수명 bin에 포함된 engine unit 수다.
- FD003/FD004는 FD001/FD002보다 긴 수명 꼬리를 보인다. 이를 엔진 기종 차이로 결론내릴 수는 없고, 공개 메타정보상 고장모드 수 차이와 연결해서 해석하는 것이 안전하다.
- FD001 baseline에서는 상수 feature 제거, near-constant feature 검토, 유효 sensor 유지 후 rolling/slope/delta feature 추가가 적절하다.
- FD002/FD004에서는 setting이 sensor 분포에 강하게 영향을 주므로 condition-aware scaling 또는 condition별 normalization을 검토해야 한다.

## FD 구조 요약

{df_to_markdown(life_summary.round(2))}

{fig_md("dataset_overview", "Dataset Overview")}

{fig_md("life_distribution", "Life Distribution")}

## Setting-Sensor 관계

FD002/FD004는 6개 운전조건이 섞여 있어 sensor 값이 condition에 따라 층처럼 갈라진다. 따라서 FD001 기준 전처리를 그대로 적용하지 말고, FD별/condition별 scaling을 비교해야 한다.

{fig_md("setting_sensor_corr", "Setting Sensor Correlation")}

{fig_md("condition_sensor_boxplots", "Condition Sensor Boxplots")}

{fig_md("constant_sensor_check", "Constant Sensor Check")}

## FD001 Feature Screening

{df_to_markdown(decisions)}

{fig_md("feature_screening", "FD001 Feature Screening")}

## FD001 시계열 및 RUL Trend

원본 feature 자체가 RUL 신호를 가지므로 smoothing된 값만 쓰는 것보다, raw scaled value와 rolling mean/std/slope/delta를 함께 넣는 방식이 더 안전하다. `relative_cycle`은 EDA용이며 test 시점에는 total life를 모르므로 모델 feature로 쓰면 leakage다.

{fig_md("all_feature_timeseries", "All Feature Time Series")}

{fig_md("mean_trends", "Mean Feature Trends")}

{fig_md("rul_trends", "RUL Trends")}

## Trend Correlation

{df_to_markdown(corr.round(3))}

{fig_md("trend_correlation", "Trend Correlation")}

{fig_md("sensor_corr", "Sensor Correlation")}

## 전처리 제안

1. FD별로 train 기준 상수 feature를 자동 제거한다.
2. `dominant_ratio >= 0.95` 또는 매우 낮은 표준편차 feature는 near-constant로 별도 검토한다.
3. FD001 baseline은 raw scaled sensor + rolling mean/std/slope/delta feature를 우선 실험한다.
4. smoothing 또는 평탄화만 적용한 feature set은 별도 ablation으로 비교한다.
5. FD002/FD004는 rounded setting condition별 scaling과 전체 scaling을 비교한다.
6. 고상관 sensor는 초기 baseline에서는 유지하고, correlation pruning/PCA는 후속 실험군으로 둔다.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def build_notebook() -> None:
    code_setup = r"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.9)

for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "data" / "raw").exists():
        ROOT = candidate
        break
else:
    raise FileNotFoundError("Cannot find data/raw from current working directory or its parents.")
RAW = ROOT / "data" / "raw"

FD_LIST = ["FD001", "FD002", "FD003", "FD004"]
SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"sensor{i}" for i in range(1, 22)]
FEATURES = SETTINGS + SENSORS
COLS = ["unit", "cycle"] + SETTINGS + SENSORS

def load_train(fd):
    return pd.read_csv(RAW / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)

def load_test(fd):
    return pd.read_csv(RAW / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)

def load_rul(fd):
    return pd.read_csv(RAW / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["true_RUL"]).assign(
        unit=lambda x: np.arange(1, len(x) + 1)
    )

def add_rul(df, clip=125):
    out = df.copy()
    out["total_life"] = out.groupby("unit")["cycle"].transform("max")
    out["RUL_raw"] = out["total_life"] - out["cycle"]
    out["RUL"] = out["RUL_raw"].clip(upper=clip)
    out["relative_cycle"] = out["cycle"] / out["total_life"]
    return out

def rounded_settings(df):
    return df[SETTINGS].round({"setting1": 0, "setting2": 2, "setting3": 0})

def add_condition_label(df):
    out = df.copy()
    rounded = rounded_settings(out)
    out["condition"] = rounded.apply(
        lambda r: f"s1={r['setting1']:.0f}, s2={r['setting2']:.2f}, s3={r['setting3']:.0f}",
        axis=1,
    )
    return out
""".strip()

    code_summary = r"""
summary_rows = []
life_rows = []
for fd in FD_LIST:
    tr = load_train(fd)
    te = load_test(fd)
    rul = load_rul(fd)
    life = tr.groupby("unit")["cycle"].max()
    constant_sensors = tr[SENSORS].nunique().loc[lambda s: s <= 1].index.tolist()
    summary_rows.append({
        "set": fd,
        "train_rows": len(tr),
        "train_units": tr["unit"].nunique(),
        "test_rows": len(te),
        "test_units": te["unit"].nunique(),
        "life_min": life.min(),
        "life_median": life.median(),
        "life_mean": life.mean(),
        "life_max": life.max(),
        "rounded_conditions": len(rounded_settings(tr).drop_duplicates()),
        "constant_sensor_count": len(constant_sensors),
        "constant_sensors": ", ".join(constant_sensors) if constant_sensors else "(none)",
        "true_RUL_min": rul["true_RUL"].min(),
        "true_RUL_max": rul["true_RUL"].max(),
    })
    life_rows.extend({"set": fd, "unit": unit, "life": value} for unit, value in life.items())

summary = pd.DataFrame(summary_rows)
life_df = pd.DataFrame(life_rows)
display(summary.round(2))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sns.barplot(data=summary, x="set", y="train_rows", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Train row count")
sns.barplot(data=summary, x="set", y="train_units", ax=axes[0, 1], color="#55A868")
axes[0, 1].set_title("Train unit count")
sns.barplot(data=summary, x="set", y="rounded_conditions", ax=axes[1, 0], color="#DD8452")
axes[1, 0].set_title("Rounded operating condition count")
sns.barplot(data=summary, x="set", y="constant_sensor_count", ax=axes[1, 1], color="#8172B2")
axes[1, 1].set_title("Constant sensor count")
plt.tight_layout()
plt.show()
""".strip()

    code_life = r"""
life_summary = life_df.groupby("set")["life"].agg(
    units="count",
    min="min",
    q25=lambda x: x.quantile(0.25),
    median="median",
    mean="mean",
    q75=lambda x: x.quantile(0.75),
    max="max",
).reset_index()
display(life_summary.round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
sns.boxplot(data=life_df, x="set", y="life", ax=axes[0])
axes[0].set_title("Total life by FD")
sns.histplot(data=life_df, x="life", hue="set", bins=30, element="step", fill=False, stat="probability", common_norm=False, ax=axes[1])
axes[1].set_title("Normalized life histogram")
sns.ecdfplot(data=life_df, x="life", hue="set", ax=axes[2])
axes[2].set_title("Life ECDF")
plt.tight_layout()
plt.show()
""".strip()

    code_setting_sensor = r"""
fig, axes = plt.subplots(2, 2, figsize=(18, 8.5), sharex=True)
for ax, fd in zip(axes.ravel(), FD_LIST):
    tr = load_train(fd)
    corr = tr[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, linewidths=0.2, cbar=True)
    ax.set_title(f"{fd}: corr(setting, sensor)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
plt.tight_layout()
plt.show()

for fd in ["FD002", "FD004"]:
    tr = add_condition_label(load_train(fd))
    corr = tr[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
    top_sensors = corr.abs().max(axis=0).sort_values(ascending=False).head(6).index.tolist()
    order = tr.groupby("condition").size().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, sensor in zip(axes.ravel(), top_sensors):
        sns.boxplot(data=tr, x="condition", y=sensor, order=order, ax=ax, color="#D6EAF8", fliersize=0.5)
        ax.set_title(f"{fd}: {sensor} by condition")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.show()
""".strip()

    code_screen = r"""
df = add_rul(load_train("FD001"))
screen_rows = []
for feature in FEATURES:
    value_share = df[feature].value_counts(dropna=False, normalize=True)
    corr = df[[feature, "RUL_raw"]].corr().iloc[0, 1]
    screen_rows.append({
        "feature": feature,
        "nunique": df[feature].nunique(),
        "std": df[feature].std(),
        "dominant_ratio": value_share.iloc[0],
        "corr_RUL_raw": corr,
        "abs_corr_RUL_raw": abs(corr) if pd.notna(corr) else np.nan,
    })

feature_screen = pd.DataFrame(screen_rows)

def classify_feature(row):
    if row["nunique"] <= 1 or row["std"] == 0:
        return "drop_constant"
    if row["dominant_ratio"] >= 0.95 or row["std"] < 0.01:
        return "review_near_constant"
    if row["abs_corr_RUL_raw"] >= 0.30:
        return "keep_rul_signal"
    return "review_weak_signal"

feature_screen["decision"] = feature_screen.apply(classify_feature, axis=1)
display(feature_screen.sort_values(["decision", "abs_corr_RUL_raw"], ascending=[True, False]).round(5))

palette = {
    "keep_rul_signal": "#4C72B0",
    "review_near_constant": "#DD8452",
    "review_weak_signal": "#8172B2",
    "drop_constant": "#8C8C8C",
}
plot_screen = feature_screen.copy()
plot_screen["std_log10"] = np.log10(plot_screen["std"].replace(0, np.nan))
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.scatterplot(
    data=plot_screen,
    x="std_log10",
    y="abs_corr_RUL_raw",
    hue="decision",
    size="nunique",
    sizes=(35, 220),
    palette=palette,
    ax=axes[0],
)
for _, row in plot_screen.dropna(subset=["std_log10", "abs_corr_RUL_raw"]).iterrows():
    axes[0].text(row["std_log10"] + 0.02, row["abs_corr_RUL_raw"], row["feature"], fontsize=7)
axes[0].axhline(0.30, color="black", linestyle="--", linewidth=1)
axes[0].axvline(np.log10(0.01), color="black", linestyle=":", linewidth=1)
axes[0].set_title("Signal strength vs variance")
bar = feature_screen.sort_values("abs_corr_RUL_raw", ascending=True)
axes[1].barh(bar["feature"], bar["abs_corr_RUL_raw"], color=bar["decision"].map(palette))
axes[1].axvline(0.30, color="black", linestyle="--", linewidth=1)
axes[1].set_title("|corr(feature, RUL)|")
dom = feature_screen.sort_values("dominant_ratio", ascending=True)
axes[2].barh(dom["feature"], dom["dominant_ratio"], color=dom["decision"].map(palette))
axes[2].axvline(0.95, color="black", linestyle="--", linewidth=1)
axes[2].set_title("Dominant value ratio")
plt.tight_layout()
plt.show()
""".strip()

    code_timeseries = r"""
selected_units = [1, 2, 3, 10, 20]
ncols = 4
nrows = int(np.ceil(len(FEATURES) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.25))
axes = axes.ravel()
for ax, feature in zip(axes, FEATURES):
    for unit in selected_units:
        g = df[df["unit"] == unit]
        ax.plot(g["cycle"], g[feature], marker="o", markersize=1.8, linewidth=0.65, alpha=0.78, label=f"u{unit}")
    ax.set_title(feature, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
for ax in axes[len(FEATURES):]:
    ax.axis("off")
axes[0].legend(fontsize=7, ncol=3)
fig.suptitle("FD001: all feature time series for selected units", y=1.002)
plt.tight_layout()
plt.show()
""".strip()

    code_trends = r"""
trend = df.copy()
trend["rel_bin"] = pd.cut(trend["relative_cycle"], bins=np.linspace(0, 1, 41), include_lowest=True, labels=False)
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.25), sharex=True)
axes = axes.ravel()
for ax, feature in zip(axes, FEATURES):
    agg = trend.groupby("rel_bin")[feature].agg(["mean", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]).reset_index()
    agg.columns = ["rel_bin", "mean", "q10", "q90"]
    agg["relative_cycle"] = (agg["rel_bin"] + 0.5) / 40
    ax.plot(agg["relative_cycle"], agg["mean"], marker="o", markersize=2.4, linewidth=1.1, color="#1f618d")
    ax.fill_between(agg["relative_cycle"], agg["q10"], agg["q90"], color="#1f618d", alpha=0.15)
    ax.set_title(feature, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
for ax in axes[len(FEATURES):]:
    ax.axis("off")
fig.suptitle("FD001: mean feature trend over normalized life", y=1.002)
plt.tight_layout()
plt.show()

rul_max = min(250, df["RUL_raw"].max())
rul_df = df[df["RUL_raw"] <= rul_max].copy()
rul_df["rul_bin"] = pd.cut(rul_df["RUL_raw"], bins=np.linspace(0, rul_max, 51), include_lowest=True, labels=False)
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.25), sharex=True)
axes = axes.ravel()
for ax, feature in zip(axes, FEATURES):
    agg = rul_df.groupby("rul_bin")[feature].mean().reset_index()
    agg["RUL_mid"] = (agg["rul_bin"] + 0.5) * (rul_max / 50)
    ax.plot(agg["RUL_mid"], agg[feature], marker="o", markersize=2.4, linewidth=1.1, color="#C44E52")
    ax.invert_xaxis()
    ax.set_title(feature, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
for ax in axes[len(FEATURES):]:
    ax.axis("off")
fig.suptitle("FD001: feature trend by RUL_raw", y=1.002)
plt.tight_layout()
plt.show()
""".strip()

    code_corr = r"""
trend_corr_rows = []
bins = np.linspace(0, 1, 41)
rel = df.copy()
rel["rel_bin"] = pd.cut(rel["relative_cycle"], bins=bins, include_lowest=True, labels=False)
rul_max = min(250, df["RUL_raw"].max())
rul = df[df["RUL_raw"] <= rul_max].copy()
rul["failure_progress"] = 1 - (rul["RUL_raw"] / rul_max)
rul["progress_bin"] = pd.cut(rul["failure_progress"], bins=bins, include_lowest=True, labels=False)

for feature in FEATURES:
    corr_rul = df[[feature, "RUL_raw"]].corr().iloc[0, 1]
    rel_curve = rel.groupby("rel_bin")[feature].mean().reindex(range(40)).to_numpy(dtype=float)
    rul_curve = rul.groupby("progress_bin")[feature].mean().reindex(range(40)).to_numpy(dtype=float)
    valid = np.isfinite(rel_curve) & np.isfinite(rul_curve)
    if valid.sum() > 2 and np.nanstd(rel_curve[valid]) > 0 and np.nanstd(rul_curve[valid]) > 0:
        curve_corr = np.corrcoef(rel_curve[valid], rul_curve[valid])[0, 1]
    else:
        curve_corr = np.nan
    trend_corr_rows.append({
        "feature": feature,
        "corr_RUL_raw": corr_rul,
        "abs_corr_RUL_raw": abs(corr_rul) if pd.notna(corr_rul) else np.nan,
        "curve_corr_rel_vs_rul": curve_corr,
    })

trend_corr = pd.DataFrame(trend_corr_rows).sort_values("abs_corr_RUL_raw", ascending=False)
display(trend_corr.round(4))

valid_corr = trend_corr.dropna(subset=["corr_RUL_raw"])
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
bar = valid_corr.sort_values("corr_RUL_raw")
axes[0].barh(bar["feature"], bar["corr_RUL_raw"], color=np.where(bar["corr_RUL_raw"] >= 0, "#4C72B0", "#C44E52"))
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title("Feature vs RUL_raw correlation")
curve = trend_corr.dropna(subset=["curve_corr_rel_vs_rul"]).sort_values("curve_corr_rel_vs_rul")
axes[1].barh(curve["feature"], curve["curve_corr_rel_vs_rul"], color="#8172B2")
axes[1].set_title("Mean trend curve similarity")
plt.tight_layout()
plt.show()

valid_sensors = [s for s in SENSORS if load_train("FD001")[s].nunique() > 1]
plt.figure(figsize=(10, 8))
sns.heatmap(load_train("FD001")[valid_sensors].corr(), cmap="vlag", center=0, square=True, linewidths=0.25)
plt.title("FD001 valid sensor correlation")
plt.show()
""".strip()

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# Final EDA - NASA C-MAPSS\n\n불필요한 설명을 줄인 최종 EDA 노트북이다. 모든 결과는 raw 파일에서 직접 계산한다."),
        nbf.v4.new_code_cell(code_setup),
        nbf.v4.new_markdown_cell("## 1. FD 구조 요약"),
        nbf.v4.new_code_cell(code_summary),
        nbf.v4.new_markdown_cell("## 2. 수명 분포\n\nHistogram count는 수명 bin에 포함된 engine unit 개수다. FD 간 비교는 normalized histogram과 ECDF를 함께 본다."),
        nbf.v4.new_code_cell(code_life),
        nbf.v4.new_markdown_cell("## 3. Setting-Sensor 관계\n\nFD002/FD004는 6개 운전조건이 섞여 있어 condition-aware scaling이 필요할 수 있다."),
        nbf.v4.new_code_cell(code_setting_sensor),
        nbf.v4.new_markdown_cell("## 4. FD001 Feature Screening"),
        nbf.v4.new_code_cell(code_screen),
        nbf.v4.new_markdown_cell("## 5. FD001 전체 Feature 시계열"),
        nbf.v4.new_code_cell(code_timeseries),
        nbf.v4.new_markdown_cell("## 6. Mean Trend와 RUL Trend"),
        nbf.v4.new_code_cell(code_trends),
        nbf.v4.new_markdown_cell("## 7. 상관관계 요약"),
        nbf.v4.new_code_cell(code_corr),
        nbf.v4.new_markdown_cell(
            "## 전처리 결론\n\n"
            "- FD별 train 기준 상수 feature를 자동 제거한다.\n"
            "- FD001에서는 `sensor6`, `setting1`, `setting2`를 near-constant로 검토한다.\n"
            "- raw scaled value를 유지하고 rolling mean/std/slope/delta를 추가하는 방식이 평탄화만 하는 것보다 안전하다.\n"
            "- `relative_cycle`은 EDA용이며 test에서는 total life를 모르므로 모델 feature로 쓰지 않는다.\n"
            "- FD002/FD004는 condition-aware scaling 또는 condition별 normalization을 비교한다."
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NOTEBOOK_OUT)


def main() -> None:
    ensure_dirs()
    figs = make_figures()
    build_report(figs)
    build_notebook()
    manifest = {
        "notebook": str(NOTEBOOK_OUT.relative_to(ROOT)),
        "report": str(REPORT_OUT.relative_to(ROOT)),
        "figures": [str((FIG_DIR / name).relative_to(ROOT)) for name in figs.values()],
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
