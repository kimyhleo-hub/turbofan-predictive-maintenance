from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NOTEBOOK_OUT = ROOT / "_build" / "fd004_deep_dive.ipynb"
REPORT_OUT = ROOT / "reports" / "fd004_deep_dive_report.md"
FIG_DIR = ROOT / "reports" / "figures" / "fd004_deep_dive"

FD_LIST = ["FD001", "FD004"]
SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"sensor{i}" for i in range(1, 22)]
FEATURES = SETTINGS + SENSORS
COLS = ["unit", "cycle"] + SETTINGS + SENSORS


def ensure_dirs() -> None:
    NOTEBOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in FIG_DIR.glob("*.png"):
        old_png.unlink()


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


def add_condition_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = add_condition_label(df)
    for sensor in SENSORS:
        mean = out.groupby("condition")[sensor].transform("mean")
        std = out.groupby("condition")[sensor].transform("std").replace(0, np.nan)
        out[f"z_{sensor}"] = ((out[sensor] - mean) / std).fillna(0)
    return out


def savefig(fig: plt.Figure, filename: str) -> Path:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_overview() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    life_rows = []
    for fd in FD_LIST:
        train = load_train(fd)
        test = load_test(fd)
        rul = load_rul(fd)
        life = train.groupby("unit")["cycle"].max()
        constant = train[SENSORS].nunique().loc[lambda s: s <= 1].index.tolist()
        rows.append(
            {
                "set": fd,
                "train_rows": len(train),
                "train_units": train["unit"].nunique(),
                "test_rows": len(test),
                "test_units": test["unit"].nunique(),
                "life_min": int(life.min()),
                "life_median": float(life.median()),
                "life_mean": float(life.mean()),
                "life_max": int(life.max()),
                "rounded_conditions": len(rounded_settings(train).drop_duplicates()),
                "constant_sensor_count": len(constant),
                "constant_sensors": ", ".join(constant) if constant else "(none)",
                "test_RUL_min": int(rul["true_RUL"].min()),
                "test_RUL_max": int(rul["true_RUL"].max()),
            }
        )
        life_rows.extend({"set": fd, "unit": unit, "life": value} for unit, value in life.items())
    return pd.DataFrame(rows), pd.DataFrame(life_rows)


def fd004_corr_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_condition_zscores(add_rul(load_train("FD004")))
    rows = []
    for sensor in SENSORS:
        raw_corr = df[[sensor, "RUL_raw"]].corr().iloc[0, 1]
        z_corr = df[[f"z_{sensor}", "RUL_raw"]].corr().iloc[0, 1]
        setting_abs = df[SETTINGS + [sensor]].corr().loc[SETTINGS, sensor].abs().max()
        rows.append(
            {
                "sensor": sensor,
                "nunique": df[sensor].nunique(),
                "std": df[sensor].std(),
                "max_abs_setting_corr": setting_abs,
                "raw_corr_RUL": raw_corr,
                "raw_abs_corr_RUL": abs(raw_corr) if pd.notna(raw_corr) else np.nan,
                "cond_z_corr_RUL": z_corr,
                "cond_z_abs_corr_RUL": abs(z_corr) if pd.notna(z_corr) else np.nan,
                "abs_corr_change": (abs(z_corr) if pd.notna(z_corr) else np.nan)
                - (abs(raw_corr) if pd.notna(raw_corr) else np.nan),
            }
        )
    corr = pd.DataFrame(rows).sort_values("cond_z_abs_corr_RUL", ascending=False)
    setting_corr = df[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
    return corr, setting_corr


def sensor9_sensor14_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = add_condition_zscores(add_rul(load_train("FD004")))

    def variance_decomp(sensor: str) -> tuple[float, float]:
        overall = df[sensor].mean()
        total_ss = ((df[sensor] - overall) ** 2).sum()
        condition_mean = df.groupby("condition")[sensor].transform("mean")
        between_ss = ((condition_mean - overall) ** 2).sum()
        within_ss = ((df[sensor] - condition_mean) ** 2).sum()
        return between_ss / total_ss, within_ss / total_ss

    def partial_corr_by_settings(a: str, b: str) -> float:
        x = df[SETTINGS].to_numpy()
        ya = df[a].to_numpy()
        yb = df[b].to_numpy()
        ra = ya - LinearRegression().fit(x, ya).predict(x)
        rb = yb - LinearRegression().fit(x, yb).predict(x)
        return float(np.corrcoef(ra, rb)[0, 1])

    rows = [
        {
            "metric": "FD004 raw corr(sensor9, sensor14)",
            "value": df[["sensor9", "sensor14"]].corr().iloc[0, 1],
        },
        {
            "metric": "FD004 condition-z corr(z_sensor9, z_sensor14)",
            "value": df[["z_sensor9", "z_sensor14"]].corr().iloc[0, 1],
        },
        {
            "metric": "FD004 partial corr controlling numeric settings",
            "value": partial_corr_by_settings("sensor9", "sensor14"),
        },
        {
            "metric": "FD004 corr(sensor9, RUL_raw)",
            "value": df[["sensor9", "RUL_raw"]].corr().iloc[0, 1],
        },
        {
            "metric": "FD004 corr(sensor14, RUL_raw)",
            "value": df[["sensor14", "RUL_raw"]].corr().iloc[0, 1],
        },
        {
            "metric": "FD004 corr(z_sensor9, RUL_raw)",
            "value": df[["z_sensor9", "RUL_raw"]].corr().iloc[0, 1],
        },
        {
            "metric": "FD004 corr(z_sensor14, RUL_raw)",
            "value": df[["z_sensor14", "RUL_raw"]].corr().iloc[0, 1],
        },
    ]
    summary = pd.DataFrame(rows)

    var_rows = []
    for sensor in ["sensor9", "sensor14"]:
        between, within = variance_decomp(sensor)
        var_rows.append({"sensor": sensor, "component": "between_condition", "share": between})
        var_rows.append({"sensor": sensor, "component": "within_condition", "share": within})
    variance = pd.DataFrame(var_rows)

    condition_corr = (
        df.groupby("condition")
        .apply(lambda g: g[["sensor9", "sensor14"]].corr().iloc[0, 1])
        .rename("corr_sensor9_sensor14")
        .reset_index()
        .sort_values("corr_sensor9_sensor14", ascending=False)
    )
    return summary, variance, condition_corr


def cluster_late_degradation(df_z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    late = df_z[df_z["relative_cycle"] >= 0.8]
    unit_features = late.groupby("unit")[[f"z_{s}" for s in SENSORS]].mean()
    x = StandardScaler().fit_transform(unit_features)
    labels = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(x)
    pcs = PCA(n_components=2, random_state=42).fit_transform(x)
    out = pd.DataFrame(
        {
            "unit": unit_features.index,
            "proxy_cluster": labels,
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
        }
    )
    life = df_z.groupby("unit")["cycle"].max().rename("life").reset_index()
    out = out.merge(life, on="unit")

    centroids = unit_features.copy()
    centroids["proxy_cluster"] = labels
    cluster_profile = centroids.groupby("proxy_cluster").mean()
    cluster_profile.columns = [c.replace("z_", "") for c in cluster_profile.columns]
    return out, cluster_profile


def make_figures() -> dict[str, str]:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    figs: dict[str, str] = {}

    overview, life_df = build_overview()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.barplot(data=overview, x="set", y="train_rows", ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Train rows")
    sns.barplot(data=overview, x="set", y="train_units", ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("Train units")
    sns.barplot(data=overview, x="set", y="rounded_conditions", ax=axes[1, 0], color="#DD8452")
    axes[1, 0].set_title("Rounded operating conditions")
    sns.barplot(data=overview, x="set", y="constant_sensor_count", ax=axes[1, 1], color="#8172B2")
    axes[1, 1].set_title("Constant sensor count")
    fig.tight_layout()
    figs["overview"] = savefig(fig, "01_fd001_vs_fd004_overview.png").name

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    sns.boxplot(data=life_df, x="set", y="life", ax=axes[0])
    axes[0].set_title("Train total life")
    sns.histplot(data=life_df, x="life", hue="set", stat="probability", common_norm=False, element="step", fill=False, bins=30, ax=axes[1])
    axes[1].set_title("Normalized life histogram")
    sns.ecdfplot(data=life_df, x="life", hue="set", ax=axes[2])
    axes[2].set_title("Life ECDF")
    fig.tight_layout()
    figs["life"] = savefig(fig, "02_life_distribution_fd001_vs_fd004.png").name

    fd004 = add_condition_label(add_rul(load_train("FD004")))
    condition_order = fd004.groupby("condition").size().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(data=fd004, y="condition", order=condition_order, ax=axes[0], color="#4C72B0")
    axes[0].set_title("FD004 row count by condition")
    per_unit = fd004.groupby(["unit", "condition"]).size().reset_index(name="rows")
    sns.boxplot(data=per_unit, x="rows", y="condition", order=condition_order, ax=axes[1], color="#D6EAF8", fliersize=1)
    axes[1].set_title("Condition rows per unit")
    cond_per_unit = fd004.groupby("unit")["condition"].nunique().reset_index(name="condition_count")
    sns.histplot(data=cond_per_unit, x="condition_count", discrete=True, ax=axes[2], color="#55A868")
    axes[2].set_title("Number of conditions observed per unit")
    fig.tight_layout()
    figs["condition_counts"] = savefig(fig, "03_fd004_condition_counts.png").name

    fig, axes = plt.subplots(1, 2, figsize=(18, 4.8))
    for ax, fd in zip(axes, FD_LIST):
        tr = load_train(fd)
        corr = tr[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, linewidths=0.2)
        ax.set_title(f"{fd}: corr(setting, sensor)")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    figs["setting_sensor_corr"] = savefig(fig, "04_setting_sensor_corr_fd001_vs_fd004.png").name

    corr_table, _ = fd004_corr_tables()
    top_condition_sensors = corr_table.sort_values("max_abs_setting_corr", ascending=False).head(6)["sensor"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    for ax, sensor in zip(axes.ravel(), top_condition_sensors):
        sns.boxplot(data=fd004, x="condition", y=sensor, order=condition_order, ax=ax, color="#D6EAF8", fliersize=0.5)
        sample = fd004[["condition", sensor]].sample(min(len(fd004), 2500), random_state=42)
        sns.stripplot(data=sample, x="condition", y=sensor, order=condition_order, ax=ax, color="#2E4053", size=1.2, alpha=0.18)
        ax.set_title(f"{sensor} by condition")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.suptitle("FD004 sensors most affected by operating condition", y=1.02)
    fig.tight_layout()
    figs["condition_boxplots"] = savefig(fig, "05_fd004_condition_sensor_boxplots.png").name

    bar = corr_table.sort_values("cond_z_abs_corr_RUL", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].barh(bar["sensor"], bar["raw_abs_corr_RUL"], color="#8E9A9A")
    axes[0].set_title("Raw |corr(sensor, RUL)|")
    axes[1].barh(bar["sensor"], bar["cond_z_abs_corr_RUL"], color="#4C72B0")
    axes[1].set_title("Condition-normalized |corr(z_sensor, RUL)|")
    change = corr_table.sort_values("abs_corr_change", ascending=True)
    axes[2].barh(change["sensor"], change["abs_corr_change"], color=np.where(change["abs_corr_change"] >= 0, "#55A868", "#C44E52"))
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title("Change after condition normalization")
    fig.tight_layout()
    figs["corr_compare"] = savefig(fig, "06_fd004_raw_vs_condition_normalized_corr.png").name

    df_z = add_condition_zscores(add_rul(load_train("FD004")))
    top_rul_sensors = corr_table.head(6)["sensor"].tolist()
    bins = np.linspace(0, 1, 41)
    trend = df_z.copy()
    trend["rel_bin"] = pd.cut(trend["relative_cycle"], bins=bins, include_lowest=True, labels=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True)
    for ax, sensor in zip(axes.ravel(), top_rul_sensors):
        raw_agg = trend.groupby("rel_bin")[sensor].mean().reset_index()
        z_agg = trend.groupby("rel_bin")[f"z_{sensor}"].mean().reset_index()
        raw_agg["relative_cycle"] = (raw_agg["rel_bin"] + 0.5) / 40
        z_agg["relative_cycle"] = (z_agg["rel_bin"] + 0.5) / 40
        ax2 = ax.twinx()
        ax.plot(raw_agg["relative_cycle"], raw_agg[sensor], marker="o", markersize=2.5, linewidth=1.1, color="#8E9A9A", label="raw mean")
        ax2.plot(z_agg["relative_cycle"], z_agg[f"z_{sensor}"], marker="o", markersize=2.5, linewidth=1.1, color="#C44E52", label="condition z mean")
        ax.set_title(sensor)
        ax.set_xlabel("relative_cycle")
        ax.set_ylabel("raw", color="#8E9A9A")
        ax2.set_ylabel("condition z", color="#C44E52")
    fig.suptitle("FD004 top RUL sensors: raw trend vs condition-normalized trend", y=1.02)
    fig.tight_layout()
    figs["trend_raw_vs_z"] = savefig(fig, "07_fd004_raw_vs_condition_z_trends.png").name

    selected_units = [1, 2, 3, 10, 20]
    condition_codes = {cond: i for i, cond in enumerate(condition_order)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    for ax, sensor in zip(axes.ravel(), top_rul_sensors):
        for unit in selected_units:
            g = df_z[df_z["unit"] == unit].sort_values("cycle")
            ax.plot(g["cycle"], g[f"z_{sensor}"], linewidth=0.65, alpha=0.55)
            ax.scatter(g["cycle"], g[f"z_{sensor}"], c=g["condition"].map(condition_codes), s=6, alpha=0.7, cmap="tab10", vmin=0, vmax=max(condition_codes.values()))
        ax.set_title(f"z_{sensor}")
        ax.set_xlabel("cycle")
        ax.set_ylabel("condition-normalized value")
    fig.suptitle("FD004 selected units: points colored by operating condition", y=1.02)
    fig.tight_layout()
    figs["selected_units"] = savefig(fig, "08_fd004_selected_unit_timeseries.png").name

    s914_summary, s914_variance, s914_condition_corr = sensor9_sensor14_tables()
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    sample = df_z.sample(min(len(df_z), 9000), random_state=42)
    sns.scatterplot(
        data=sample,
        x="sensor9",
        y="sensor14",
        hue="condition",
        s=9,
        alpha=0.45,
        linewidth=0,
        ax=axes[0, 0],
        legend=False,
    )
    axes[0, 0].set_title("Raw sensor9 vs sensor14 by condition")
    sns.scatterplot(
        data=sample,
        x="z_sensor9",
        y="z_sensor14",
        hue="relative_cycle",
        palette="viridis",
        s=9,
        alpha=0.45,
        linewidth=0,
        ax=axes[0, 1],
        legend=False,
    )
    axes[0, 1].set_title("Condition-normalized z_sensor9 vs z_sensor14")

    sns.barplot(data=s914_variance, x="sensor", y="share", hue="component", ax=axes[0, 2])
    axes[0, 2].set_title("Variance decomposition")
    axes[0, 2].set_ylim(0, 1.05)

    corr_bars = pd.DataFrame(
        [
            {"sensor": "sensor9", "space": "raw", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("FD004 corr(sensor9, RUL_raw)"), "value"].iloc[0]},
            {"sensor": "sensor14", "space": "raw", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("FD004 corr(sensor14, RUL_raw)"), "value"].iloc[0]},
            {"sensor": "sensor9", "space": "condition_z", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("FD004 corr(z_sensor9, RUL_raw)"), "value"].iloc[0]},
            {"sensor": "sensor14", "space": "condition_z", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("FD004 corr(z_sensor14, RUL_raw)"), "value"].iloc[0]},
        ]
    )
    sns.barplot(data=corr_bars, x="sensor", y="corr_RUL", hue="space", ax=axes[1, 0])
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("RUL correlation before/after condition normalization")

    sns.barplot(data=s914_condition_corr, x="corr_sensor9_sensor14", y="condition", ax=axes[1, 1], color="#4C72B0")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_title("Within-condition corr(sensor9, sensor14)")

    trend = df_z.copy()
    trend["rel_bin"] = pd.cut(trend["relative_cycle"], bins=np.linspace(0, 1, 41), include_lowest=True, labels=False)
    for sensor, color in [("z_sensor9", "#4C72B0"), ("z_sensor14", "#C44E52")]:
        agg = trend.groupby("rel_bin")[sensor].mean().reset_index()
        agg["relative_cycle"] = (agg["rel_bin"] + 0.5) / 40
        axes[1, 2].plot(agg["relative_cycle"], agg[sensor], marker="o", markersize=3, linewidth=1.3, label=sensor, color=color)
    axes[1, 2].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1, 2].set_title("Condition-normalized mean trend")
    axes[1, 2].set_xlabel("relative_cycle")
    axes[1, 2].legend()

    fig.suptitle("FD004 sensor9 and sensor14 relationship", y=1.02)
    fig.tight_layout()
    figs["sensor9_sensor14"] = savefig(fig, "09_fd004_sensor9_sensor14_analysis.png").name

    cluster_df, cluster_profile = cluster_late_degradation(df_z)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="proxy_cluster", size="life", sizes=(30, 180), palette="Set2", ax=axes[0])
    axes[0].set_title("Late-life condition-normalized sensor profile clusters")
    sns.boxplot(data=cluster_df, x="proxy_cluster", y="life", ax=axes[1], palette="Set2", hue="proxy_cluster", legend=False)
    axes[1].set_title("Life distribution by proxy cluster")
    profile_diff = (cluster_profile.loc[1] - cluster_profile.loc[0]).sort_values()
    axes[2].barh(profile_diff.index, profile_diff.values, color=np.where(profile_diff.values >= 0, "#4C72B0", "#C44E52"))
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title("Cluster 1 - Cluster 0 late-life profile")
    fig.tight_layout()
    figs["proxy_clusters"] = savefig(fig, "10_fd004_proxy_fault_mode_clusters.png").name

    test = load_test("FD004")
    test_rul = load_rul("FD004")
    test_last = test.groupby("unit")["cycle"].max().rename("last_observed_cycle").reset_index()
    test_check = test_last.merge(test_rul, on="unit")
    test_check["implied_failure_cycle"] = test_check["last_observed_cycle"] + test_check["true_RUL"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    sns.histplot(test_check["last_observed_cycle"], bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set_title("FD004 test last observed cycle")
    sns.histplot(test_check["true_RUL"], bins=30, ax=axes[1], color="#55A868")
    axes[1].set_title("FD004 test true RUL")
    sns.scatterplot(data=test_check, x="last_observed_cycle", y="true_RUL", ax=axes[2], s=25, alpha=0.75)
    axes[2].set_title("Test truncation vs true RUL")
    fig.tight_layout()
    figs["test_rul"] = savefig(fig, "11_fd004_test_rul.png").name

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
    overview, _ = build_overview()
    corr_table, _ = fd004_corr_tables()
    top_corr = corr_table[
        ["sensor", "max_abs_setting_corr", "raw_corr_RUL", "cond_z_corr_RUL", "abs_corr_change"]
    ].head(12)
    s914_summary, s914_variance, s914_condition_corr = sensor9_sensor14_tables()

    def fig_md(key: str, title: str) -> str:
        return f"![{title}](figures/fd004_deep_dive/{figs[key]})\n\n"

    report = f"""# Leo FD004 Deep Dive EDA

## 목적

FD001 중심 EDA에서 벗어나 FD004가 왜 더 복잡한지 확인한다. 특히 FD004는 운전조건 6개와 고장모드 2개가 섞인 데이터셋이므로, setting 효과와 열화/RUL 신호를 분리해서 봐야 한다.

## 핵심 결론

- FD004는 FD001보다 row 수와 unit 수가 훨씬 크다.
- FD004는 rounded operating condition이 6개이고, FD001은 1개다.
- FD001에서 상수였던 센서들이 FD004에서는 condition에 따라 값이 달라져 상수가 아니다.
- FD004에서 sensor 분포는 setting의 영향을 강하게 받는다.
- FD004 모델링에서는 raw sensor만 넣기보다 condition-aware scaling 또는 condition별 normalization을 반드시 비교해야 한다.
- FD004의 고장모드 2개는 raw label로 제공되지 않는다. late-life condition-normalized sensor profile clustering은 가능한 proxy 분석일 뿐, 실제 fault mode label은 아니다.

## FD001 vs FD004 구조 비교

{df_to_markdown(overview.round(2))}

{fig_md("overview", "FD001 vs FD004 Overview")}
{fig_md("life", "FD001 vs FD004 Life Distribution")}

## FD004 운전조건 구조

{fig_md("condition_counts", "FD004 Condition Counts")}

## Setting-Sensor 관계

{fig_md("setting_sensor_corr", "Setting Sensor Correlation")}
{fig_md("condition_boxplots", "FD004 Condition Sensor Boxplots")}

## Raw Sensor vs Condition-Normalized Sensor

{df_to_markdown(top_corr.round(4))}

{fig_md("corr_compare", "Raw vs Condition-Normalized RUL Correlation")}
{fig_md("trend_raw_vs_z", "Raw Trend vs Condition Z Trend")}
{fig_md("selected_units", "Selected Unit Time Series")}

## Sensor 9와 Sensor 14 추가 확인

Sensor 9와 Sensor 14는 raw 그래프에서 편차가 크고 서로 높은 상관을 보인다. FD004에서는 두 센서의 전체 분산 대부분이 운전조건 차이에서 발생하지만, 같은 condition 안에서도 두 센서의 상관은 높게 유지된다. 따라서 두 센서는 단순한 노이즈가 아니라 같은 엔진 동역학에 함께 반응하는 중복성 높은 정보로 해석할 수 있다.

{df_to_markdown(s914_summary.round(4))}

{df_to_markdown(s914_variance.round(4))}

{df_to_markdown(s914_condition_corr.round(4))}

{fig_md("sensor9_sensor14", "FD004 Sensor 9 and Sensor 14 Analysis")}

## 고장모드 2개에 대한 Proxy 확인

FD004의 fault mode는 raw column으로 주어지지 않는다. 아래 clustering은 late-life condition-normalized sensor profile을 2개 군집으로 나눈 proxy 분석이다. 실제 fault mode label로 해석하면 안 되고, 모델링 전 데이터 이질성 확인용으로만 사용한다.

{fig_md("proxy_clusters", "Proxy Fault Mode Clusters")}

## Test Split 확인

{fig_md("test_rul", "FD004 Test RUL")}

## 전처리 제안

1. FD004에서는 FD001 상수 센서 제거 목록을 그대로 적용하지 않는다.
2. `condition = rounded(setting1, setting2, setting3)`를 생성한다.
3. raw scaling과 condition-aware scaling을 모두 실험한다.
4. condition-normalized sensor에 rolling mean/std/slope/delta feature를 추가한다.
5. fault mode label은 없으므로, proxy cluster를 모델 feature로 바로 쓰기보다 ablation 또는 diagnostic으로만 사용한다.
6. train/test leakage 방지를 위해 `relative_cycle`과 `total_life`는 EDA용으로만 사용한다.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def build_notebook() -> None:
    setup = r"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=0.9)

for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "data" / "raw").exists():
        ROOT = candidate
        break
else:
    raise FileNotFoundError("Cannot find data/raw")

RAW = ROOT / "data" / "raw"
FD_LIST = ["FD001", "FD004"]
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

def add_condition_zscores(df):
    out = add_condition_label(df)
    for sensor in SENSORS:
        mean = out.groupby("condition")[sensor].transform("mean")
        std = out.groupby("condition")[sensor].transform("std").replace(0, np.nan)
        out[f"z_{sensor}"] = ((out[sensor] - mean) / std).fillna(0)
    return out
""".strip()

    overview = r"""
rows = []
life_rows = []
for fd in FD_LIST:
    train = load_train(fd)
    test = load_test(fd)
    rul = load_rul(fd)
    life = train.groupby("unit")["cycle"].max()
    constant = train[SENSORS].nunique().loc[lambda s: s <= 1].index.tolist()
    rows.append({
        "set": fd,
        "train_rows": len(train),
        "train_units": train["unit"].nunique(),
        "test_rows": len(test),
        "test_units": test["unit"].nunique(),
        "life_min": life.min(),
        "life_median": life.median(),
        "life_mean": life.mean(),
        "life_max": life.max(),
        "rounded_conditions": len(rounded_settings(train).drop_duplicates()),
        "constant_sensor_count": len(constant),
        "constant_sensors": ", ".join(constant) if constant else "(none)",
        "test_RUL_min": rul["true_RUL"].min(),
        "test_RUL_max": rul["true_RUL"].max(),
    })
    life_rows.extend({"set": fd, "unit": unit, "life": value} for unit, value in life.items())

overview = pd.DataFrame(rows)
life_df = pd.DataFrame(life_rows)
display(overview.round(2))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sns.barplot(data=overview, x="set", y="train_rows", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Train rows")
sns.barplot(data=overview, x="set", y="train_units", ax=axes[0, 1], color="#55A868")
axes[0, 1].set_title("Train units")
sns.barplot(data=overview, x="set", y="rounded_conditions", ax=axes[1, 0], color="#DD8452")
axes[1, 0].set_title("Rounded operating conditions")
sns.barplot(data=overview, x="set", y="constant_sensor_count", ax=axes[1, 1], color="#8172B2")
axes[1, 1].set_title("Constant sensor count")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
sns.boxplot(data=life_df, x="set", y="life", ax=axes[0])
axes[0].set_title("Train total life")
sns.histplot(data=life_df, x="life", hue="set", stat="probability", common_norm=False, element="step", fill=False, bins=30, ax=axes[1])
axes[1].set_title("Normalized life histogram")
sns.ecdfplot(data=life_df, x="life", hue="set", ax=axes[2])
axes[2].set_title("Life ECDF")
plt.tight_layout()
plt.show()
""".strip()

    conditions = r"""
fd004 = add_condition_label(add_rul(load_train("FD004")))
condition_order = fd004.groupby("condition").size().sort_values(ascending=False).index.tolist()
display(fd004.groupby("condition").size().loc[condition_order].reset_index(name="row_count"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(data=fd004, y="condition", order=condition_order, ax=axes[0], color="#4C72B0")
axes[0].set_title("FD004 row count by condition")
per_unit = fd004.groupby(["unit", "condition"]).size().reset_index(name="rows")
sns.boxplot(data=per_unit, x="rows", y="condition", order=condition_order, ax=axes[1], color="#D6EAF8", fliersize=1)
axes[1].set_title("Condition rows per unit")
cond_per_unit = fd004.groupby("unit")["condition"].nunique().reset_index(name="condition_count")
sns.histplot(data=cond_per_unit, x="condition_count", discrete=True, ax=axes[2], color="#55A868")
axes[2].set_title("Number of conditions observed per unit")
plt.tight_layout()
plt.show()
""".strip()

    setting_sensor = r"""
fig, axes = plt.subplots(1, 2, figsize=(18, 4.8))
for ax, fd in zip(axes, FD_LIST):
    tr = load_train(fd)
    corr = tr[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, linewidths=0.2)
    ax.set_title(f"{fd}: corr(setting, sensor)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.show()

setting_corr = fd004[SETTINGS + SENSORS].corr().loc[SETTINGS, SENSORS]
top_condition_sensors = setting_corr.abs().max(axis=0).sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
for ax, sensor in zip(axes.ravel(), top_condition_sensors):
    sns.boxplot(data=fd004, x="condition", y=sensor, order=condition_order, ax=ax, color="#D6EAF8", fliersize=0.5)
    sample = fd004[["condition", sensor]].sample(min(len(fd004), 2500), random_state=42)
    sns.stripplot(data=sample, x="condition", y=sensor, order=condition_order, ax=ax, color="#2E4053", size=1.2, alpha=0.18)
    ax.set_title(f"{sensor} by condition")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
plt.tight_layout()
plt.show()
""".strip()

    normalization = r"""
df_z = add_condition_zscores(add_rul(load_train("FD004")))
corr_rows = []
for sensor in SENSORS:
    raw_corr = df_z[[sensor, "RUL_raw"]].corr().iloc[0, 1]
    z_corr = df_z[[f"z_{sensor}", "RUL_raw"]].corr().iloc[0, 1]
    setting_abs = df_z[SETTINGS + [sensor]].corr().loc[SETTINGS, sensor].abs().max()
    corr_rows.append({
        "sensor": sensor,
        "max_abs_setting_corr": setting_abs,
        "raw_corr_RUL": raw_corr,
        "raw_abs_corr_RUL": abs(raw_corr),
        "cond_z_corr_RUL": z_corr,
        "cond_z_abs_corr_RUL": abs(z_corr),
        "abs_corr_change": abs(z_corr) - abs(raw_corr),
    })
corr_table = pd.DataFrame(corr_rows).sort_values("cond_z_abs_corr_RUL", ascending=False)
display(corr_table.round(4))

bar = corr_table.sort_values("cond_z_abs_corr_RUL", ascending=True)
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].barh(bar["sensor"], bar["raw_abs_corr_RUL"], color="#8E9A9A")
axes[0].set_title("Raw |corr(sensor, RUL)|")
axes[1].barh(bar["sensor"], bar["cond_z_abs_corr_RUL"], color="#4C72B0")
axes[1].set_title("Condition-normalized |corr(z_sensor, RUL)|")
change = corr_table.sort_values("abs_corr_change", ascending=True)
axes[2].barh(change["sensor"], change["abs_corr_change"], color=np.where(change["abs_corr_change"] >= 0, "#55A868", "#C44E52"))
axes[2].axvline(0, color="black", linewidth=1)
axes[2].set_title("Change after condition normalization")
plt.tight_layout()
plt.show()
""".strip()

    trends = r"""
top_rul_sensors = corr_table.head(6)["sensor"].tolist()
bins = np.linspace(0, 1, 41)
trend = df_z.copy()
trend["rel_bin"] = pd.cut(trend["relative_cycle"], bins=bins, include_lowest=True, labels=False)
fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True)
for ax, sensor in zip(axes.ravel(), top_rul_sensors):
    raw_agg = trend.groupby("rel_bin")[sensor].mean().reset_index()
    z_agg = trend.groupby("rel_bin")[f"z_{sensor}"].mean().reset_index()
    raw_agg["relative_cycle"] = (raw_agg["rel_bin"] + 0.5) / 40
    z_agg["relative_cycle"] = (z_agg["rel_bin"] + 0.5) / 40
    ax2 = ax.twinx()
    ax.plot(raw_agg["relative_cycle"], raw_agg[sensor], marker="o", markersize=2.5, linewidth=1.1, color="#8E9A9A")
    ax2.plot(z_agg["relative_cycle"], z_agg[f"z_{sensor}"], marker="o", markersize=2.5, linewidth=1.1, color="#C44E52")
    ax.set_title(sensor)
    ax.set_xlabel("relative_cycle")
    ax.set_ylabel("raw", color="#8E9A9A")
    ax2.set_ylabel("condition z", color="#C44E52")
plt.tight_layout()
plt.show()

selected_units = [1, 2, 3, 10, 20]
condition_codes = {cond: i for i, cond in enumerate(condition_order)}
fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
for ax, sensor in zip(axes.ravel(), top_rul_sensors):
    for unit in selected_units:
        g = df_z[df_z["unit"] == unit].sort_values("cycle")
        ax.plot(g["cycle"], g[f"z_{sensor}"], linewidth=0.65, alpha=0.55)
        ax.scatter(g["cycle"], g[f"z_{sensor}"], c=g["condition"].map(condition_codes), s=6, alpha=0.7, cmap="tab10", vmin=0, vmax=max(condition_codes.values()))
    ax.set_title(f"z_{sensor}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("condition-normalized value")
plt.tight_layout()
plt.show()
""".strip()

    sensor9_14 = r"""
def variance_decomp(sensor):
    overall = df_z[sensor].mean()
    total_ss = ((df_z[sensor] - overall) ** 2).sum()
    condition_mean = df_z.groupby("condition")[sensor].transform("mean")
    between_ss = ((condition_mean - overall) ** 2).sum()
    within_ss = ((df_z[sensor] - condition_mean) ** 2).sum()
    return between_ss / total_ss, within_ss / total_ss

def partial_corr_by_settings(a, b):
    x = df_z[SETTINGS].to_numpy()
    ya = df_z[a].to_numpy()
    yb = df_z[b].to_numpy()
    ra = ya - LinearRegression().fit(x, ya).predict(x)
    rb = yb - LinearRegression().fit(x, yb).predict(x)
    return np.corrcoef(ra, rb)[0, 1]

s914_summary = pd.DataFrame([
    {"metric": "raw corr(sensor9, sensor14)", "value": df_z[["sensor9", "sensor14"]].corr().iloc[0, 1]},
    {"metric": "condition-z corr(z_sensor9, z_sensor14)", "value": df_z[["z_sensor9", "z_sensor14"]].corr().iloc[0, 1]},
    {"metric": "partial corr controlling numeric settings", "value": partial_corr_by_settings("sensor9", "sensor14")},
    {"metric": "corr(sensor9, RUL_raw)", "value": df_z[["sensor9", "RUL_raw"]].corr().iloc[0, 1]},
    {"metric": "corr(sensor14, RUL_raw)", "value": df_z[["sensor14", "RUL_raw"]].corr().iloc[0, 1]},
    {"metric": "corr(z_sensor9, RUL_raw)", "value": df_z[["z_sensor9", "RUL_raw"]].corr().iloc[0, 1]},
    {"metric": "corr(z_sensor14, RUL_raw)", "value": df_z[["z_sensor14", "RUL_raw"]].corr().iloc[0, 1]},
])

variance_rows = []
for sensor in ["sensor9", "sensor14"]:
    between, within = variance_decomp(sensor)
    variance_rows.append({"sensor": sensor, "component": "between_condition", "share": between})
    variance_rows.append({"sensor": sensor, "component": "within_condition", "share": within})
s914_variance = pd.DataFrame(variance_rows)

s914_condition_corr = (
    df_z.groupby("condition")
    .apply(lambda g: g[["sensor9", "sensor14"]].corr().iloc[0, 1])
    .rename("corr_sensor9_sensor14")
    .reset_index()
    .sort_values("corr_sensor9_sensor14", ascending=False)
)

display(s914_summary.round(4))
display(s914_variance.round(4))
display(s914_condition_corr.round(4))

fig, axes = plt.subplots(2, 3, figsize=(21, 11))
sample = df_z.sample(min(len(df_z), 9000), random_state=42)
sns.scatterplot(data=sample, x="sensor9", y="sensor14", hue="condition", s=9, alpha=0.45, linewidth=0, ax=axes[0, 0], legend=False)
axes[0, 0].set_title("Raw sensor9 vs sensor14 by condition")
sns.scatterplot(data=sample, x="z_sensor9", y="z_sensor14", hue="relative_cycle", palette="viridis", s=9, alpha=0.45, linewidth=0, ax=axes[0, 1], legend=False)
axes[0, 1].set_title("Condition-normalized z_sensor9 vs z_sensor14")
sns.barplot(data=s914_variance, x="sensor", y="share", hue="component", ax=axes[0, 2])
axes[0, 2].set_title("Variance decomposition")
axes[0, 2].set_ylim(0, 1.05)

corr_bars = pd.DataFrame([
    {"sensor": "sensor9", "space": "raw", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("corr(sensor9, RUL_raw)"), "value"].iloc[0]},
    {"sensor": "sensor14", "space": "raw", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("corr(sensor14, RUL_raw)"), "value"].iloc[0]},
    {"sensor": "sensor9", "space": "condition_z", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("corr(z_sensor9, RUL_raw)"), "value"].iloc[0]},
    {"sensor": "sensor14", "space": "condition_z", "corr_RUL": s914_summary.loc[s914_summary["metric"].eq("corr(z_sensor14, RUL_raw)"), "value"].iloc[0]},
])
sns.barplot(data=corr_bars, x="sensor", y="corr_RUL", hue="space", ax=axes[1, 0])
axes[1, 0].axhline(0, color="black", linewidth=1)
axes[1, 0].set_title("RUL correlation before/after condition normalization")
sns.barplot(data=s914_condition_corr, x="corr_sensor9_sensor14", y="condition", ax=axes[1, 1], color="#4C72B0")
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_title("Within-condition corr(sensor9, sensor14)")

trend_s914 = df_z.copy()
trend_s914["rel_bin"] = pd.cut(trend_s914["relative_cycle"], bins=np.linspace(0, 1, 41), include_lowest=True, labels=False)
for sensor, color in [("z_sensor9", "#4C72B0"), ("z_sensor14", "#C44E52")]:
    agg = trend_s914.groupby("rel_bin")[sensor].mean().reset_index()
    agg["relative_cycle"] = (agg["rel_bin"] + 0.5) / 40
    axes[1, 2].plot(agg["relative_cycle"], agg[sensor], marker="o", markersize=3, linewidth=1.3, label=sensor, color=color)
axes[1, 2].axhline(0, color="black", linewidth=1, alpha=0.5)
axes[1, 2].set_title("Condition-normalized mean trend")
axes[1, 2].set_xlabel("relative_cycle")
axes[1, 2].legend()
plt.tight_layout()
plt.show()
""".strip()

    clusters = r"""
late = df_z[df_z["relative_cycle"] >= 0.8]
unit_features = late.groupby("unit")[[f"z_{s}" for s in SENSORS]].mean()
x = StandardScaler().fit_transform(unit_features)
labels = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(x)
pcs = PCA(n_components=2, random_state=42).fit_transform(x)
cluster_df = pd.DataFrame({"unit": unit_features.index, "proxy_cluster": labels, "PC1": pcs[:, 0], "PC2": pcs[:, 1]})
cluster_df = cluster_df.merge(df_z.groupby("unit")["cycle"].max().rename("life").reset_index(), on="unit")
display(cluster_df.groupby("proxy_cluster")["life"].describe().round(2))

centroids = unit_features.copy()
centroids["proxy_cluster"] = labels
cluster_profile = centroids.groupby("proxy_cluster").mean()
cluster_profile.columns = [c.replace("z_", "") for c in cluster_profile.columns]

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="proxy_cluster", size="life", sizes=(30, 180), palette="Set2", ax=axes[0])
axes[0].set_title("Late-life condition-normalized profile clusters")
sns.boxplot(data=cluster_df, x="proxy_cluster", y="life", ax=axes[1], palette="Set2", hue="proxy_cluster", legend=False)
axes[1].set_title("Life distribution by proxy cluster")
profile_diff = (cluster_profile.loc[1] - cluster_profile.loc[0]).sort_values()
axes[2].barh(profile_diff.index, profile_diff.values, color=np.where(profile_diff.values >= 0, "#4C72B0", "#C44E52"))
axes[2].axvline(0, color="black", linewidth=1)
axes[2].set_title("Cluster 1 - Cluster 0 late-life profile")
plt.tight_layout()
plt.show()
""".strip()

    test = r"""
test = load_test("FD004")
test_rul = load_rul("FD004")
test_last = test.groupby("unit")["cycle"].max().rename("last_observed_cycle").reset_index()
test_check = test_last.merge(test_rul, on="unit")
test_check["implied_failure_cycle"] = test_check["last_observed_cycle"] + test_check["true_RUL"]
display(test_check.describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
sns.histplot(test_check["last_observed_cycle"], bins=30, ax=axes[0], color="#4C72B0")
axes[0].set_title("FD004 test last observed cycle")
sns.histplot(test_check["true_RUL"], bins=30, ax=axes[1], color="#55A868")
axes[1].set_title("FD004 test true RUL")
sns.scatterplot(data=test_check, x="last_observed_cycle", y="true_RUL", ax=axes[2], s=25, alpha=0.75)
axes[2].set_title("Test truncation vs true RUL")
plt.tight_layout()
plt.show()
""".strip()

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# FD004 Deep Dive EDA\n\nFD004는 운전조건 6개와 고장모드 2개가 섞인 데이터셋이다. FD001과 분리해서 setting 효과와 RUL 신호를 확인한다."),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## 1. FD001 vs FD004 구조 비교"),
        nbf.v4.new_code_cell(overview),
        nbf.v4.new_markdown_cell("## 2. FD004 운전조건 분포"),
        nbf.v4.new_code_cell(conditions),
        nbf.v4.new_markdown_cell("## 3. Setting-Sensor 관계"),
        nbf.v4.new_code_cell(setting_sensor),
        nbf.v4.new_markdown_cell("## 4. Raw Sensor vs Condition-Normalized Sensor"),
        nbf.v4.new_code_cell(normalization),
        nbf.v4.new_markdown_cell("## 5. FD004 Trend와 Selected Unit 시계열"),
        nbf.v4.new_code_cell(trends),
        nbf.v4.new_markdown_cell("## 6. Sensor 9와 Sensor 14 관계 추가 확인\n\n두 센서의 큰 편차가 운전조건 때문인지, condition 안에서도 함께 움직이는지 확인한다."),
        nbf.v4.new_code_cell(sensor9_14),
        nbf.v4.new_markdown_cell("## 7. 고장모드 2개에 대한 Proxy Clustering\n\n실제 fault mode label은 raw에 없으므로, 이 결과는 데이터 이질성 확인용 proxy로만 사용한다."),
        nbf.v4.new_code_cell(clusters),
        nbf.v4.new_markdown_cell("## 8. FD004 Test Split 확인"),
        nbf.v4.new_code_cell(test),
        nbf.v4.new_markdown_cell(
            "## 전처리 결론\n\n"
            "- FD004에는 FD001 상수 센서 제거 목록을 그대로 적용하지 않는다.\n"
            "- `condition = rounded(setting1, setting2, setting3)`를 만들고 condition-aware scaling을 비교한다.\n"
            "- raw sensor, condition-normalized sensor, rolling mean/std/slope/delta feature를 ablation으로 비교한다.\n"
            "- fault mode label은 없으므로 proxy cluster는 diagnostic으로만 사용한다.\n"
            "- `relative_cycle`, `total_life`는 EDA용이며 모델 feature로 쓰면 leakage다."
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
