"""Shared RUL prediction and maintenance scheduling utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]


# ---------------- Data and RUL prediction ----------------
def load_train(RAW, fd="FD001"):
    return pd.read_csv(Path(RAW) / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)


def load_test(RAW, fd="FD001"):
    test = pd.read_csv(Path(RAW) / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    true_rul = pd.read_csv(Path(RAW) / f"RUL_{fd}.txt", header=None, names=["RUL"])["RUL"].values
    return test, true_rul


def add_rul(df, clip=125):
    life = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (life - df["cycle"]).clip(upper=clip)
    return df


def valid_sensors(df):
    return [s for s in SENSORS if df[s].nunique() > 1]


def build_rul_features(df, sensors=None, windows=(10, 30), include_settings=True):
    """Build row-level degradation features for train or test trajectories.

    The returned feature list can be reused across train/test as long as the
    same sensors and windows are supplied.
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    sensors = list(sensors or valid_sensors(df))

    features = ["cycle"] + sensors
    if include_settings:
        features += ["set1", "set2", "set3"]

    frames = [df]
    group = df.groupby("unit", group_keys=False)
    diff = group[sensors].diff().fillna(0.0)

    for window in windows:
        rolled = group[sensors].rolling(window, min_periods=1)
        mean = rolled.mean().reset_index(level=0, drop=True).add_suffix(f"_ma{window}")
        std = (
            rolled.std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .add_suffix(f"_sd{window}")
        )
        dmean = (
            diff.groupby(df["unit"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .add_suffix(f"_dmean{window}")
        )
        frames.extend([mean, std, dmean])
        features.extend(mean.columns)
        features.extend(std.columns)
        features.extend(dmean.columns)

    return pd.concat(frames, axis=1), list(features)


def phm08_score(y_true, y_pred):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    score = np.where(err < 0, np.exp(-err / 13.0) - 1, np.exp(err / 10.0) - 1)
    return float(score.sum())


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "PHM08": phm08_score(y_true, y_pred),
    }


def _make_rul_model(model="lgbm", seed=42):
    model = model.lower()
    if model == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if model == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if model == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            random_state=seed,
        )
    if model == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )

    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    except Exception:
        return HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            random_state=seed,
        )


def evaluate_rul_models(RAW, fd="FD001", clip=125, windows=(10, 30), seed=42):
    """Engine-group validation on train trajectories."""
    train = add_rul(load_train(RAW, fd), clip)
    sensors = valid_sensors(train)
    feat_df, features = build_rul_features(train, sensors=sensors, windows=windows)
    X, y, groups = feat_df[features], feat_df["RUL"], feat_df["unit"]
    tr, va = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            X, y, groups=groups
        )
    )

    rows = []
    for name in ["rf", "extra_trees", "hgb", "lgbm", "xgb"]:
        model = _make_rul_model(name, seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = np.clip(model.predict(X.iloc[va]), 0, clip)
        row = {"model": name, **regression_metrics(y.iloc[va], pred)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def fleet_rul(RAW, fd="FD001", clip=125, seed=42, model="lgbm", windows=(10, 30)):
    """Train on full train trajectories and predict final test-engine RUL."""
    train = add_rul(load_train(RAW, fd), clip)
    test, true = load_test(RAW, fd)
    sensors = valid_sensors(train)

    train_feat, features = build_rul_features(train, sensors=sensors, windows=windows)
    test_feat, _ = build_rul_features(test, sensors=sensors, windows=windows)

    reg = _make_rul_model(model, seed)
    reg.fit(train_feat[features], train_feat["RUL"])

    last = test_feat.groupby("unit").last().reset_index()
    pred = np.clip(reg.predict(last[features]), 0, clip)
    return pd.DataFrame(
        {"unit": last["unit"].values, "pred_RUL": pred, "true_RUL": true}
    )


# ---------------- Scheduling ----------------
def default_K(n_engines, T, util=1.3):
    """함대 크기에 맞춘 창당 정비팀 용량 제안.

    전체 처리량(T*K)이 엔진 수의 util 배가 되도록 K를 정해, 세트가 달라도
    자원 희소성 수준을 일정하게 유지한다(K 고정 시 큰 세트에서 포화되어 전략 차이가 사라짐).
    """
    return max(3, round(n_engines / T * util))


def make_deadlines(rul, cycles_per_window=15, safety_margin=0):
    """RUL cycles to maintenance deadline windows, with optional margin."""
    d = np.ceil(np.asarray(rul, dtype=float) / cycles_per_window).astype(int)
    d = np.maximum(d - int(safety_margin), 1)
    return d


def make_part_types(units, n_types=4):
    units = np.asarray(units, dtype=int)
    return (units - 1) % int(n_types)


def count_failures(schedule, deadline):
    schedule = np.asarray(schedule)
    deadline = np.asarray(deadline)
    ontime = (schedule > 0) & (schedule <= deadline)
    return int((~ontime).sum())


def wasted_life(schedule, deadline):
    schedule = np.asarray(schedule)
    deadline = np.asarray(deadline)
    ontime = (schedule > 0) & (schedule <= deadline)
    return float(np.where(ontime, np.maximum(0, deadline - schedule), 0).sum())


def setup_count(schedule, part_type=None):
    schedule = np.asarray(schedule)
    if part_type is None:
        return int((schedule > 0).sum())
    part_type = np.asarray(part_type)
    total = 0
    for t in sorted(np.unique(schedule[schedule > 0])):
        total += len(np.unique(part_type[schedule == t]))
    return int(total)


def schedule_metrics(
    schedule,
    deadline,
    part_type=None,
    c_pm=1.0,
    c_fail=10.0,
    w=0.2,
    setup_cost=0.0,
):
    schedule = np.asarray(schedule)
    deadline = np.asarray(deadline)
    ontime = (schedule > 0) & (schedule <= deadline)
    scheduled = schedule > 0
    failures = int((~ontime).sum())
    pm_count = int(ontime.sum())
    waste = wasted_life(schedule, deadline)
    setups = setup_count(np.where(scheduled, schedule, 0), part_type)
    total = c_pm * pm_count + c_fail * failures + w * waste + setup_cost * setups
    return {
        "pm_count": pm_count,
        "failures": failures,
        "wasted_life": waste,
        "setups": setups,
        "total_cost": float(total),
    }


def total_cost(
    schedule,
    deadline,
    c_pm=1,
    c_fail=5,
    w=0.1,
    part_type=None,
    setup_cost=0.0,
):
    return schedule_metrics(
        schedule,
        deadline,
        part_type=part_type,
        c_pm=c_pm,
        c_fail=c_fail,
        w=w,
        setup_cost=setup_cost,
    )["total_cost"]


def _repair(schedule, T, K):
    schedule = np.asarray(schedule).copy()
    schedule[(schedule < 0) | (schedule > T)] = 0
    for t in range(1, T + 1):
        idx = np.where(schedule == t)[0]
        if len(idx) > K:
            schedule[idx[K:]] = 0
    return schedule


def edf_schedule(deadline, T, K):
    """Earliest-deadline-first, assigning to the earliest feasible window."""
    n = len(deadline)
    schedule = np.zeros(n, dtype=int)
    load = np.zeros(T + 2, dtype=int)
    for i in np.argsort(deadline):
        d = min(int(deadline[i]), T)
        for t in range(1, d + 1):
            if load[t] < K:
                schedule[i] = t
                load[t] += 1
                break
    return schedule


def latest_deadline_schedule(deadline, T, K):
    """Deadline-feasible schedule that keeps maintenance as late as possible."""
    n = len(deadline)
    schedule = np.zeros(n, dtype=int)
    load = np.zeros(T + 2, dtype=int)
    for i in np.argsort(deadline):
        d = min(int(deadline[i]), T)
        for t in range(d, 0, -1):
            if load[t] < K:
                schedule[i] = t
                load[t] += 1
                break
    return schedule


def grouped_greedy_schedule(deadline, T, K, part_type):
    """Greedy baseline that fills each window with one part type when possible."""
    deadline = np.asarray(deadline)
    part_type = np.asarray(part_type)
    n = len(deadline)
    schedule = np.zeros(n, dtype=int)
    remaining = set(range(n))

    for t in range(1, T + 1):
        eligible = [i for i in remaining if deadline[i] >= t]
        if not eligible:
            continue
        counts = {}
        for i in eligible:
            counts[part_type[i]] = counts.get(part_type[i], 0) + 1
        preferred = sorted(counts, key=counts.get, reverse=True)
        chosen = []
        for p in preferred:
            pool = sorted(
                [i for i in eligible if part_type[i] == p and i in remaining],
                key=lambda i: deadline[i],
            )
            for i in pool:
                if len(chosen) >= K:
                    break
                chosen.append(i)
            if len(chosen) >= K:
                break
        for i in chosen:
            schedule[i] = t
            remaining.remove(i)
    return schedule


def random_schedule(deadline, T, K, seed=0):
    rng = np.random.default_rng(seed)
    n = len(deadline)
    schedule = np.zeros(n, dtype=int)
    load = np.zeros(T + 2, dtype=int)
    for i in rng.permutation(n):
        t = int(rng.integers(1, T + 1))
        if load[t] < K:
            schedule[i] = t
            load[t] += 1
    return schedule


def _neighbor(schedule, T, K, rng):
    cand = np.asarray(schedule).copy()
    n = len(cand)
    if rng.random() < 0.35:
        i, j = rng.choice(n, size=2, replace=False)
        cand[i], cand[j] = cand[j], cand[i]
        return cand

    i = int(rng.integers(n))
    cur_t = cand[i]
    load = np.bincount(cand, minlength=T + 1)
    feasible = [0]
    feasible.extend(t for t in range(1, T + 1) if load[t] < K or t == cur_t)
    cand[i] = int(rng.choice(feasible))
    return cand


def simulated_annealing(
    deadline,
    T,
    K,
    n_iter=3000,
    temp=5.0,
    cooling=0.997,
    seed=0,
    c_pm=1,
    c_fail=5,
    w=0.1,
    record=False,
    part_type=None,
    setup_cost=0.0,
    initial="best",
):
    """Local search over engine-to-window assignments."""
    rng = np.random.default_rng(seed)
    if initial == "edf":
        cur = edf_schedule(deadline, T, K)
    elif initial == "latest":
        cur = latest_deadline_schedule(deadline, T, K)
    elif initial == "grouped" and part_type is not None:
        cur = grouped_greedy_schedule(deadline, T, K, part_type)
    else:
        candidates = [edf_schedule(deadline, T, K), latest_deadline_schedule(deadline, T, K)]
        if part_type is not None:
            candidates.append(grouped_greedy_schedule(deadline, T, K, part_type))
        cur = min(
            candidates,
            key=lambda s: total_cost(s, deadline, c_pm, c_fail, w, part_type, setup_cost),
        )

    cur = _repair(cur, T, K)
    cur_cost = total_cost(cur, deadline, c_pm, c_fail, w, part_type, setup_cost)
    best = cur.copy()
    best_cost = cur_cost
    hist = [best_cost]
    tval = temp

    for _ in range(n_iter):
        cand = _neighbor(cur, T, K, rng)
        cand = _repair(cand, T, K)
        cc = total_cost(cand, deadline, c_pm, c_fail, w, part_type, setup_cost)
        delta = cc - cur_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(tval, 1e-9)):
            cur, cur_cost = cand, cc
            if cc < best_cost:
                best, best_cost = cand.copy(), cc
        tval *= cooling
        if record:
            hist.append(best_cost)
    return (best, best_cost, hist) if record else (best, best_cost)


def multistart_simulated_annealing(
    deadline,
    T,
    K,
    n_starts=3,
    n_iter=7000,
    temp=12.0,
    cooling=0.998,
    seed=42,
    c_pm=1,
    c_fail=18,
    w=0.4,
    part_type=None,
    setup_cost=5.0,
    record=False,
):
    """Run SA multiple times and keep a conservative low-cost candidate.

    Selection first keeps predicted failures low, then reduces setup count, then
    favors more deadline buffer as a hedge against RUL overestimation.
    """
    best_key = None
    best_schedule = None
    best_cost = None
    best_hist = None

    for offset in range(int(n_starts)):
        result = simulated_annealing(
            deadline,
            T,
            K,
            n_iter=n_iter,
            temp=temp,
            cooling=cooling,
            seed=seed + offset,
            c_pm=c_pm,
            c_fail=c_fail,
            w=w,
            record=record,
            part_type=part_type,
            setup_cost=setup_cost,
        )
        if record:
            schedule, cost, hist = result
        else:
            schedule, cost = result
            hist = None

        metrics = schedule_metrics(
            schedule,
            deadline,
            part_type,
            c_pm=c_pm,
            c_fail=c_fail,
            w=w,
            setup_cost=setup_cost,
        )
        key = (metrics["failures"], metrics["setups"], -metrics["wasted_life"], cost)
        if best_key is None or key < best_key:
            best_key = key
            best_schedule = schedule
            best_cost = cost
            best_hist = hist

    return (best_schedule, best_cost, best_hist) if record else (best_schedule, best_cost)


def replay(schedule, true_deadline):
    return count_failures(schedule, true_deadline)
