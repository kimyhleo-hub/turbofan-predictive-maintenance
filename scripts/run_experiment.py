"""Run the local predict-then-optimize experiment and save summary tables."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline as P


def repo_root() -> Path:
    return ROOT


def strategy_schedules(pred_deadline, true_deadline, T, K, part_type, args):
    sa_schedule = P.multistart_simulated_annealing(
        pred_deadline,
        T,
        K,
        n_starts=args.sa_restarts,
        n_iter=args.sa_iter,
        temp=args.sa_temp,
        cooling=args.sa_cooling,
        seed=args.seed,
        c_fail=args.c_fail,
        w=args.waste_weight,
        part_type=part_type,
        setup_cost=args.setup_cost,
    )[0]

    return {
        "Random": P.random_schedule(pred_deadline, T, K, seed=args.seed),
        "EDF": P.edf_schedule(pred_deadline, T, K),
        "LatestEDF": P.latest_deadline_schedule(pred_deadline, T, K),
        "GroupedGreedy": P.grouped_greedy_schedule(pred_deadline, T, K, part_type),
        "SA": sa_schedule,
        "Oracle": P.latest_deadline_schedule(true_deadline, T, K),
    }


def compare_strategies(pred_deadline, true_deadline, T, K, part_type, args):
    rows = []
    schedules = strategy_schedules(pred_deadline, true_deadline, T, K, part_type, args)
    for name, schedule in schedules.items():
        planned_deadline = true_deadline if name == "Oracle" else pred_deadline
        planned = P.schedule_metrics(
            schedule,
            planned_deadline,
            part_type,
            c_fail=args.c_fail,
            w=args.waste_weight,
            setup_cost=args.setup_cost,
        )
        actual = P.schedule_metrics(
            schedule,
            true_deadline,
            part_type,
            c_fail=args.c_fail,
            w=args.waste_weight,
            setup_cost=args.setup_cost,
        )
        rows.append(
            {
                "strategy": name,
                "planned_failures": planned["failures"],
                "planned_wasted_life": planned["wasted_life"],
                "planned_setups": planned["setups"],
                "planned_total_cost": round(planned["total_cost"], 3),
                "actual_failures": actual["failures"],
                "actual_wasted_life": actual["wasted_life"],
                "actual_setups": actual["setups"],
                "actual_total_cost": round(actual["total_cost"], 3),
            }
        )
    return pd.DataFrame(rows).sort_values("actual_total_cost").reset_index(drop=True)


def capacity_sensitivity(pred_deadline, true_deadline, T, part_type, args):
    rows = []
    for K in args.capacity_grid:
        schedules = strategy_schedules(pred_deadline, true_deadline, T, K, part_type, args)
        for name, schedule in schedules.items():
            actual = P.schedule_metrics(
                schedule,
                true_deadline,
                part_type,
                c_fail=args.c_fail,
                w=args.waste_weight,
                setup_cost=args.setup_cost,
            )
            rows.append(
                {
                    "K": K,
                    "strategy": name,
                    "actual_failures": actual["failures"],
                    "actual_total_cost": round(actual["total_cost"], 3),
                    "actual_setups": actual["setups"],
                    "actual_wasted_life": actual["wasted_life"],
                }
            )
    return pd.DataFrame(rows)


def safety_margin_sensitivity(fleet, true_deadline, part_type, args):
    rows = []
    for margin in args.margin_grid:
        pred_deadline = P.make_deadlines(
            fleet["pred_RUL"],
            cycles_per_window=args.cycles_per_window,
            safety_margin=margin,
        )
        T = int(max(pred_deadline.max(), true_deadline.max()))
        table = compare_strategies(pred_deadline, true_deadline, T, args.capacity, part_type, args)
        for _, row in table.iterrows():
            rows.append({"safety_margin": margin, **row.to_dict()})
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", default="FD001")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--cycles-per-window", type=int, default=15)
    parser.add_argument("--safety-margin", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=10)
    parser.add_argument("--part-types", type=int, default=4)
    parser.add_argument("--c-fail", type=float, default=18.0)
    parser.add_argument("--waste-weight", type=float, default=0.4)
    parser.add_argument("--setup-cost", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sa-iter", type=int, default=7000)
    parser.add_argument("--sa-restarts", type=int, default=3)
    parser.add_argument("--sa-temp", type=float, default=12.0)
    parser.add_argument("--sa-cooling", type=float, default=0.998)
    parser.add_argument("--capacity-grid", type=int, nargs="+", default=[6, 8, 10, 12, 15])
    parser.add_argument("--margin-grid", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    raw_dir = root / args.raw_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    fleet = P.fleet_rul(raw_dir, fd=args.fd, model=args.model, seed=args.seed)
    rul_metrics = P.regression_metrics(fleet["true_RUL"], fleet["pred_RUL"])

    pred_deadline = P.make_deadlines(
        fleet["pred_RUL"],
        cycles_per_window=args.cycles_per_window,
        safety_margin=args.safety_margin,
    )
    true_deadline = P.make_deadlines(
        fleet["true_RUL"], cycles_per_window=args.cycles_per_window
    )
    T = int(max(pred_deadline.max(), true_deadline.max()))
    part_type = P.make_part_types(fleet["unit"], args.part_types)

    comparison = compare_strategies(
        pred_deadline, true_deadline, T, args.capacity, part_type, args
    )
    capacity = capacity_sensitivity(pred_deadline, true_deadline, T, part_type, args)
    margins = safety_margin_sensitivity(fleet, true_deadline, part_type, args)

    fleet.to_csv(out_dir / "fleet_rul_predictions.csv", index=False)
    comparison.to_csv(out_dir / "strategy_comparison.csv", index=False)
    capacity.to_csv(out_dir / "capacity_sensitivity.csv", index=False)
    margins.to_csv(out_dir / "safety_margin_sensitivity.csv", index=False)

    elapsed = time.perf_counter() - start
    print("RUL metrics:", {k: round(v, 3) for k, v in rul_metrics.items()})
    print(f"Scenario: T={T}, K={args.capacity}, safety_margin={args.safety_margin}")
    print("\nStrategy comparison")
    print(comparison.to_string(index=False))
    print(f"\nSaved results to {out_dir}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
