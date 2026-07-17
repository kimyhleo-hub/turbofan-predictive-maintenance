"""메타휴리스틱 어댑터 템플릿 — 팀원용.

이 파일을 복사해서 scripts/algo_{알고리즘}.py 로 만들고,
`search()` 함수 안만 여러분의 알고리즘으로 채우면 됩니다.
나머지(문제 설정·평가·저장)는 전원 동일하게 고정되어 있습니다.

실행:  python scripts/algo_template.py
결과:  results/{ALGO_NAME}_results.json   ← 이 파일만 보내주세요

규격 상세: TODO.md 의 "공정 비교 규격" 참고
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import make_deadlines, make_part_types, total_cost, _repair, edf_schedule

# ═══════════════════════════════════════════════════════════════
# ▼▼▼ 여기만 수정하세요 ▼▼▼
# ═══════════════════════════════════════════════════════════════
ALGO_NAME = "TabuSearch"          # TabuSearch / GeneticAlgorithm / PSO / ACO
AUTHOR = "이름"
PARAMS = {                         # 여러분이 튜닝한 하이퍼파라미터 (기록용)
    "tenure": 20,
    "neighborhood_size": 30,
}


def search(problem, rng, budget):
    """여러분의 알고리즘. 예산(FEV) 안에서 최선의 스케줄을 찾으세요.

    Parameters
    ----------
    problem : Problem
        problem.cost(schedule) -> float   목적값 (호출 1회 = FEV 1)   ★ 이것만 쓰세요
        problem.n      : 707              엔진 수
        problem.T      : 10               창 수
        problem.K      : 75               창당 용량
        problem.repair(schedule)          용량 제약 위반 복구
        problem.random_schedule(rng)      무작위 가용해
        problem.edf_schedule()            휴리스틱 초기해 (EDF)
    rng : np.random.Generator             시드 고정된 난수 생성기
    budget : int                          남은 FEV (problem.cost 호출 가능 횟수)

    Returns
    -------
    best : np.ndarray, shape (707,)   최선 스케줄 (정수, 0=미배정, 1..T=창)
    history : list[float]             best-so-far 목적값 궤적 (수렴 곡선용)
    """
    # ── 예시: 단순 지역탐색 (여러분 알고리즘으로 교체) ──
    cur = problem.random_schedule(rng)          # 또는 problem.edf_schedule()
    cur_cost = problem.cost(cur)
    best, best_cost = cur.copy(), cur_cost
    history = [best_cost]

    while problem.fev < budget:
        # 이웃 생성 (예: 엔진 한 대의 창을 바꿈)
        cand = cur.copy()
        i = int(rng.integers(problem.n))
        cand[i] = int(rng.integers(0, problem.T + 1))
        cand = problem.repair(cand)             # 용량 위반 복구

        c = problem.cost(cand)                  # ← FEV 1 소모
        if c < cur_cost:                        # 개선되면 이동
            cur, cur_cost = cand, c
            if c < best_cost:
                best, best_cost = cand.copy(), c
        history.append(best_cost)

    return best, history
# ═══════════════════════════════════════════════════════════════
# ▲▲▲ 여기까지 ▲▲▲   아래는 수정하지 마세요 (전원 동일 조건)
# ═══════════════════════════════════════════════════════════════

CPW, T, K = 13, 10, 75
FEV_BUDGET = 24_000
SEEDS = [1000 + 37 * i for i in range(10)]
OBJ = dict(c_pm=1.0, c_fail=18.0, w=-0.3, setup_cost=5.0)


class Problem:
    """공통 문제 정의. cost() 호출마다 FEV를 센다."""

    def __init__(self):
        fleet = pd.read_csv(ROOT / "data/processed/fleet_rul_predictions_unified.csv")
        self.pred = fleet.pred_RUL.to_numpy(float)
        self.n = len(fleet)
        self.T, self.K = T, K
        self.part = make_part_types(np.arange(1, self.n + 1), 4)
        self.deadline = make_deadlines(self.pred, CPW, 0)
        self.fev = 0

    def cost(self, schedule) -> float:
        self.fev += 1
        return total_cost(schedule, self.deadline, part_type=self.part, **OBJ)

    def repair(self, schedule):
        return _repair(np.asarray(schedule, int), self.T, self.K)

    def random_schedule(self, rng):
        s = rng.integers(0, self.T + 1, size=self.n)
        return self.repair(s)

    def edf_schedule(self):
        return edf_schedule(self.deadline, self.T, self.K)

    def reset(self):
        self.fev = 0


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    runs = []

    for seed in SEEDS:
        p = Problem()
        p.reset()
        rng = np.random.default_rng(seed)
        t0 = time.time()
        best, history = search(p, rng, FEV_BUDGET)
        sec = time.time() - t0

        best = p.repair(best)                      # 안전장치
        best_cost = total_cost(best, p.deadline, part_type=p.part, **OBJ)

        # 수렴 곡선은 200개로 다운샘플 (파일 크기)
        if len(history) > 200:
            idx = np.linspace(0, len(history) - 1, 200).astype(int)
            history = [float(history[i]) for i in idx]

        runs.append({
            "seed": int(seed),
            "best_cost": float(best_cost),
            "schedule": [int(x) for x in best],
            "fev_used": int(p.fev),
            "seconds": round(sec, 2),
            "history": [float(h) for h in history],
        })
        print(f"  seed {seed}: cost={best_cost:9.1f}  FEV={p.fev:6d}  ({sec:.1f}s)")

    costs = [r["best_cost"] for r in runs]
    print(f"\n{ALGO_NAME}: mean {np.mean(costs):.1f} ± {np.std(costs):.1f} "
          f"| best {min(costs):.1f} | worst {max(costs):.1f}")

    out = out_dir / f"{ALGO_NAME}_results.json"
    json.dump({
        "algorithm": ALGO_NAME,
        "author": AUTHOR,
        "params": PARAMS,
        "fev_budget": FEV_BUDGET,
        "runs": runs,
    }, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n저장: {out}  ← 이 파일을 보내주세요")


if __name__ == "__main__":
    main()
