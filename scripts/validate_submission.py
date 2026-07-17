"""팀원 제출물 검증 — 규격 위반을 즉시 잡는다.

사용:
  python scripts/validate_submission.py results/TabuSearch_results.json
  python scripts/validate_submission.py results/*.json          (전체 검사)

검사 항목:
  1. JSON 스키마 (필수 키)
  2. 시드 10개 (단일 실행 금지)
  3. FEV 예산 준수 (24,000 ±5%)
  4. 스케줄 유효성 (길이 707, 값 범위, 용량 제약 K=75)
  5. 보고한 best_cost가 실제 목적값과 일치하는가 (조작·버그 검출)
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import make_deadlines, make_part_types, total_cost

CPW, T, K = 13, 10, 75
FEV_BUDGET = 24_000
EXPECTED_SEEDS = {1000 + 37 * i for i in range(10)}
OBJ = dict(c_pm=1.0, c_fail=18.0, w=-0.3, setup_cost=5.0)

fleet = pd.read_csv(ROOT / "data/processed/fleet_rul_predictions_unified.csv")
N = len(fleet)
PART = make_part_types(np.arange(1, N + 1), 4)
DEADLINE = make_deadlines(fleet.pred_RUL.to_numpy(float), CPW, 0)


def validate(path: Path) -> bool:
    print(f"\n{'='*60}\n검증: {path.name}")
    errs, warns = [], []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ❌ JSON 파싱 실패: {e}")
        return False

    # 1. 스키마
    for k in ["algorithm", "author", "params", "fev_budget", "runs"]:
        if k not in d:
            errs.append(f"필수 키 없음: '{k}'")
    if errs:
        for e in errs:
            print(f"  ❌ {e}")
        return False

    runs = d["runs"]
    print(f"  알고리즘: {d['algorithm']}  |  작성: {d['author']}")
    print(f"  파라미터: {d['params']}")

    # 2. 시드
    seeds = {r.get("seed") for r in runs}
    if len(runs) < 10:
        errs.append(f"실행 {len(runs)}회 — 10회 필요 (단일/소수 실행 금지)")
    if seeds != EXPECTED_SEEDS:
        warns.append(f"시드 불일치 — 규격: {sorted(EXPECTED_SEEDS)[:3]}... / 제출: {sorted(seeds)[:3]}...")

    # 3~5. 각 run 검사
    costs, fevs, mismatch = [], [], 0
    for r in runs:
        s = np.asarray(r.get("schedule", []), int)
        if len(s) != N:
            errs.append(f"seed {r.get('seed')}: 스케줄 길이 {len(s)} (707이어야 함)")
            continue
        if s.min() < 0 or s.max() > T:
            errs.append(f"seed {r.get('seed')}: 창 번호 범위 위반 [{s.min()},{s.max()}]")
        load = np.bincount(s, minlength=T + 1)[1:]
        if (load > K).any():
            errs.append(f"seed {r.get('seed')}: 용량 위반 (창별 최대 {load.max()} > K={K})")

        actual = total_cost(s, DEADLINE, part_type=PART, **OBJ)
        reported = float(r.get("best_cost", np.nan))
        if not np.isfinite(reported) or abs(actual - reported) > 1e-6:
            mismatch += 1
            if mismatch <= 2:
                errs.append(f"seed {r.get('seed')}: 보고 비용 {reported:.1f} ≠ 실제 {actual:.1f}")
        costs.append(actual)

        fev = r.get("fev_used", 0)
        fevs.append(fev)
        if fev > FEV_BUDGET * 1.05:
            errs.append(f"seed {r.get('seed')}: FEV {fev:,} > 예산 {FEV_BUDGET:,} (초과)")
        elif fev < FEV_BUDGET * 0.5:
            warns.append(f"seed {r.get('seed')}: FEV {fev:,} — 예산({FEV_BUDGET:,})을 절반도 안 씀")

    # 결과
    if costs:
        print(f"  목적값: mean {np.mean(costs):.1f} ± {np.std(costs):.1f} "
              f"| best {min(costs):.1f} | worst {max(costs):.1f}")
        print(f"  FEV   : 평균 {np.mean(fevs):,.0f} / 예산 {FEV_BUDGET:,}")
        if any("history" in r and r["history"] for r in runs):
            print(f"  수렴곡선: 있음 ✅")
        else:
            warns.append("history(수렴 곡선) 없음 — 수렴 비교 그림을 못 그림")

    for w in warns:
        print(f"  ⚠️  {w}")
    for e in errs:
        print(f"  ❌ {e}")

    ok = not errs
    print(f"  {'✅ 통과 — 비교에 사용 가능' if ok else '❌ 불합격 — 수정 후 재제출 필요'}")
    return ok


def main():
    args = sys.argv[1:] or [str(ROOT / "results" / "*.json")]
    paths = [Path(p) for a in args for p in glob.glob(a)]
    if not paths:
        print("검증할 파일이 없습니다. 예: python scripts/validate_submission.py results/TabuSearch_results.json")
        return
    results = {p.name: validate(p) for p in paths}
    print(f"\n{'='*60}\n요약: {sum(results.values())}/{len(results)} 통과")
    for n, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {n}")


if __name__ == "__main__":
    main()
