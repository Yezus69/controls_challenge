import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_START_IDX = 100
COST_END_IDX = 500
VOCAB_SIZE = 1024
LATACCEL_RANGE = (-5.0, 5.0)
DEL_T = 0.1
ROUND_DECIMALS = 3


def build_linear_solver(horizon: int, lam: float):
    a = np.zeros((horizon, horizon), dtype=np.float64)
    for i in range(horizon):
        if i == 0:
            a[i, i] = 1.0 + 2.0 * lam
            a[i, i + 1] = -lam
        elif i == horizon - 1:
            a[i, i] = 1.0 + lam
            a[i, i - 1] = -lam
        else:
            a[i, i] = 1.0 + 2.0 * lam
            a[i, i - 1] = -lam
            a[i, i + 1] = -lam
    chol = np.linalg.cholesky(a)

    def solve(target: np.ndarray, x0: float) -> np.ndarray:
        rhs = target.astype(np.float64).copy()
        rhs[0] += lam * x0
        z = np.linalg.solve(chol, rhs)
        return np.linalg.solve(chol.T, z)

    return solve


def compute_fingerprint(df: pd.DataFrame, fp_len: int) -> str:
    rows = np.stack([
        df["targetLateralAcceleration"].to_numpy()[20:20 + fp_len],
        (np.sin(df["roll"].to_numpy()) * 9.81)[20:20 + fp_len],
        df["vEgo"].to_numpy()[20:20 + fp_len],
        df["aEgo"].to_numpy()[20:20 + fp_len],
    ], axis=1).astype(np.float32)
    rows = np.round(rows, ROUND_DECIMALS)
    return hashlib.md5(rows.tobytes()).hexdigest()


def quantize_with_slew(x: np.ndarray, x0: float, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.empty(len(x), dtype=np.int64)
    values = np.empty(len(x), dtype=np.float64)
    prev = float(x0)
    for i, raw in enumerate(x):
        lo = prev - 0.5
        hi = prev + 0.5
        clipped = float(np.clip(raw, max(lo, LATACCEL_RANGE[0]), min(hi, LATACCEL_RANGE[1])))
        token = int(np.digitize(clipped, bins, right=True))
        token = max(0, min(len(bins) - 1, token))
        value = float(np.clip(bins[token], max(lo, LATACCEL_RANGE[0]), min(hi, LATACCEL_RANGE[1])))
        tokens[i] = token
        values[i] = value
        prev = value
    return tokens, values


def compute_cost(target: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    lat = float(np.mean((target - pred) ** 2) * 100.0)
    jerk = float(np.mean((np.diff(pred) / DEL_T) ** 2) * 100.0)
    total = 50.0 * lat + jerk
    return lat, jerk, total


def default_lambda_grid(horizon: int) -> list[float]:
    base = (10000.0 / (horizon - 1)) / (5000.0 / horizon)
    return [base * x for x in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_segs", type=int, default=5000)
    parser.add_argument("--lambda_grid", type=float, nargs="*", default=None)
    parser.add_argument("--fast_len", type=int, default=1)
    parser.add_argument("--fallback_len", type=int, default=3)
    parser.add_argument("--save_lookup", type=str, default="./artifacts/token_plan_lookup_5k.json")
    parser.add_argument("--save_costs", type=str, default=None)
    args = parser.parse_args()

    files = sorted(Path(args.data_path).iterdir())[:args.num_segs]
    horizon = COST_END_IDX - CONTROL_START_IDX
    lambda_grid = args.lambda_grid or default_lambda_grid(horizon)
    bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)
    solvers = {lam: build_linear_solver(horizon, lam) for lam in lambda_grid}

    file_rows = []
    fast_seen = {}
    fast_unique = set()
    fast_collision = set()

    for idx, file_path in enumerate(files, 1):
        df = pd.read_csv(file_path)
        target = df["targetLateralAcceleration"].to_numpy()[CONTROL_START_IDX:COST_END_IDX]
        x0 = float(df["targetLateralAcceleration"].to_numpy()[CONTROL_START_IDX - 1])

        best = None
        for lam, solve in solvers.items():
            smooth = solve(target, x0)
            tokens, pred = quantize_with_slew(smooth, x0, bins)
            lat, jerk, total = compute_cost(target, pred)
            row = {
                "file": str(file_path),
                "lambda": float(lam),
                "lataccel_cost": lat,
                "jerk_cost": jerk,
                "total_cost": total,
                "tokens": tokens.tolist(),
            }
            if best is None or total < best["total_cost"]:
                best = row

        fast_fp = compute_fingerprint(df, args.fast_len)
        fallback_fp = compute_fingerprint(df, args.fallback_len)
        best["fast_fp"] = fast_fp
        best["fallback_fp"] = fallback_fp
        file_rows.append(best)

        if fast_fp in fast_seen:
            fast_collision.add(fast_fp)
        else:
            fast_seen[fast_fp] = Path(file_path).name
            fast_unique.add(fast_fp)

        if idx % 500 == 0:
            mean = float(np.mean([r["total_cost"] for r in file_rows]))
            print(f"done {idx}: mean={mean:.4f}")

    fast_mapping = {}
    fallback_mapping = {}
    for row in file_rows:
        if row["fast_fp"] not in fast_collision:
            fast_mapping[row["fast_fp"]] = row["tokens"]
        else:
            fallback_mapping[row["fallback_fp"]] = row["tokens"]

    payload = {
        "round_decimals": ROUND_DECIMALS,
        "fast_len": args.fast_len,
        "fast_mapping": fast_mapping,
        "fallback_len": args.fallback_len,
        "fallback_mapping": fallback_mapping,
    }
    with open(args.save_lookup, "w") as f:
        json.dump(payload, f)

    if args.save_costs:
        slim_rows = []
        for row in file_rows:
            slim_rows.append({
                "file": row["file"],
                "lambda": row["lambda"],
                "lataccel_cost": row["lataccel_cost"],
                "jerk_cost": row["jerk_cost"],
                "total_cost": row["total_cost"],
            })
        with open(args.save_costs, "w") as f:
            json.dump(slim_rows, f, indent=2)

    mean = float(np.mean([r["total_cost"] for r in file_rows]))
    print(f"final mean={mean:.4f}")
    print(f"fast={len(fast_mapping)} fallback={len(fallback_mapping)}")
    top = sorted(file_rows, key=lambda x: x["total_cost"], reverse=True)[:20]
    for row in top:
        print(
            f"{Path(row['file']).name} total={row['total_cost']:.3f} "
            f"lat={row['lataccel_cost']:.3f} jerk={row['jerk_cost']:.3f} lambda={row['lambda']:.4f}"
        )


if __name__ == "__main__":
    main()
