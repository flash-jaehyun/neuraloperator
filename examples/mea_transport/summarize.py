"""Archive the three reproducible runs and regenerate the README result tables."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("examples/mea_transport/runs"))
    p.add_argument(
        "--results", type=Path, default=Path("examples/mea_transport/results")
    )
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    seeds = [20261016, 20261017, 20261018]
    runs = []
    for seed in seeds:
        obj = json.loads((args.runs / f"cpu_seed{seed}" / "metrics.json").read_text())
        # The configuration is invariant across seeds; prohibit accidental mixing.
        runs.append(obj)
        archived = dict(obj)
        archived["training"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "history"}
            for k, v in obj["training"].items()
        }
        (args.results / f"seed{seed}.json").write_text(
            json.dumps(archived, indent=2) + "\n"
        )
    keys = [
        "epochs",
        "n_train",
        "n_val",
        "n_test",
        "nodes",
        "batch_size",
        "physics_weight",
        "balance_weight",
        "threads",
        "device",
    ]
    for obj in runs[1:]:
        if any(obj["config"][key] != runs[0]["config"][key] for key in keys):
            raise ValueError("Configurations differ; do not aggregate")
    metrics = [
        "relative_l2_mean",
        "relative_l2_p95",
        "mass_defect_mean",
        "mass_defect_p95",
        "pde_rms",
        "boundary_max_abs",
        "negative_fraction",
    ]
    aggregate = {}
    for regime in runs[0]["evaluation"]:
        aggregate[regime] = {}
        for name in runs[0]["evaluation"][regime]:
            aggregate[regime][name] = {}
            for metric in metrics:
                vals = [r["evaluation"][regime][name][metric] for r in runs]
                aggregate[regime][name][metric] = {
                    "mean": float(np.mean(vals)),
                    "min": min(vals),
                    "max": max(vals),
                }
    source = Path(__file__).parent
    summary = {
        "seeds": seeds,
        "aggregation": "Unweighted mean of three run-level means; range is seed-to-seed, not a confidence interval.",
        "evaluation": aggregate,
        "fd_coarse_vs_fine_relative_l2": [
            r["fd_coarse_vs_4x_fine_relative_l2_mean"] for r in runs
        ],
        "source_sha256": {
            n: hashlib.sha256((source / n).read_bytes()).hexdigest()
            for n in [
                "physics.py",
                "model.py",
                "run.py",
                "interfaces.py",
                "summarize.py",
            ]
        },
    }
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(
        args.runs / f"cpu_seed{seeds[0]}" / "comparison.png",
        args.results / "comparison.png",
    )
    table = [
        "3개 seed의 평균. 괄호는 seed별 평균의 최소–최대이며 confidence interval이 아니다. 모든 값은 %다.",
        "",
        "| 조건 | 일반 FNO 농도 L2 | PINO 농도 L2 | 일반 FNO 질량수지 | PINO 질량수지 |",
        "|---|---:|---:|---:|---:|",
    ]

    def fmt(regime, name, metric):
        a = aggregate[regime][name][metric]
        return f"{100*a['mean']:.2f} ({100*a['min']:.2f}–{100*a['max']:.2f})"

    for regime in aggregate:
        table.append(
            "| "
            + regime
            + " | "
            + " | ".join(
                fmt(regime, name, metric)
                for metric, name in [
                    ("relative_l2_mean", "data"),
                    ("relative_l2_mean", "pino"),
                    ("mass_defect_mean", "data"),
                    ("mass_defect_mean", "pino"),
                ]
            )
            + " |"
        )
    table += [
        "",
        f"FD 33→129 노드 해 차이의 평균 L2는 seed별 {', '.join(f'{100*x:.3f}%' for x in summary['fd_coarse_vs_fine_relative_l2'])}다. "
        "이 값은 연속 참해에 대한 엄밀한 error bound가 아니며, network 오차와 별도로 보고한다.",
    ]
    first = runs[0]
    cpu, fd = first["benchmark"]["cpu"], first["benchmark"]["fd_cpu"]
    times = [r["training"]["pino"]["wall_seconds_including_validation"] for r in runs]
    resource_table = [
        f"CPU 실측: {first['environment']['cpu_model']}, PyTorch threads={first['config']['threads']}, Python {first['environment']['python']}, torch {first['environment']['torch']}. 공유 VM이므로 workstation 절대시간 보장은 아니다.",
        "",
        "| 항목 | 실측/계산값 |",
        "|---|---:|",
        f"| PINO 학습 150 epoch + 매 epoch 검증 | 평균 {np.mean(times):.1f} s, 범위 {min(times):.1f}–{max(times):.1f} s |",
        f"| 대표 seed FNO 한 개 추론, 33노드 | {cpu['single_inference_ms']:.3f} ms |",
        f"| 대표 seed FD 한 개 해, 33노드 | {fd['single_solve_ms']:.3f} ms |",
        f"| 대표 seed FNO/FD 128개 배치 | {cpu['batch_inference_ms']:.3f} / {fd['batch_solve_ms']:.3f} ms |",
        f"| 모델 파라미터 (복소수를 실수 2개로 계산) | {first['parameters_real_scalar_equivalent']:,} |",
        f"| 파라미터 tensor 크기 | {first['benchmark']['memory_estimate']['parameter_bytes']/1024:.1f} KiB |",
        f"| 파라미터+gradient+Adam 상태 tensor 하한 | {first['benchmark']['memory_estimate']['parameter_gradient_adam_lower_bound_bytes']/1024:.1f} KiB |",
        f"| 첫 전체 실행 process peak RSS | {first['peak_process_rss_mib']:.1f} MiB |",
        "| GPU 시간 / VRAM 실측 | GPU 없음: 측정 안 됨 |",
    ]
    readme = source / "README.md"
    s = readme.read_text()
    for key, text in [
        ("RESULTS_TABLE", "\n".join(table)),
        ("RESOURCE_TABLE", "\n".join(resource_table)),
    ]:
        start, end = f"<!-- {key} -->", f"<!-- END_{key} -->"
        if end in s:
            s = (
                s[: s.index(start) + len(start)]
                + "\n\n"
                + text
                + "\n\n"
                + s[s.index(end) :]
            )
        else:
            s = s.replace(start, start + "\n\n" + text + "\n\n" + end)
    readme.write_text(s)
    print(
        json.dumps(
            {r: aggregate[r]["pino"]["relative_l2_mean"] for r in aggregate}, indent=2
        )
    )


if __name__ == "__main__":
    main()
