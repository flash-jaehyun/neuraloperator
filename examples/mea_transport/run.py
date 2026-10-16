"""Run at repository root: python -m examples.mea_transport.run --out ..."""

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .physics import sample_cases, solve_fd, diagnostics
from .model import TransportFNO, torch_physics


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        return next(
            (
                line.split(":", 1)[1].strip()
                for line in path.read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
    return platform.processor() or platform.machine()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def latency(fn, device, repeats=30):
    for _ in range(5):
        fn()
    sync(device)
    timings = []
    for _ in range(repeats):
        sync(device)
        t = time.perf_counter()
        fn()
        sync(device)
        timings.append((time.perf_counter() - t) * 1000)
    return float(np.median(timings))


def benchmark(model, cases, threads):
    out = {}
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            out[device] = {
                "available": False,
                "note": "Not measured; no CUDA device in this execution environment.",
            }
            continue
        m = copy.deepcopy(model).to(device).eval()
        f = torch.tensor(cases.fields(), device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            one = latency(lambda: m(f[:1]), device)
            batch = latency(lambda: m(f), device)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=0.001)

        def step():
            opt.zero_grad(set_to_none=True)
            c = m(f[:32])
            r, b = torch_physics(c, f[:32])
            loss = (
                c.square().mean() + 0.01 * r.square().mean() + 0.1 * b.square().mean()
            )
            loss.backward()
            opt.step()

        train_ms = latency(step, device, repeats=10)
        out[device] = {
            "available": True,
            "single_inference_ms": one,
            "batch_inference_ms": batch,
            "batch_size": len(f),
            "train_step_batch32_ms": train_ms,
            "device_name": (
                torch.cuda.get_device_name()
                if device == "cuda"
                else platform.processor()
            ),
            "threads": threads if device == "cpu" else None,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
        }
        del m, f, opt
    single_case = sample_cases(1, cases.D.shape[-1], seed=7)
    out["fd_cpu"] = {
        "single_solve_ms": latency(lambda: solve_fd(single_case), "cpu"),
        "batch_solve_ms": latency(lambda: solve_fd(cases), "cpu"),
        "batch_size": len(cases.D),
        "note": "Both FD and FNO use pre-built coefficients/inputs. No network initialization, imports or host/device copies included.",
    }
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    out["memory_estimate"] = {
        "parameter_bytes": params_bytes,
        "parameter_gradient_adam_lower_bound_bytes": 4 * params_bytes,
        "note": "Tensor-only lower bound: parameters, gradient, two Adam moments; excludes activations, FFT workspace, allocator and CUDA context. No VRAM fit or speed claim.",
    }
    return out


def predict(model, fields, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for f in torch.tensor(fields).split(64):
            outputs.append(model(f.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def train(name, args, train_fields, train_y, val_fields, val_y):
    torch.manual_seed(args.seed)  # same initialization and minibatch order for ablation
    model = TransportFNO().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.0001
    )
    f = torch.tensor(train_fields, device=args.device)
    y = torch.tensor(train_y, dtype=torch.float32, device=args.device)
    gen = torch.Generator().manual_seed(args.seed + 1)
    best, history, best_epoch = float("inf"), [], 0
    sync(args.device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(f), generator=gen).to(args.device)
        total = 0.0
        for idx in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = model(f[idx])
            data = (
                (pred - y[idx]).square().mean(-1) / (y[idx].square().mean(-1) + 1e-8)
            ).mean()
            loss = data
            if name == "pino":
                r, b = torch_physics(pred, f[idx])
                loss = (
                    loss
                    + args.physics_weight * r.square().mean()
                    + args.balance_weight * b.square().mean()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        vp = predict(model, val_fields, args.device)
        vl = float(
            np.mean(
                np.linalg.norm(vp - val_y, axis=-1) / np.linalg.norm(val_y, axis=-1)
            )
        )
        history.append(
            {"epoch": epoch, "train_loss": total / len(f), "validation_relative_l2": vl}
        )
        if vl < best:
            best, best_epoch = vl, epoch
            state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if torch.is_tensor(v)
            }
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"{name}: epoch={epoch} loss={total/len(f):.5g} val_relL2={vl:.4%}",
                flush=True,
            )
    sync(args.device)
    seconds = time.perf_counter() - start
    model.load_state_dict(state)
    torch.save(
        {
            "state_dict": state,
            "args": {**vars(args), "out": str(args.out)},
            "best_epoch": best_epoch,
            "channels": ["water_proxy", "ion_proxy"],
            "schema": "mea_transport_v1",
        },
        args.out / f"{name}.pt",
    )
    return model, {
        "wall_seconds_including_validation": seconds,
        "best_epoch": best_epoch,
        "best_validation_relative_l2": best,
        "history": history,
    }


def make_plot(records, histories, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, regime in zip(axes[0], ["id", "D_ood", "interface_ood"]):
        r = records[regime]
        x = np.linspace(0, 1, r["truth"].shape[-1])
        ax.plot(x, r["truth"][0, 0], "k-", label="FD")
        ax.plot(x, r["data"][0, 0], "--", label="data FNO")
        ax.plot(x, r["pino"][0, 0], ":", label="PINO")
        ax.set(
            title=regime + " (case 0, water proxy)", xlabel="x / L", ylabel="c / c_ref"
        )
        ax.legend(fontsize=8)
    for name, r in histories.items():
        axes[1, 0].semilogy(
            [h["epoch"] for h in r["history"]],
            [h["validation_relative_l2"] for h in r["history"]],
            label=name,
        )
    axes[1, 0].set(xlabel="Epoch", ylabel="Validation relative L2")
    axes[1, 0].legend()
    regimes = list(records)
    for j, metric in enumerate(["relative_l2_mean", "mass_defect_mean"], start=1):
        for name, off in [("data", -0.18), ("pino", 0.18)]:
            vals = [records[r]["metrics"][name][metric] for r in regimes]
            axes[1, j].bar(np.arange(len(regimes)) + off, vals, width=0.36, label=name)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xticks(
            np.arange(len(regimes)), regimes, rotation=40, ha="right", fontsize=8
        )
        axes[1, j].set_ylabel(metric)
        axes[1, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, default=Path("examples/mea_transport/runs/default")
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=96)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--nodes", type=int, default=33)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--physics-weight", type=float, default=0.01)
    p.add_argument("--balance-weight", type=float, default=0.1)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error(
            "CUDA requested but unavailable. Install a CUDA torch build and use a CUDA-capable host."
        )
    if (
        args.nodes < 17
        or min(
            args.epochs,
            args.n_train,
            args.n_val,
            args.n_test,
            args.batch_size,
            args.threads,
        )
        < 1
    ):
        p.error("Need nodes >=17 and positive epoch, sample, batch and thread counts.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(args.seed)
    t = time.perf_counter()
    tr = sample_cases(args.n_train, args.nodes, args.seed)
    va = sample_cases(args.n_val, args.nodes, args.seed + 100)
    train_y, val_y = solve_fd(tr), solve_fd(va)
    tf, vf = tr.fields(), va.fields()
    np.savez_compressed(
        args.out / "training.npz",
        train_fields=tf,
        train_y=train_y,
        val_fields=vf,
        val_y=val_y,
    )
    generation_s = time.perf_counter() - t
    models, histories = {}, {}
    for name in ["data", "pino"]:
        models[name], histories[name] = train(name, args, tf, train_y, vf, val_y)
    records = {}
    for regime in ["id", "bc_ood", "D_ood", "v_ood", "interface_ood", "mesh_ood"]:
        n = 2 * (args.nodes - 1) + 1 if regime == "mesh_ood" else args.nodes
        cases = sample_cases(
            args.n_test, n, args.seed + 200, "id" if regime == "mesh_ood" else regime
        )
        truth = solve_fd(cases)
        metrics = {"fd": diagnostics(truth, truth, cases)}
        preds = {
            name: predict(model, cases.fields(), args.device)
            for name, model in models.items()
        }
        for name, pred in preds.items():
            metrics[name] = diagnostics(pred, truth, cases)
        x = np.linspace(0, 1, n)
        lift = cases.left[..., None] * (1 - x) + cases.right[..., None] * x
        metrics["linear_lift"] = diagnostics(lift, truth, cases)
        records[regime] = {"truth": truth, **preds, "metrics": metrics}
        np.savez_compressed(
            args.out / f"{regime}.npz", fields=cases.fields(), truth=truth, **preds
        )
    fine = solve_fd(
        sample_cases(args.n_test, 4 * (args.nodes - 1) + 1, args.seed + 200)
    )
    coarse = records["id"]["truth"]
    mesh_error = np.linalg.norm(coarse - fine[..., ::4], axis=-1) / np.linalg.norm(
        fine[..., ::4], axis=-1
    )
    bench = benchmark(
        models["pino"],
        sample_cases(args.n_test, args.nodes, args.seed + 300),
        args.threads,
    )
    try:
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base = "unavailable"
    summary = {
        "config": {**vars(args), "out": str(args.out)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "logical_cpus_visible": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "cpu_model": cpu_model_name(),
            "base_commit": base,
        },
        "parameters_real_scalar_equivalent": sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in models["pino"].parameters()
        ),
        "data_generation_train_val_seconds": generation_s,
        "training": histories,
        "evaluation": {r: v["metrics"] for r, v in records.items()},
        "fd_coarse_vs_4x_fine_relative_l2_mean": float(mesh_error.mean()),
        "benchmark": bench,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if platform.system() == "Darwin" else 1024),
        "dataset_sha256": hashlib.sha256(
            (args.out / "training.npz").read_bytes()
        ).hexdigest(),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plot(records, histories, args.out)
    print(
        json.dumps(
            {
                r: {
                    name: {k: v[k] for k in ["relative_l2_mean", "mass_defect_mean"]}
                    for name, v in rec["metrics"].items()
                }
                for r, rec in records.items()
            },
            indent=2,
        )
    )
    print(f"Saved {args.out}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
