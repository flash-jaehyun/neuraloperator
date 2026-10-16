"""Dimensionless, steady 1D conservative transport. No electrochemical closure."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded


@dataclass
class Cases:
    # arrays are (case, species, node); species 0=water proxy, 1=ion proxy
    D: np.ndarray
    v: np.ndarray
    k: np.ndarray
    source: np.ndarray
    left: np.ndarray
    right: np.ndarray

    def fields(self):
        n = self.D.shape[-1]
        x = np.broadcast_to(np.linspace(0, 1, n), self.D.shape)
        # Fixed physical scales, never fitted on validation/test data.
        return np.concatenate(
            [
                np.log(self.D / 0.1),
                self.v / 0.2,
                self.k / 0.8,
                self.source / 0.2,
                np.broadcast_to(self.left[..., None], self.D.shape),
                np.broadcast_to(self.right[..., None], self.D.shape),
                x[:, :1],
            ],
            axis=1,
        ).astype("float32")


def sample_cases(count, n=33, seed=0, regime="id"):
    """Independent seeds by split; same seed at another n gives same continuum case."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)[None, None, :]
    shape = (count, 2, 1)
    d0 = rng.uniform(0.05, 0.18, shape)
    amp = rng.uniform(-0.45, 0.45, shape)
    phase = rng.uniform(0, 2 * np.pi, shape)
    D = d0 * np.exp(amp * np.sin(2 * np.pi * x + phase))
    v = np.broadcast_to(rng.uniform(-0.18, 0.18, shape), D.shape).copy()
    k = np.broadcast_to(rng.uniform(0, 0.6, shape), D.shape).copy()
    source = rng.uniform(0, 0.2, shape) * (1 + 0.2 * np.sin(np.pi * x))
    left = rng.uniform(0.2, 1, (count, 2))
    right = rng.uniform(0.2, 1, (count, 2))
    if regime == "bc_ood":
        left = rng.uniform(1.15, 1.5, (count, 2))
        right = rng.uniform(0.05, 0.15, (count, 2))
    elif regime == "D_ood":
        D *= 0.25
    elif regime == "v_ood":
        v = np.broadcast_to(
            rng.choice([-1.0, 1.0], shape) * rng.uniform(0.35, 0.5, shape), D.shape
        ).copy()
    elif regime == "interface_ood":
        D = np.broadcast_to(d0, D.shape).copy() * np.where(x < 0.5, 1.0, 0.2)
    elif regime != "id":
        raise ValueError(f"Unknown regime: {regime}")
    return Cases(D, v, k, source, left, right)


def face_flux(c, cases):
    """Harmonic diffusion + first-order upwind drift on half-grid faces."""
    h = 1 / (c.shape[-1] - 1)
    Df = (
        2
        * cases.D[..., :-1]
        * cases.D[..., 1:]
        / (cases.D[..., :-1] + cases.D[..., 1:])
    )
    vf = 0.5 * (cases.v[..., :-1] + cases.v[..., 1:])
    return (
        -Df * np.diff(c, axis=-1) / h
        + np.maximum(vf, 0) * c[..., :-1]
        + np.minimum(vf, 0) * c[..., 1:]
    )


def residual(c, cases):
    h = 1 / (c.shape[-1] - 1)
    return (
        np.diff(face_flux(c, cases), axis=-1) / h
        + cases.k[..., 1:-1] * c[..., 1:-1]
        - cases.source[..., 1:-1]
    )


def solve_fd(cases):
    """Banded conservative finite differences (equiv. nodal finite volumes)."""
    b, channels, n = cases.D.shape
    h = 1 / (n - 1)
    c = np.empty_like(cases.D)
    c[..., 0], c[..., -1] = cases.left, cases.right
    Df = (
        2
        * cases.D[..., :-1]
        * cases.D[..., 1:]
        / (cases.D[..., :-1] + cases.D[..., 1:])
    )
    vf = 0.5 * (cases.v[..., :-1] + cases.v[..., 1:])
    a = Df / h + np.maximum(vf, 0)  # J_face = a*c_left + b*c_right
    z = -Df / h + np.minimum(vf, 0)
    for i in range(b):
        for ch in range(channels):
            lower = -a[i, ch, :-1] / h
            diag = (a[i, ch, 1:] - z[i, ch, :-1]) / h + cases.k[i, ch, 1:-1]
            upper = z[i, ch, 1:] / h
            rhs = cases.source[i, ch, 1:-1].copy()
            rhs[0] -= lower[0] * cases.left[i, ch]
            rhs[-1] -= upper[-1] * cases.right[i, ch]
            ab = np.zeros((3, n - 2))
            ab[0, 1:], ab[1], ab[2, :-1] = upper[:-1], diag, lower[1:]
            c[i, ch, 1:-1] = solve_banded((1, 1), ab, rhs)
    return c


def diagnostics(c, truth, cases):
    h = 1 / (c.shape[-1] - 1)
    flux = face_flux(c, cases)
    src = h * cases.source[..., 1:-1].sum(-1)
    sink = h * (cases.k[..., 1:-1] * c[..., 1:-1]).sum(-1)
    imbalance = flux[..., -1] - flux[..., 0] + sink - src
    throughput = (
        np.abs(flux[..., -1])
        + np.abs(flux[..., 0])
        + np.abs(src)
        + np.abs(sink)
        + 1e-10
    )
    rel = np.linalg.norm(c - truth, axis=-1) / np.maximum(
        np.linalg.norm(truth, axis=-1), 1e-10
    )
    mass = np.abs(imbalance) / throughput
    return {
        "relative_l2_mean": float(rel.mean()),
        "relative_l2_p95": float(np.quantile(rel, 0.95)),
        "relative_l2_per_species": rel.mean(0).tolist(),
        "mass_defect_mean": float(mass.mean()),
        "mass_defect_p95": float(np.quantile(mass, 0.95)),
        "absolute_mass_defect_mean": float(np.abs(imbalance).mean()),
        "pde_rms": float(np.sqrt(np.mean(residual(c, cases) ** 2))),
        "boundary_max_abs": float(
            max(
                np.max(abs(c[..., 0] - cases.left)),
                np.max(abs(c[..., -1] - cases.right)),
            )
        ),
        "negative_fraction": float((c < 0).mean()),
        "minimum_c": float(c.min()),
    }
