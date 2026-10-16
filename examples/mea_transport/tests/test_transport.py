from pathlib import Path
import numpy as np
import pytest
import torch
from examples.mea_transport.physics import (
    Cases,
    sample_cases,
    solve_fd,
    diagnostics,
    residual,
)
from examples.mea_transport.model import TransportFNO, torch_physics
from examples.mea_transport.interfaces import load_transport, output_si, load_eis_csv


def test_pure_diffusion_linear_exact():
    cases = sample_cases(3, seed=8)
    cases.D[:] = 0.1
    cases.v[:] = 0
    cases.k[:] = 0
    cases.source[:] = 0
    x = np.linspace(0, 1, 33)
    exact = cases.left[..., None] * (1 - x) + cases.right[..., None] * x
    np.testing.assert_allclose(solve_fd(cases), exact, atol=1e-13)


def test_reference_matches_manufactured_continuum_and_converges():
    errors = []
    for n in [33, 65, 129]:
        x = np.linspace(0, 1, n)[None, None]
        c = 0.5 + 0.2 * x + 0.2 * np.sin(np.pi * x)
        cp = 0.2 + 0.2 * np.pi * np.cos(np.pi * x)
        cpp = -0.2 * np.pi**2 * np.sin(np.pi * x)
        D = 0.1 * (1 + 0.2 * np.cos(2 * np.pi * x))
        Dp = -0.04 * np.pi * np.sin(2 * np.pi * x)
        v, k = np.full_like(D, 0.12), np.full_like(D, 0.3)
        source = v * cp - Dp * cp - D * cpp + k * c
        cases = Cases(D, v, k, source, c[..., 0], c[..., -1])
        errors.append(np.linalg.norm(solve_fd(cases) - c) / np.linalg.norm(c))
    assert errors[-1] < 0.002
    assert errors[0] / errors[1] > 1.7
    assert errors[1] / errors[2] > 1.7


def test_conservation_and_torch_numpy_residual_agree():
    cases = sample_cases(5, seed=123)
    y = solve_fd(cases)
    assert diagnostics(y, y, cases)["mass_defect_mean"] < 1e-11
    np.testing.assert_allclose(residual(y, cases), 0, atol=1e-11)
    # Use float64 inputs to distinguish stencil disagreement from float32 roundoff.
    f = cases.fields().astype("float64")
    r, balance = torch_physics(torch.tensor(y), torch.tensor(f))
    np.testing.assert_allclose(r.numpy(), residual(y, cases), atol=2e-6)
    assert balance.abs().max() < 2e-7
    perturbed = y.copy()
    perturbed[..., 1] += 0.05
    assert diagnostics(perturbed, y, cases)["mass_defect_mean"] > 0.001


def test_fno_hard_boundary_gradient_and_resolution():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = TransportFNO()
    for n in [33, 65]:
        cases = sample_cases(2, n, seed=7)
        f = torch.tensor(cases.fields())
        y = model(f)
        np.testing.assert_allclose(y.detach().numpy()[..., 0], cases.left, atol=1e-7)
        np.testing.assert_allclose(y.detach().numpy()[..., -1], cases.right, atol=1e-7)
        r, b = torch_physics(y, f)
        (r.square().mean() + b.square().mean()).backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        model.zero_grad()


def test_no_split_overlap_and_resolution_cases_are_paired():
    tr, te = sample_cases(5, seed=1), sample_cases(5, seed=201)
    assert not np.array_equal(tr.D, te.D)
    fine = sample_cases(5, 65, seed=201)
    np.testing.assert_allclose(te.D, fine.D[..., ::2])
    np.testing.assert_allclose(te.left, fine.left)
    with pytest.raises(ValueError):
        sample_cases(1, regime="typo")


def test_si_and_eis_contracts(tmp_path):
    import json

    folder = Path(__file__).parents[1] / "interfaces"
    req, cases, flags = load_transport(folder / "transport_request.json")
    y = solve_fd(cases)
    result = output_si(y, cases, req, flags)
    np.testing.assert_allclose(
        np.asarray(result["concentration_mol_m3"])[:, 0], req["boundary_left_mol_m3"]
    )
    assert result["impedance_prediction"] is None
    obs = load_eis_csv(folder / "eis_observation.csv", folder / "eis_metadata.json")
    assert obs["Z_ohm"][0].imag < 0
    np.testing.assert_allclose(obs["Z_area_ohm_m2"], obs["Z_ohm"] * 0.0001)
    req["D_m2_s"][0] = -1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(req))
    with pytest.raises(ValueError):
        load_transport(bad)
    metadata = json.loads((folder / "eis_metadata.json").read_text())
    metadata["z_imag_sign"] = "-Im(Z)"
    bad.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Convert vendor"):
        load_eis_csv(folder / "eis_observation.csv", bad)


def test_tensor_only_checkpoint_round_trip(tmp_path):
    model = TransportFNO().eval()
    state = {k: v for k, v in model.state_dict().items() if torch.is_tensor(v)}
    path = tmp_path / "model.pt"
    torch.save({"state_dict": state}, path)
    restored = TransportFNO().eval()
    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    f = torch.tensor(sample_cases(1).fields())
    torch.testing.assert_close(model(f), restored(f))
