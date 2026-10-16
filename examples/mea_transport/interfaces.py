"""SI transport contract and explicitly signed BioLogic CSV import.

DFT/MD results require a validated continuum closure before using this adapter.
An EIS CSV is observed data only: this prototype does not predict impedance.
"""

import csv
import json
from pathlib import Path
import numpy as np
from .physics import Cases, face_flux


def load_transport(path):
    obj = json.loads(Path(path).read_text())
    if obj.get("schema_version") != "mea_transport_si_v1":
        raise ValueError("Expected mea_transport_si_v1")
    if obj.get("species") != ["water_proxy", "ion_proxy"]:
        raise ValueError("Expected species order: water_proxy, ion_proxy")
    if obj.get("boundary_type", "dirichlet") != "dirichlet":
        raise ValueError("Only Dirichlet boundaries are implemented")
    L, Dr = float(obj["length_m"]), float(obj["D_ref_m2_s"])
    cr = np.asarray(obj["concentration_ref_mol_m3"], dtype=float)
    x = np.asarray(obj["x_m"], dtype=float)
    if not (
        np.isfinite(L)
        and np.isfinite(Dr)
        and L > 0
        and Dr > 0
        and cr.shape == (2,)
        and np.all(np.isfinite(cr))
        and np.all(cr > 0)
    ):
        raise ValueError(
            "Positive finite length, D_ref and two concentration references required"
        )
    if (
        len(x) < 17
        or not np.all(np.isfinite(x))
        or not np.allclose(x, np.linspace(0, L, len(x)), rtol=1e-7, atol=L * 1e-9)
    ):
        raise ValueError("Need a uniform grid of at least 17 nodes spanning [0,L]")
    shape = (2, len(x))

    def field(key):
        a = np.asarray(obj[key], dtype=float)
        if a.shape == (2,):
            a = a[:, None]
        try:
            a = np.broadcast_to(a, shape).copy()
        except ValueError as exc:
            raise ValueError(f"{key}: expected two constants or (2,N)") from exc
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{key}: finite values required")
        return a[None]

    D = field("D_m2_s") / Dr
    v = field("v_m_s") * L / Dr
    k = field("k_per_s") * L**2 / Dr
    src = field("source_mol_m3_s") * L**2 / (Dr * cr[None, :, None])
    left = np.asarray(obj["boundary_left_mol_m3"], dtype=float)
    right = np.asarray(obj["boundary_right_mol_m3"], dtype=float)
    if (
        left.shape != (2,)
        or right.shape != (2,)
        or not np.all(np.isfinite([left, right]))
    ):
        raise ValueError("Need two finite concentrations for each boundary")
    if np.any(D <= 0) or np.any(k < 0) or np.any(left < 0) or np.any(right < 0):
        raise ValueError(
            "D must be positive; k and boundary concentrations nonnegative"
        )
    cases = Cases(D, v, k, src, left[None] / cr, right[None] / cr)
    # Advisory only: marginal ranges cannot detect all distribution shifts.
    flags = []
    for label, val, lo, hi in [
        ("D", D, 0.05 * np.exp(-0.45), 0.18 * np.exp(0.45)),
        ("v", v, -0.18, 0.18),
        ("k", k, 0, 0.6),
        ("source", src, 0, 0.24),
        ("left_boundary", cases.left, 0.2, 1),
        ("right_boundary", cases.right, 0.2, 1),
    ]:
        if np.any((val < lo - 1e-7) | (val > hi + 1e-7)):
            flags.append(label)
    return obj, cases, flags


def output_si(c, cases, request, flags):
    cr = np.asarray(request["concentration_ref_mol_m3"])
    L, Dr = request["length_m"], request["D_ref_m2_s"]
    J = face_flux(c, cases)[0] * cr[:, None] * Dr / L
    return {
        "schema_version": "mea_transport_prediction_si_v1",
        "species": ["water_proxy", "ion_proxy"],
        "x_m": request["x_m"],
        "concentration_mol_m3": (c[0] * cr[:, None]).tolist(),
        "flux_faces_mol_m2_s": J.tolist(),
        "flux_face_x_m": (
            (np.asarray(request["x_m"])[:-1] + np.asarray(request["x_m"])[1:]) / 2
        ).tolist(),
        "marginal_ood_flags": flags,
        "joint_distribution_validated": False,
        "impedance_prediction": None,
        "note": "Steady uncoupled proxies only. Fluxes at half-grid faces. No transient, migration, EIS, FE or degradation prediction.",
    }


def load_eis_csv(path, metadata_path):
    """Normalized EC-Lab export; z_imag_ohm must be Im(Z), not -Im(Z)."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = ["frequency_hz", "z_real_ohm", "z_imag_ohm"]
        if not reader.fieldnames or not set(names).issubset(reader.fieldnames):
            raise ValueError("Required headers: " + ",".join(names))
        data = np.array([[float(row[k]) for k in names] for row in reader])
    if (
        data.ndim != 2
        or len(data) < 2
        or not np.all(np.isfinite(data))
        or np.any(data[:, 0] <= 0)
    ):
        raise ValueError("Need >=2 finite observations and positive frequencies")
    meta = json.loads(Path(metadata_path).read_text())
    required = [
        "schema_version",
        "sample_id",
        "active_area_m2",
        "temperature_K",
        "dc_current_A",
        "dc_voltage_V",
        "ac_amplitude",
        "ac_amplitude_unit",
        "control_mode",
        "timestamp_utc",
        "stationarity_checked",
        "z_imag_sign",
    ]
    if (
        any(k not in meta for k in required)
        or meta["schema_version"] != "mea_eis_observation_v1"
    ):
        raise ValueError("Incomplete mea_eis_observation_v1 metadata")
    if meta["z_imag_sign"] != "Im(Z)":
        raise ValueError("Convert vendor -Im(Z) values by multiplying by -1")
    if meta["control_mode"] not in ["GEIS", "PEIS"]:
        raise ValueError("control_mode must be GEIS or PEIS")
    numeric = np.array(
        [
            meta["temperature_K"],
            meta["dc_current_A"],
            meta["dc_voltage_V"],
            meta["ac_amplitude"],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(numeric)) or numeric[0] <= 0 or numeric[-1] <= 0:
        raise ValueError(
            "Finite operating values, positive temperature and AC amplitude required"
        )
    expected = (
        ["A_peak", "A_rms"] if meta["control_mode"] == "GEIS" else ["V_peak", "V_rms"]
    )
    if meta["ac_amplitude_unit"] not in expected:
        raise ValueError(
            "Explicit peak/rms amplitude unit must match GEIS/PEIS control"
        )
    area = float(meta["active_area_m2"])
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Positive active_area_m2 required")
    z = data[:, 1] + 1j * data[:, 2]
    return {
        "frequency_hz": data[:, 0],
        "Z_ohm": z,
        "Z_area_ohm_m2": z * area,
        "metadata": meta,
    }


def main():
    import argparse
    import torch
    from .model import TransportFNO

    p = argparse.ArgumentParser(
        description="Infer on one SI transport request using a locally trained checkpoint."
    )
    p.add_argument("--request", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    req, cases, flags = load_transport(args.request)
    model = TransportFNO().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        c = model(torch.tensor(cases.fields())).numpy()
    Path(args.output).write_text(
        json.dumps(output_si(c, cases, req, flags), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
