"""Auditable shared-IL dataset relationships using development labels only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/dataset_relationship_formulas_v1.json"
METRICS = ("spearman", "distance_correlation", "continuous_mi", "binary_mi",
           "binary_i_over_h", "multiclass_mi", "predictability_cv_nmae")
SYMMETRIC = set(METRICS) - {"binary_i_over_h", "predictability_cv_nmae"}
FORMULAS = {"linear_temperature", "linear_temperature_pressure", "log_density",
            "vft", "cauchy", "inverse_temperature", "solute_linear_temperature",
            "solute_only", "unconditioned", "pending_formula"}


def json_text(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("version") != 1 or set(config["formulas"].values()) - FORMULAS:
        raise ValueError("Unsupported formula configuration")
    for key in ("temperature_K", "pressure_kPa", "wavelength_nm"):
        if not np.isfinite(config["references"][key]) or config["references"][key] <= 0:
            raise ValueError(f"Invalid reference: {key}")
    if any(not np.isfinite(value) for value in config["binary_thresholds"].values()):
        raise ValueError("Binary thresholds must be finite")
    if any(not isinstance(value, int) or value < 0 for value in config["stability"].values()):
        raise ValueError("Stability settings must be nonnegative integers")
    return config


def discover(input_root, config):
    root = Path(input_root)
    catalog_path = root / "task_catalog.csv"
    catalog = pd.read_csv(catalog_path).fillna("")
    nodes, inputs = [], {str(catalog_path): file_hash(catalog_path)}
    for _, entry in catalog.iterrows():
        stage = int(entry.stage)
        if stage not in (2, 3):
            continue
        excluded = ("excluded_transfer_organic" if "transfer_organic" in entry.source_file
                    else "excluded_non_il_identity" if entry.system_type not in ("il", "il_solute")
                    else "")
        directory = root / entry.materialized_path
        files = []
        frame = None
        if not excluded:
            if stage == 2:
                files = [directory / "train.csv", directory / "valid.csv"]
            else:
                directory = directory / "IL"
                if int(entry.repeats) > 1:
                    directory = directory / "cv1"
                files = [directory / f"fold{i}.csv" for i in range(1, 6)]
            frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
            for p in files:
                inputs[str(p)] = file_hash(p)
            required = ["cation", "anion"] + (["solute"] if entry.system_type == "il_solute" else [])
            missing = set(required) - set(frame)
            if missing:
                raise ValueError(f"{entry.task_id}: missing identity columns {missing}")
        for target in str(entry.target_columns).split(";"):
            node_id = f"{entry.source_file}::{target}"
            formula = config["formulas"].get(entry.source_file, "pending_formula")
            if frame is not None and target not in frame:
                raise ValueError(f"{node_id}: missing target")
            nodes.append({"node_id": node_id, "source_dataset": entry.source_file,
                          "target_property": target, "stage": stage,
                          "system_type": entry.system_type, "excluded_reason": excluded,
                          "formula": formula, "condition_columns": str(entry.condition_columns),
                          "files": files, "frame": frame})
    ids = [n["node_id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate catalog nodes")
    return nodes, inputs


def valid_frame(node):
    frame = node["frame"].copy()
    frame["_value"] = pd.to_numeric(frame[node["target_property"]], errors="coerce")
    roles = ["cation", "anion"] + (["solute"] if node["system_type"] == "il_solute" else [])
    mask = np.isfinite(frame._value)
    for role in roles:
        mask &= frame[role].notna() & frame[role].astype(str).str.strip().ne("")
    return frame.loc[mask].copy()


def fit_signature(group, formula, refs):
    """Return a scalar plus explicit fit/condition provenance; never fall back."""
    y = group._value.to_numpy(float)
    result = {"signature": np.nan, "n_observations": len(y), "status": "ok",
              "fit_formula": formula, "fit_method": "none", "fit_quality": "{}",
              "uncertainty": np.nan, "parameters": "{}", "residual_df": 0,
              "requested_reference": json_text(refs), "actual_reference": "{}",
              "reference_mismatch": False, "extrapolated": False,
              "extrapolation_distance": "{}", "signature_kind": "condition_corrected"}
    observed = {}
    for col in refs:
        if col in group:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            observed[col] = [float(finite.min()), float(finite.max())] if len(finite) else None
    result["condition_ranges"] = json_text(observed)
    if len(y) == 1:
        result.update(signature=float(y[0]), signature_kind="single_raw")
        result["actual_reference"] = json_text({c: bounds[0] if bounds else None for c, bounds in observed.items()})
        result["reference_mismatch"] = any(b is not None and not np.isclose(b[0], refs[c]) for c, b in observed.items())
        return result
    if formula == "unconditioned":
        result.update(signature=float(np.median(y)), fit_method="median", signature_kind="unconditioned_median",
                      fit_quality=json_text({"mad": float(np.median(np.abs(y - np.median(y))))}))
        return result
    if formula == "pending_formula":
        result["status"] = "pending_formula"
        return result
    columns = ["temperature_K"]
    if formula in {"linear_temperature_pressure", "log_density", "vft", "cauchy"}:
        columns.append("pressure_kPa")
    if formula == "cauchy":
        columns.append("wavelength_nm")
    if formula == "solute_only":
        columns = ["temperature_K"]  # Already fixed experimentally; no inferred correction.
    active, actual, distances, raw = [], {}, {}, {}
    for col in columns:
        if col not in group:
            result["status"] = "missing_condition"
            return result
        values = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            result["status"] = "invalid_condition"
            return result
        raw[col] = values
        constant = np.all(values == values[0])
        actual[col] = float(values[0]) if constant else refs[col]
        if not constant:
            active.append(col)
        distances[col] = float(max(values.min() - refs[col], refs[col] - values.max(), 0)) if not constant else 0.0
    result.update(actual_reference=json_text(actual), extrapolation_distance=json_text(distances),
                  extrapolated=any(v > 0 for v in distances.values()),
                  reference_mismatch=any(not np.isclose(v, refs[c]) for c, v in actual.items()))
    if formula == "solute_only":
        if active:
            result["status"] = "unexpected_transfer_temperature_variation"
            return result
        result.update(signature=float(np.median(y)), fit_method="median", signature_kind="constant_condition_median")
        return result
    if not active:
        result.update(signature=float(np.median(y)), fit_method="median", signature_kind="constant_condition_median")
        return result
    if formula == "log_density":
        if (y <= 0).any():
            result["status"] = "invalid_log_density"
            return result
        y = np.log(y)
    features = []
    for col in active:
        values = raw[col]
        if col == "wavelength_nm":
            features.append(values ** -2 - refs[col] ** -2)
        elif col == "temperature_K" and formula == "inverse_temperature":
            features.append(1 / values - 1 / refs[col])
        else:
            features.append(values - refs[col])
    x = np.column_stack([np.ones(len(y)), *features])
    scales = np.r_[1., np.max(np.abs(x[:, 1:]), axis=0)]
    scaled = x / scales
    if np.linalg.matrix_rank(scaled) != scaled.shape[1] or len(y) < scaled.shape[1]:
        result["status"] = "insufficient_condition_design"
        return result
    nonlinear = formula == "vft" and "temperature_K" in active
    parameter_names = ["intercept", *active]
    if nonlinear:
        t = raw["temperature_K"]
        upper = min(float(t.min()), refs["temperature_K"])
        upper = np.nextafter(upper, 0)
        ti = active.index("temperature_K") + 1
        if len(np.unique(t)) < 4 or len(y) <= x.shape[1]:
            result["status"] = "insufficient_vft_design"
            return result
        def design(t0):
            design_x = scaled.copy()
            design_x[:, ti] = (1 / (t - t0) - 1 / (refs["temperature_K"] - t0)) * refs["temperature_K"]
            return design_x
        candidates = []
        for fraction in (0.15, 0.5, 0.85):
            t0 = upper * fraction
            start = np.r_[np.linalg.lstsq(design(t0), y, rcond=None)[0], t0]
            opt = least_squares(lambda p: design(p[-1]) @ p[:-1] - y, start,
                                bounds=(np.r_[np.full(x.shape[1], -np.inf), 0],
                                        np.r_[np.full(x.shape[1], np.inf), upper]),
                                max_nfev=500, x_scale="jac")
            if opt.success and np.isfinite(opt.x).all():
                candidates.append(opt)
        if not candidates:
            result["status"] = "vft_not_converged"
            return result
        opt = min(candidates, key=lambda candidate: candidate.cost)
        if min(opt.x[-1], upper - opt.x[-1]) < max(1e-5, upper * 1e-6):
            result["status"] = "vft_boundary"
            return result
        jac = opt.jac
        if np.linalg.matrix_rank(jac) < len(opt.x):
            result["status"] = "vft_not_identifiable"
            return result
        beta = opt.x[:-1]
        predicted = design(opt.x[-1]) @ beta
        physical = beta / scales
        physical[ti] = beta[ti] * refs["temperature_K"]
        params = dict(zip(parameter_names, physical.tolist()))
        params["T0"] = float(opt.x[-1])
        p_count = len(opt.x)
        covariance_base = np.linalg.pinv(jac.T @ jac)
        result["fit_method"] = "bounded_multistart_least_squares"
    else:
        beta = np.linalg.lstsq(scaled, y, rcond=None)[0]
        predicted = scaled @ beta
        params = dict(zip(parameter_names, (beta / scales).tolist()))
        covariance_base = np.linalg.pinv(scaled.T @ scaled)
        p_count = len(beta)
        result["fit_method"] = "least_squares"
    residual = y - predicted
    df = len(y) - p_count
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    signature = float(beta[0])
    if df > 0:
        result["uncertainty"] = float(np.sqrt(max(0, covariance_base[0, 0] * np.sum(residual ** 2) / df)))
    if formula == "log_density":
        with np.errstate(over="ignore"):
            signature = float(np.exp(signature))
        result["uncertainty"] *= signature
    if not np.isfinite(signature):
        result["status"] = "nonfinite_signature"
        return result
    result.update(signature=signature, residual_df=df, parameters=json_text(params),
                  fit_quality=json_text({"rmse_fit_space": rmse, "n_parameters": p_count,
                                         "uncertainty_status": "estimated" if df > 0 else "no_residual_df"}))
    return result


def solute_signatures(frame, node, refs):
    units = []
    formula = "linear_temperature" if node["formula"] == "solute_linear_temperature" else "solute_only"
    for keys, group in frame.groupby(["cation", "anion", "solute"], sort=True):
        fit = fit_signature(group, formula, refs)
        units.append(dict(zip(["cation", "anion", "solute"], keys)) | fit)
    units = pd.DataFrame(units)
    if units.empty:
        return [], units
    good = units.loc[np.isfinite(units.signature)].copy()
    output = []
    all_keys = sorted(set(zip(frame.cation, frame.anion)))
    retained = {}
    if not good.empty:
        il, il_keys = pd.factorize(pd.MultiIndex.from_frame(good[["cation", "anion"]]), sort=True)
        sol, sol_keys = pd.factorize(good.solute, sort=True)
        ni, ns = len(il_keys), len(sol_keys)
        graph = coo_matrix((np.ones(len(good)), (il, sol + ni)), shape=(ni + ns, ni + ns))
        count, labels = connected_components(graph, directed=False)
        eligible = [i for i in range(count) if np.sum(labels[:ni] == i) >= 2]
        if eligible:
            component = max(eligible, key=lambda i: (np.sum(labels[:ni] == i), -i))
            keep = labels[il] == component
            selected = good.loc[keep]
            il2, keys2 = pd.factorize(pd.MultiIndex.from_frame(selected[["cation", "anion"]]), sort=True)
            sol2, solkeys2 = pd.factorize(selected.solute, sort=True)
            # Solve full indicator model, then impose mean(solute effect)=0.
            design = np.zeros((len(selected), len(keys2) + len(solkeys2)))
            design[np.arange(len(selected)), il2] = 1
            design[np.arange(len(selected)), len(keys2) + sol2] = 1
            beta = np.linalg.lstsq(design, selected.signature.to_numpy(float), rcond=None)[0]
            shift = beta[len(keys2):].mean()
            effects = beta[:len(keys2)] + shift
            solute_effects = beta[len(keys2):] - shift
            units.loc[selected.index, "solute_effect"] = solute_effects[sol2]
            units.loc[selected.index, "il_effect"] = effects[il2]
            units.loc[selected.index, "additive_residual"] = selected.signature.to_numpy(float) - effects[il2] - solute_effects[sol2]
            units.loc[selected.index, "retained_component"] = component
            residual = selected.signature.to_numpy(float) - design @ beta
            rank = len(keys2) + len(solkeys2) - 1
            df = len(selected) - rank
            # Linear contrast of the minimum-norm coefficients for the constrained IL effect.
            covariance = np.linalg.pinv(design.T @ design)
            for i, key in enumerate(keys2):
                contrast = np.zeros(len(beta)); contrast[i] = 1
                contrast[len(keys2):] = 1 / len(solkeys2)
                se = float(np.sqrt(max(0, contrast @ covariance @ contrast * np.sum(residual ** 2) / df))) if df > 0 else np.nan
                subset = selected.loc[il2 == i]
                retained[tuple(key)] = {"signature": float(effects[i]), "status": "ok",
                    "uncertainty": se, "residual_df": df, "n_solute_units": len(subset),
                    "reference_mismatch": bool(subset.reference_mismatch.any()),
                    "extrapolated": bool(subset.extrapolated.any()),
                    "actual_reference": json_text(sorted(set(subset.actual_reference))),
                    "extrapolation_distance": json_text(sorted(set(subset.extrapolation_distance))),
                    "unit_signature_kinds": json_text(sorted(set(subset.signature_kind))),
                    "mixed_unit_signature_kinds": subset.signature_kind.nunique() > 1,
                    "fit_quality": json_text({"solute_model_rmse": float(np.sqrt(np.mean(residual ** 2))),
                                              "n_components": count, "component": component,
                                              "uncertainty_scope": "conditional_on_unit_signatures"})}
    for key in all_keys:
        group = frame.loc[(frame.cation == key[0]) & (frame.anion == key[1])]
        base = {"cation": key[0], "anion": key[1], "n_observations": len(group),
                "signature": np.nan, "status": "unidentifiable_solute_component",
                "fit_formula": node["formula"], "fit_method": "equal_weight_il_solute_additive",
                "signature_kind": "solute_controlled", "parameters": "{}", "fit_quality": "{}",
                "requested_reference": json_text(refs), "actual_reference": "{}",
                "reference_mismatch": False, "extrapolated": False, "extrapolation_distance": "{}",
                "uncertainty": np.nan, "residual_df": 0,
                "unit_signature_kinds": "[]", "mixed_unit_signature_kinds": False}
        base.update(retained.get(key, {}))
        output.append(base)
    return output, units


def build_signatures(nodes, config):
    inventory, review, signatures, units = [], [], [], []
    for node in nodes:
        print(f"Signatures: {node['node_id']}", file=sys.stderr, flush=True)
        record = {k: node[k] for k in ("node_id", "source_dataset", "target_property", "stage",
                                      "system_type", "excluded_reason", "formula", "condition_columns")}
        record["source_files"] = json_text([str(p) for p in node["files"]])
        record.update(n_rows=0, n_valid_observations=0, n_observation_systems=0, n_valid_signatures=0)
        if not node["excluded_reason"]:
            frame = valid_frame(node)
            record.update(n_rows=len(node["frame"]), n_valid_observations=len(frame),
                          n_observation_systems=frame.groupby(["cation", "anion"]).ngroups)
            coverage = {}
            for col in filter(None, node["condition_columns"].split(";")):
                if col not in frame:
                    coverage[col] = {"missing_column": True}
                    continue
                v = pd.to_numeric(frame[col], errors="coerce")
                coverage[col] = {"n_missing": int((~np.isfinite(v)).sum()),
                                 "min": float(v.min()) if np.isfinite(v.min()) else None,
                                 "max": float(v.max()) if np.isfinite(v.max()) else None,
                                 "n_varied_systems": int((frame.groupby(["cation", "anion"])[col].nunique() > 1).sum())}
            if node["system_type"] == "il_solute" and node["formula"].startswith("solute_"):
                rows, unit_frame = solute_signatures(frame, node, config["references"])
                if not unit_frame.empty:
                    unit_frame["node_id"] = node["node_id"]
                    units.extend(unit_frame.to_dict("records"))
            else:
                rows = [dict(zip(["cation", "anion"], keys)) | fit_signature(group, node["formula"], config["references"])
                        for keys, group in frame.groupby(["cation", "anion"], sort=True)]
            for row in rows:
                row.update(node_id=node["node_id"], dataset=node["source_dataset"],
                           target_property=node["target_property"], stage=node["stage"],
                           formula_version=config["version"], source_files=record["source_files"])
                row["system"] = json_text([row["cation"], row["anion"]])
                signatures.append(row)
            record["n_valid_signatures"] = sum(np.isfinite(r["signature"]) for r in rows)
            statuses = pd.Series([r["status"] for r in rows], dtype=str).value_counts().to_dict()
            review.append({"node_id": node["node_id"], "formula": node["formula"],
                           "approval_status": "pending_formula" if node["formula"] == "pending_formula" else "approved",
                           "condition_coverage": json_text(coverage), "signature_status_counts": json_text(statuses),
                           "n_single_observation_systems": int((frame.groupby(["cation", "anion"]).size() == 1).sum())})
        inventory.append(record)
    signature_frame = pd.DataFrame(signatures)
    if signature_frame.empty:
        signature_frame = pd.DataFrame(columns=["node_id", "system", "signature", "signature_kind", "reference_mismatch", "actual_reference", "extrapolated"])
    return pd.DataFrame(inventory), pd.DataFrame(review), signature_frame, pd.DataFrame(units)


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def quantile_labels(values, bins):
    cuts = np.quantile(values, np.linspace(0, 1, bins + 1))
    if len(np.unique(cuts)) != bins + 1:
        return None
    labels = np.searchsorted(cuts[1:-1], values, side="right")
    return labels if len(np.unique(labels)) == bins else None


def ksg_mi(x, y, k=3):
    xy = np.column_stack([x, y])
    if len(np.unique(xy, axis=0)) != len(xy):
        return np.nan
    radii = cKDTree(xy).query(xy, k=k + 1, p=np.inf)[0][:, -1]
    if (radii <= 0).any():
        return np.nan
    radii = np.nextafter(radii, 0)
    nx = cKDTree(x[:, None]).query_ball_point(x[:, None], radii, p=np.inf, return_length=True) - 1
    ny = cKDTree(y[:, None]).query_ball_point(y[:, None], radii, p=np.inf, return_length=True) - 1
    return float(digamma(k) + digamma(len(x)) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def distance_correlation(x, y):
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a -= a.mean(axis=0)[None, :] + a.mean(axis=1)[:, None] - a.mean()
    b -= b.mean(axis=0)[None, :] + b.mean(axis=1)[:, None] - b.mean()
    denominator = np.sqrt(np.mean(a * a) * np.mean(b * b))
    return float(np.sqrt(max(0, np.mean(a * b) / denominator)))


def metric_values(x, y, thresholds=(None, None)):
    n = len(x)
    values = dict.fromkeys(METRICS, np.nan)
    reasons = dict.fromkeys(METRICS, "insufficient_samples")
    if n == 0:
        return values, dict.fromkeys(METRICS, "no_shared_signatures")
    constant = np.std(x) == 0 or np.std(y) == 0
    if n >= 2:
        lx = x > (np.median(x) if thresholds[0] is None else thresholds[0])
        ly = y > (np.median(y) if thresholds[1] is None else thresholds[1])
        values["binary_mi"] = float(mutual_info_score(lx, ly)); reasons["binary_mi"] = "ok"
        h = entropy(ly)
        reasons["binary_i_over_h"] = "zero_target_entropy" if h == 0 else "ok"
        if h:
            values["binary_i_over_h"] = values["binary_mi"] / h
    for metric, minimum in (("spearman", 3), ("distance_correlation", 5), ("continuous_mi", 20)):
        if n >= minimum:
            reasons[metric] = "constant_signature" if constant else "ok"
            if not constant:
                if metric == "spearman":
                    values[metric] = float(spearmanr(x, y).statistic)
                else:
                    zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
                    values[metric] = distance_correlation(zx, zy) if metric == "distance_correlation" else ksg_mi(zx, zy)
                    if not np.isfinite(values[metric]):
                        reasons[metric] = "degenerate_duplicate_coordinates"
    if n >= 30:
        bins = 5 if n >= 150 else 4 if n >= 80 else 3
        lx, ly = quantile_labels(x, bins), quantile_labels(y, bins)
        if lx is None or ly is None:
            reasons["multiclass_mi"] = "degenerate_quantile_bins"
        else:
            values["multiclass_mi"] = float(mutual_info_score(lx, ly)); reasons["multiclass_mi"] = "ok"
    return values, reasons


def cv_nmae(x, y, seed):
    if len(x) < 20:
        return np.nan, "insufficient_samples"
    if len(np.unique(x)) < 4:
        return np.nan, "insufficient_distinct_input"
    model_error, baseline_error = 0., 0.
    for train, test in KFold(5, shuffle=True, random_state=seed).split(x):
        if len(np.unique(x[train])) < 4:
            return np.nan, "insufficient_fold_input"
        model = make_pipeline(StandardScaler(), SplineTransformer(n_knots=3, degree=3), Ridge(alpha=1))
        model.fit(x[train, None], y[train])
        prediction = model.predict(x[test, None])
        if not np.isfinite(prediction).all():
            return np.nan, "nonfinite_fold_prediction"
        model_error += np.abs(prediction - y[test]).sum()
        baseline_error += np.abs(np.median(y[train]) - y[test]).sum()
    if baseline_error == 0:
        return np.nan, "zero_baseline_error"
    return float(model_error / baseline_error), "ok"


def stable_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256(json_text([seed, *parts]).encode()).digest()[:4], "little")


def confidence_rows(x, y, observed, thresholds, seed, settings):
    rng = np.random.default_rng(seed)
    metrics = [m for m in METRICS if m != "predictability_cv_nmae" and np.isfinite(observed[m])]
    boot, perm = {m: [] for m in metrics}, {m: [] for m in metrics}
    for _ in range(settings["bootstrap"] if metrics else 0):
        idx = rng.integers(0, len(x), len(x))
        vals, _ = metric_values(x[idx], y[idx], thresholds)
        for m in metrics:
            if np.isfinite(vals[m]):
                boot[m].append(vals[m])
    for _ in range(settings["permutation"] if metrics else 0):
        vals, _ = metric_values(x, rng.permutation(y), thresholds)
        for m in metrics:
            if np.isfinite(vals[m]):
                perm[m].append(vals[m])
    result = []
    for m in metrics:
        interval = np.quantile(boot[m], [.025, .975]) if len(boot[m]) >= settings["minimum_bootstrap_valid"] else [np.nan, np.nan]
        transform = np.abs if m == "spearman" else np.asarray
        exceed = np.sum(transform(perm[m]) >= (abs(observed[m]) if m == "spearman" else observed[m]))
        result.append({"metric": m, "interval_kind": "paired_system_percentile_bootstrap",
                       "lower": interval[0], "upper": interval[1], "bootstrap_valid": len(boot[m]),
                       "bootstrap_failed": settings["bootstrap"] - len(boot[m]),
                       "permutation_valid": len(perm[m]), "permutation_failed": settings["permutation"] - len(perm[m]),
                       "permutation_p": (1 + exceed) / (1 + len(perm[m])) if perm[m] else np.nan,
                       "status": "ok" if np.isfinite(interval).all() else "insufficient_valid_bootstrap"})
    if np.isfinite(observed["predictability_cv_nmae"]):
        repetitions = [cv_nmae(x, y, stable_seed(seed, "cv", i))[0] for i in range(settings["cv_repeats"])]
        valid = [v for v in repetitions if np.isfinite(v)]
        null = [cv_nmae(x, rng.permutation(y), seed)[0] for _ in range(settings["cv_permutation"])]
        null = [v for v in null if np.isfinite(v)]
        interval = np.quantile(valid, [.025, .975]) if len(valid) >= 2 else [np.nan, np.nan]
        result.append({"metric": "predictability_cv_nmae", "interval_kind": "cv_split_stability_not_parameter_ci",
                       "lower": interval[0], "upper": interval[1], "cv_valid": len(valid),
                       "cv_failed": settings["cv_repeats"] - len(valid), "permutation_valid": len(null),
                       "permutation_failed": settings["cv_permutation"] - len(null),
                       "permutation_p": (1 + np.sum(np.asarray(null) <= observed["predictability_cv_nmae"])) / (1 + len(null)) if null else np.nan,
                       "status": "ok" if np.isfinite(interval).all() else "insufficient_valid_cv"})
    missing = set(METRICS) - {r["metric"] for r in result}
    result.extend({"metric": m, "status": "metric_unavailable", "lower": np.nan, "upper": np.nan} for m in sorted(missing))
    return result


def write_csv(frame, path):
    frame.to_csv(path, index=False, na_rep="NA")


def compute_graphs(nodes, signatures, config, out, seed):
    included = [n for n in nodes if not n["excluded_reason"]]
    groups = {s: [n for n in included if n["stage"] == s] for s in (2, 3)}
    data = {}
    for node in included:
        rows = signatures.loc[signatures.node_id == node["node_id"]]
        valid = rows.loc[np.isfinite(rows.signature)].set_index("system")
        data[node["node_id"]] = (rows, valid)
    for name, sources in (("G_EE", groups[3]), ("G_SE", groups[2])):
        directory = out / name; directory.mkdir()
        source_ids = [n["node_id"] for n in sources]; target_ids = [n["node_id"] for n in groups[3]]
        matrices = {m: pd.DataFrame(np.nan, index=source_ids, columns=target_ids)
                    for m in ("n_shared", "n_observation_shared", *METRICS)}
        pairs, confidence, reasons = [], [], []
        work = [(a, b) for i, a in enumerate(sources) for j, b in enumerate(groups[3]) if name != "G_EE" or i < j]
        if name == "G_EE":
            for node in sources:
                rows, valid = data[node["node_id"]]
                matrices["n_shared"].loc[node["node_id"], node["node_id"]] = len(valid)
                matrices["n_observation_shared"].loc[node["node_id"], node["node_id"]] = len(rows)
                reasons.extend({"source": node["node_id"], "target": node["node_id"], "metric": m,
                                "reason": "diagonal_not_computed"} for m in METRICS)
        for number, (a, b) in enumerate(work, 1):
            aid, bid = a["node_id"], b["node_id"]
            print(f"{name} {number}/{len(work)}: {aid} -> {bid}", file=sys.stderr, flush=True)
            ar, av = data[aid]; br, bv = data[bid]
            shared = sorted(set(av.index) & set(bv.index))
            nobs = len(set(ar.system) & set(br.system))
            x, y = av.loc[shared].signature.to_numpy(float), bv.loc[shared].signature.to_numpy(float)
            thresholds = (config["binary_thresholds"].get(aid), config["binary_thresholds"].get(bid))
            pair_seed = stable_seed(seed, name, aid, bid)
            values, statuses = metric_values(x, y, thresholds)
            values["predictability_cv_nmae"], statuses["predictability_cv_nmae"] = cv_nmae(x, y, pair_seed)
            if not shared:
                statuses["predictability_cv_nmae"] = "no_shared_signatures"
            def store(source, target, vals, states, source_valid, target_valid):
                matrices["n_shared"].loc[source, target] = len(shared)
                matrices["n_observation_shared"].loc[source, target] = nobs
                kinds = sorted(set(source_valid.loc[shared].signature_kind) | set(target_valid.loc[shared].signature_kind))
                mismatch = bool(source_valid.loc[shared].reference_mismatch.any() or target_valid.loc[shared].reference_mismatch.any())
                actual = set(source_valid.loc[shared].actual_reference) | set(target_valid.loc[shared].actual_reference)
                pair = {"source": source, "target": target, "n_shared": len(shared), "n_observation_shared": nobs,
                        "shared_systems": json_text(shared), "signature_kinds": json_text(kinds),
                        "mixed_signature_kinds": len(kinds) > 1 or bool(
                            source_valid.loc[shared].get("mixed_unit_signature_kinds", pd.Series(dtype=bool)).fillna(False).any()
                            or target_valid.loc[shared].get("mixed_unit_signature_kinds", pd.Series(dtype=bool)).fillna(False).any()),
                        "reference_mismatch": mismatch,
                        "heterogeneous_actual_references": len(actual) > 1,
                        "extrapolated": bool(source_valid.loc[shared].extrapolated.any() or target_valid.loc[shared].extrapolated.any()),
                        "source_pending_formula": a["formula"] == "pending_formula" if source == aid else b["formula"] == "pending_formula",
                        "target_pending_formula": b["formula"] == "pending_formula" if target == bid else a["formula"] == "pending_formula"}
                for m in METRICS:
                    matrices[m].loc[source, target] = vals[m]
                    pair[m] = vals[m]; pair[m + "_status"] = states[m]
                    if not np.isfinite(vals[m]):
                        reasons.append({"source": source, "target": target, "metric": m, "reason": states[m]})
                pairs.append(pair)
            store(aid, bid, values, statuses, av, bv)
            conf = confidence_rows(x, y, values, thresholds, pair_seed, config["stability"])
            confidence.extend(dict(source=aid, target=bid, **r) for r in conf)
            if name == "G_EE":
                reverse = dict(values); reverse_states = dict(statuses)
                reverse_binary, rs = metric_values(y, x, thresholds[::-1])
                reverse["binary_i_over_h"] = reverse_binary["binary_i_over_h"]
                reverse_states["binary_i_over_h"] = rs["binary_i_over_h"]
                reverse["predictability_cv_nmae"], reverse_states["predictability_cv_nmae"] = cv_nmae(y, x, pair_seed)
                if not shared:
                    reverse_states["predictability_cv_nmae"] = "no_shared_signatures"
                store(bid, aid, reverse, reverse_states, bv, av)
                # Symmetric estimates and uncertainty are mirrored exactly.
                reverse_conf = confidence_rows(y, x, {m: reverse[m] if m not in SYMMETRIC else np.nan for m in METRICS},
                                               thresholds[::-1], pair_seed, config["stability"])
                confidence.extend(dict(source=bid, target=aid, **r) for r in conf if r["metric"] in SYMMETRIC)
                confidence.extend(dict(source=bid, target=aid, **r) for r in reverse_conf if r["metric"] not in SYMMETRIC)
        for m, matrix in matrices.items():
            matrix.rename_axis("source_node").to_csv(directory / f"{m}.csv", na_rep="NA")
        write_csv(pd.DataFrame(pairs), directory / "pairs.csv")
        write_csv(pd.DataFrame(confidence), directory / "confidence.csv")
        write_csv(pd.DataFrame(reasons), directory / "na_reasons.csv")


def run(command, input_root, output_dir, config_path=DEFAULT_CONFIG, seed=42):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is nonempty; refusing overwrite: {out}")
    input_root = Path(input_root).resolve()
    out = out.resolve()
    if out == input_root or input_root in out.parents:
        raise ValueError("Output directory must be outside training_splits")
    config = load_config(config_path)
    nodes, inputs = discover(input_root, config)
    inventory, review, signatures, units = build_signatures(nodes, config)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(inventory, out / "dataset_inventory.csv")
    write_csv(review, out / "formula_review.csv")
    write_csv(signatures, out / "system_signatures.csv")
    write_csv(units, out / "solute_unit_signatures.csv")
    manifest = {"status": "running", "command": command, "seed": seed, "input_root": str(input_root),
                "input_hashes": inputs, "config": config, "config_sha256": file_hash(config_path),
                "script_sha256": file_hash(__file__),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "confidence_scope": "conditional_on_signatures; excludes full signature estimation uncertainty",
                "cv_interval": "split stability, not parameter confidence interval",
                "arrow_meaning": "prediction, not causation"}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    try:
        if command == "compute":
            compute_graphs(nodes, signatures, config, out, seed)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        path.write_text(json.dumps(manifest, indent=2))
        raise
    manifest["status"] = "complete"
    manifest["output_hashes"] = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*.csv"))}
    path.write_text(json.dumps(manifest, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "compute"):
        p = sub.add_parser(name)
        p.add_argument("--input-root", type=Path, default=ROOT / "data/training_splits")
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.command, args.input_root, args.output_dir, args.config, args.seed)


if __name__ == "__main__":
    main()
