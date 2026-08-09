from __future__ import annotations

from pathlib import Path
import json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
from scipy.stats import spearmanr

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"C:\Users\sadia\Downloads\RaceCast_Research_Starter\racecast_research_starter\data\raw")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLE = OUT / "tables"
DATA = OUT / "data"
MODEL = OUT / "models"
for p in [OUT, FIG, TABLE, DATA, MODEL]:
    p.mkdir(parents=True, exist_ok=True)

DIST_STEP = 5.0
MIN_SPEED_MS = 5.0
CORNER_PROMINENCE_KPH = 12.0
MIN_CORNER_SPACING_M = 180.0

FEATURE_GROUPS = {
    "braking": [
        "brake_fraction_delta",
        "brake_start_rel_delta_m",
        "brake_length_delta_m",
    ],
    "corner_speed": [
        "entry_speed_delta_kph",
        "min_speed_delta_kph",
        "mean_speed_delta_kph",
    ],
    "throttle": [
        "throttle_mean_delta_pct",
        "full_throttle_fraction_delta",
        "throttle_reapply_rel_delta_m",
    ],
    "exit_acceleration": [
        "exit_speed_delta_kph",
        "exit_accel_proxy_delta",
    ],
    "powertrain": [
        "rpm_mean_delta",
        "gear_mean_delta",
        "gear_changes_delta",
        "drs_fraction_delta",
    ],
}

BASE_FEATURES = [x for g in FEATURE_GROUPS.values() for x in g] + [
    "segment_length_m",
    "reference_segment_time_s",
]

def clean_driver(x):
    return str(x).strip().upper()

def read_event_dirs():
    dirs = sorted([p for p in RAW.glob("2024_*_q") if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(
            f"No event folders found at {RAW}. "
            "Copy this RaceCast_Research_Analysis folder next to racecast_research_starter."
        )
    return dirs

def load_telemetry(event_dir: Path, driver: str, lap_number: int):
    fp = event_dir / "telemetry" / f"{driver}_lap_{int(lap_number)}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    needed = ["Distance", "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values("Distance").drop_duplicates("Distance")
    df = df[np.isfinite(df["Distance"])]
    if len(df) < 30:
        return None
    return df

def resample(df: pd.DataFrame, max_distance=None):
    max_d = float(df["Distance"].max()) if max_distance is None else min(float(df["Distance"].max()), max_distance)
    grid = np.arange(0.0, max_d, DIST_STEP)
    if len(grid) < 50:
        return None
    out = pd.DataFrame({"Distance": grid})
    continuous = ["Speed", "Throttle", "RPM", "nGear", "DRS"]
    for c in continuous:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        dist = df["Distance"].to_numpy(float)
        good = np.isfinite(vals) & np.isfinite(dist)
        if good.sum() >= 2:
            out[c] = np.interp(grid, dist[good], vals[good])
        else:
            out[c] = np.nan
    brake = pd.to_numeric(df["Brake"], errors="coerce").fillna(0).astype(float).to_numpy()
    dist = df["Distance"].to_numpy(float)
    idx = np.searchsorted(dist, grid, side="left")
    idx = np.clip(idx, 0, len(dist)-1)
    out["Brake"] = brake[idx] > 0.5
    return out

def cumulative_time(df):
    v = np.maximum(pd.to_numeric(df["Speed"], errors="coerce").to_numpy(float) / 3.6, MIN_SPEED_MS)
    dt = DIST_STEP / v
    return np.cumsum(dt)

def smooth_speed(speed):
    arr = np.asarray(speed, float)
    n = len(arr)
    win = min(31, n if n % 2 == 1 else n-1)
    if win < 7:
        return arr
    if win % 2 == 0:
        win -= 1
    return savgol_filter(arr, window_length=win, polyorder=2, mode="interp")

def detect_segments(ref):
    speed = smooth_speed(ref["Speed"].to_numpy(float))
    min_spacing_samples = max(5, int(MIN_CORNER_SPACING_M / DIST_STEP))
    peaks, props = find_peaks(-speed, prominence=CORNER_PROMINENCE_KPH, distance=min_spacing_samples)
    if len(peaks) < 4:
        # fallback: broader spacing / lower prominence
        peaks, props = find_peaks(-speed, prominence=8.0, distance=max(4, int(140/DIST_STEP)))
    if len(peaks) == 0:
        # fallback equal segments so analysis still runs
        peaks = np.linspace(int(len(ref)*0.08), int(len(ref)*0.92), 10).astype(int)

    boundaries = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        boundaries.append(int((a+b)//2))
    boundaries.append(len(ref)-1)

    segments = []
    for i, center in enumerate(peaks):
        s = boundaries[i]
        e = boundaries[i+1]
        if e - s < 8:
            continue
        segments.append((i+1, s, e, int(center)))
    return segments

def first_true_rel(brake, dist, start_dist):
    idx = np.where(np.asarray(brake, dtype=bool))[0]
    if not len(idx):
        return np.nan
    return float(dist[idx[0]] - start_dist)

def brake_len(brake):
    return float(np.sum(np.asarray(brake, dtype=bool)) * DIST_STEP)

def throttle_reapply_rel(throttle, dist, min_idx_local, start_dist):
    t = np.asarray(throttle, float)
    after = np.arange(len(t)) >= int(min_idx_local)
    idx = np.where(after & (t >= 90.0))[0]
    if not len(idx):
        return np.nan
    return float(dist[idx[0]] - start_dist)

def segment_metrics(df, s, e):
    seg = df.iloc[s:e+1].copy()
    if len(seg) < 5:
        return None
    speed = seg["Speed"].to_numpy(float)
    throttle = seg["Throttle"].to_numpy(float)
    brake = seg["Brake"].to_numpy(bool)
    rpm = seg["RPM"].to_numpy(float)
    gear = seg["nGear"].to_numpy(float)
    drs = seg["DRS"].to_numpy(float)
    dist = seg["Distance"].to_numpy(float)
    start_d = float(dist[0])
    min_i = int(np.nanargmin(speed))
    exit_tail = speed[max(0, len(speed)-3):]
    entry_head = speed[:min(3,len(speed))]
    min_speed = float(np.nanmin(speed))
    exit_speed = float(np.nanmean(exit_tail))
    entry_speed = float(np.nanmean(entry_head))
    distance_after_min = max(DIST_STEP, float(dist[-1] - dist[min_i]))
    accel_proxy = (exit_speed - min_speed) / distance_after_min
    gear_clean = gear[np.isfinite(gear)]
    gear_changes = int(np.sum(np.diff(np.round(gear_clean)) != 0)) if len(gear_clean) > 1 else 0

    v_ms = np.maximum(speed / 3.6, MIN_SPEED_MS)
    seg_time = float(np.sum(DIST_STEP / v_ms))

    return {
        "segment_time_s": seg_time,
        "entry_speed_kph": entry_speed,
        "min_speed_kph": min_speed,
        "mean_speed_kph": float(np.nanmean(speed)),
        "exit_speed_kph": exit_speed,
        "brake_fraction": float(np.mean(brake)),
        "brake_start_rel_m": first_true_rel(brake, dist, start_d),
        "brake_length_m": brake_len(brake),
        "throttle_mean_pct": float(np.nanmean(throttle)),
        "full_throttle_fraction": float(np.mean(throttle >= 95.0)),
        "throttle_reapply_rel_m": throttle_reapply_rel(throttle, dist, min_i, start_d),
        "exit_accel_proxy": float(accel_proxy),
        "rpm_mean": float(np.nanmean(rpm)),
        "gear_mean": float(np.nanmean(gear)),
        "gear_changes": gear_changes,
        "drs_fraction": float(np.mean(drs >= 10)) if np.isfinite(drs).any() else np.nan,
        "segment_length_m": float(dist[-1] - dist[0]),
    }

def safe_delta(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)

def build_dataset():
    rows = []
    delta_rows = []
    pair_rows = []
    event_stats = []

    for event_dir in read_event_dirs():
        laps = pd.read_csv(event_dir/"laps.csv")
        laps["driver"] = laps["driver"].map(clean_driver)
        laps = laps.sort_values("lap_time_s")
        event = str(laps["event"].iloc[0])
        used_pairs = 0

        for driver, g in laps.groupby("driver"):
            available = []
            for _, lap in g.sort_values("lap_time_s").iterrows():
                tel = load_telemetry(event_dir, driver, int(lap["lap_number"]))
                if tel is not None:
                    available.append((lap, tel))
            if len(available) < 2:
                continue

            ref_lap, ref_raw = min(available, key=lambda x: float(x[0]["lap_time_s"]))
            ref = resample(ref_raw)
            if ref is None:
                continue
            segments = detect_segments(ref)

            for comp_lap, comp_raw in available:
                if int(comp_lap["lap_number"]) == int(ref_lap["lap_number"]):
                    continue
                max_d = min(float(ref_raw["Distance"].max()), float(comp_raw["Distance"].max()))
                ref_al = resample(ref_raw, max_d)
                comp = resample(comp_raw, max_d)
                if ref_al is None or comp is None:
                    continue
                n = min(len(ref_al), len(comp))
                ref_al, comp = ref_al.iloc[:n].reset_index(drop=True), comp.iloc[:n].reset_index(drop=True)

                # Re-detect based on aligned ref to avoid overrun
                segs = detect_segments(ref_al)

                ref_ct = cumulative_time(ref_al)
                cmp_ct = cumulative_time(comp)
                cum_delta = cmp_ct - ref_ct

                pair_id = f"{event}_{driver}_L{int(comp_lap['lap_number'])}_vs_L{int(ref_lap['lap_number'])}"
                pair_rows.append({
                    "pair_id": pair_id, "event": event, "driver": driver,
                    "comparison_lap": int(comp_lap["lap_number"]),
                    "reference_lap": int(ref_lap["lap_number"]),
                    "comparison_lap_time_s": float(comp_lap["lap_time_s"]),
                    "reference_lap_time_s": float(ref_lap["lap_time_s"]),
                    "official_lap_delta_s": float(comp_lap["lap_time_s"] - ref_lap["lap_time_s"]),
                    "integrated_delta_s": float(cum_delta[-1]),
                    "compound_comparison": comp_lap.get("compound"),
                    "compound_reference": ref_lap.get("compound"),
                    "tyre_life_comparison": comp_lap.get("tyre_life"),
                    "tyre_life_reference": ref_lap.get("tyre_life"),
                })
                used_pairs += 1

                # thin delta trace every 50 m for compact output
                for k in range(0, n, max(1, int(50/DIST_STEP))):
                    delta_rows.append({
                        "pair_id": pair_id,
                        "event": event,
                        "driver": driver,
                        "distance_m": float(ref_al.loc[k, "Distance"]),
                        "cumulative_delta_s": float(cum_delta[k]),
                        "reference_speed_kph": float(ref_al.loc[k, "Speed"]),
                        "comparison_speed_kph": float(comp.loc[k, "Speed"]),
                    })

                for seg_id, s, e, center in segs:
                    if e >= n:
                        continue
                    rm = segment_metrics(ref_al, s, e)
                    cm = segment_metrics(comp, s, e)
                    if rm is None or cm is None:
                        continue

                    row = {
                        "pair_id": pair_id,
                        "event": event,
                        "driver": driver,
                        "segment_id": int(seg_id),
                        "segment_start_m": float(ref_al.loc[s, "Distance"]),
                        "segment_end_m": float(ref_al.loc[e, "Distance"]),
                        "segment_center_m": float(ref_al.loc[min(center, n-1), "Distance"]),
                        "segment_length_m": rm["segment_length_m"],
                        "comparison_lap_number": int(comp_lap["lap_number"]),
                        "reference_lap_number": int(ref_lap["lap_number"]),
                        "comparison_lap_time_s": float(comp_lap["lap_time_s"]),
                        "reference_lap_time_s": float(ref_lap["lap_time_s"]),
                        "target_segment_time_delta_s": safe_delta(cm["segment_time_s"], rm["segment_time_s"]),
                        "reference_segment_time_s": rm["segment_time_s"],
                        "tyre_life_delta": safe_delta(comp_lap.get("tyre_life"), ref_lap.get("tyre_life")),
                    }
                    # deltas for engineered metrics
                    mappings = {
                        "entry_speed_delta_kph": "entry_speed_kph",
                        "min_speed_delta_kph": "min_speed_kph",
                        "mean_speed_delta_kph": "mean_speed_kph",
                        "exit_speed_delta_kph": "exit_speed_kph",
                        "brake_fraction_delta": "brake_fraction",
                        "brake_start_rel_delta_m": "brake_start_rel_m",
                        "brake_length_delta_m": "brake_length_m",
                        "throttle_mean_delta_pct": "throttle_mean_pct",
                        "full_throttle_fraction_delta": "full_throttle_fraction",
                        "throttle_reapply_rel_delta_m": "throttle_reapply_rel_m",
                        "exit_accel_proxy_delta": "exit_accel_proxy",
                        "rpm_mean_delta": "rpm_mean",
                        "gear_mean_delta": "gear_mean",
                        "gear_changes_delta": "gear_changes",
                        "drs_fraction_delta": "drs_fraction",
                    }
                    for out_name, base in mappings.items():
                        row[out_name] = safe_delta(cm[base], rm[base])
                    rows.append(row)

        event_stats.append({
            "event": event,
            "selected_laps": int(len(laps)),
            "drivers_selected": int(laps["driver"].nunique()),
            "comparison_pairs": int(used_pairs),
        })

    features = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    deltas = pd.DataFrame(delta_rows)
    stats = pd.DataFrame(event_stats)

    features.to_csv(DATA/"segment_features.csv", index=False)
    pairs.to_csv(DATA/"lap_pairs.csv", index=False)
    deltas.to_csv(DATA/"delta_time_50m.csv", index=False)
    stats.to_csv(TABLE/"processed_dataset_summary.csv", index=False)
    return features, pairs, deltas, stats

def make_eda(features, pairs):
    # Dataset overview
    summary = features.groupby("event").agg(
        segment_observations=("target_segment_time_delta_s","size"),
        drivers=("driver","nunique"),
        lap_pairs=("pair_id","nunique"),
        mean_segment_delta_s=("target_segment_time_delta_s","mean"),
        median_segment_delta_s=("target_segment_time_delta_s","median"),
    ).reset_index()
    summary.to_csv(TABLE/"segment_dataset_summary.csv", index=False)

    # Spearman correlations
    corr_rows = []
    for f in BASE_FEATURES:
        if f not in features.columns:
            continue
        sub = features[[f,"target_segment_time_delta_s"]].dropna()
        if len(sub) < 10 or sub[f].nunique() < 2:
            continue
        rho, p = spearmanr(sub[f], sub["target_segment_time_delta_s"])
        corr_rows.append({"feature":f, "n":len(sub), "spearman_rho":rho, "p_value":p})
    corr = pd.DataFrame(corr_rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(TABLE/"spearman_correlations.csv", index=False)

    # Figure: target distribution
    plt.figure(figsize=(9,5))
    vals = features["target_segment_time_delta_s"].dropna()
    plt.hist(vals, bins=60)
    plt.xlabel("Segment time delta (s): comparison - reference")
    plt.ylabel("Observations")
    plt.title("Distribution of Segment-Level Time Loss")
    plt.tight_layout()
    plt.savefig(FIG/"01_segment_delta_distribution.png", dpi=200)
    plt.close()

    # Figure: strongest correlations
    if len(corr):
        top = corr.head(10).iloc[::-1]
        plt.figure(figsize=(9,6))
        plt.barh(top["feature"], top["spearman_rho"])
        plt.xlabel("Spearman correlation with segment time delta")
        plt.title("Strongest Telemetry Feature Associations")
        plt.tight_layout()
        plt.savefig(FIG/"02_feature_correlations.png", dpi=200)
        plt.close()

    return summary, corr

def prep_xy(df, features=None):
    feats = list(BASE_FEATURES if features is None else features)
    existing = [c for c in feats if c in df.columns]
    work = df[existing + ["target_segment_time_delta_s","event"]].copy()
    for c in existing:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = work[c].replace([np.inf,-np.inf], np.nan)
        work[c] = work[c].fillna(work[c].median())
    work = work.dropna(subset=["target_segment_time_delta_s"])
    return work, existing

def models():
    return {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=350, max_leaf_nodes=31,
            l2_regularization=0.5, random_state=42
        ),
    }

def metrics(y, pred):
    return {
        "MAE": mean_absolute_error(y,pred),
        "RMSE": mean_squared_error(y,pred) ** 0.5,
        "R2": r2_score(y,pred),
    }

def leave_one_circuit_out(features):
    work, feats = prep_xy(features)
    results = []
    preds_all = []
    event_names = sorted(work["event"].unique())
    for name, model in models().items():
        for held in event_names:
            train = work[work["event"] != held]
            test = work[work["event"] == held]
            Xtr, ytr = train[feats], train["target_segment_time_delta_s"]
            Xte, yte = test[feats], test["target_segment_time_delta_s"]
            model.fit(Xtr,ytr)
            pred = model.predict(Xte)
            m = metrics(yte,pred)
            results.append({"model":name,"held_out_circuit":held,"n_test":len(test),**m})
            for idx, actual, pr in zip(test.index, yte, pred):
                preds_all.append({
                    "model":name, "held_out_circuit":held,
                    "row_index":int(idx), "actual_s":float(actual),
                    "predicted_s":float(pr), "error_s":float(pr-actual),
                })
    res = pd.DataFrame(results)
    res.to_csv(TABLE/"leave_one_circuit_out_results.csv", index=False)
    pd.DataFrame(preds_all).to_csv(DATA/"model_predictions.csv", index=False)

    overall = res.groupby("model").agg(
        MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
        RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
        R2_mean=("R2","mean"), R2_std=("R2","std"),
    ).reset_index().sort_values("MAE_mean")
    overall.to_csv(TABLE/"model_summary.csv", index=False)

    plt.figure(figsize=(9,5))
    x = np.arange(len(overall))
    plt.bar(x, overall["MAE_mean"])
    plt.xticks(x, overall["model"], rotation=15)
    plt.ylabel("Mean held-out-circuit MAE (s)")
    plt.title("Cross-Circuit Model Performance")
    plt.tight_layout()
    plt.savefig(FIG/"03_model_comparison_mae.png", dpi=200)
    plt.close()

    return res, overall, feats

def ablation(features, best_model_name):
    work, feats = prep_xy(features)
    events = sorted(work["event"].unique())
    configurations = {"Full model": feats}
    for group, cols in FEATURE_GROUPS.items():
        configurations[f"Without {group}"] = [f for f in feats if f not in cols]

    result_rows = []
    for cfg, cfg_feats in configurations.items():
        maes = []
        for held in events:
            tr, te = work[work.event != held], work[work.event == held]
            model = models()[best_model_name]
            model.fit(tr[cfg_feats], tr["target_segment_time_delta_s"])
            p = model.predict(te[cfg_feats])
            maes.append(mean_absolute_error(te["target_segment_time_delta_s"],p))
        result_rows.append({
            "configuration": cfg,
            "MAE_mean": float(np.mean(maes)),
            "MAE_std": float(np.std(maes, ddof=1)) if len(maes)>1 else 0.0,
        })
    abl = pd.DataFrame(result_rows)
    full = float(abl.loc[abl.configuration=="Full model","MAE_mean"].iloc[0])
    abl["delta_MAE_vs_full"] = abl["MAE_mean"] - full
    abl.to_csv(TABLE/"ablation_results.csv", index=False)

    plot = abl.sort_values("delta_MAE_vs_full")
    plt.figure(figsize=(9,5))
    plt.barh(plot["configuration"], plot["delta_MAE_vs_full"])
    plt.xlabel("Change in cross-circuit MAE vs full model (s)")
    plt.title("Feature-Group Ablation")
    plt.tight_layout()
    plt.savefig(FIG/"04_ablation.png", dpi=200)
    plt.close()
    return abl

def explain(features, best_model_name, feats):
    work, _ = prep_xy(features, feats)
    X = work[feats]
    y = work["target_segment_time_delta_s"]
    model = models()[best_model_name]
    model.fit(X,y)
    joblib.dump(model, MODEL/"best_model.joblib")

    # Always produce permutation importance (model-agnostic).
    perm = permutation_importance(
        model, X, y, scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=42, n_jobs=-1
    )
    imp = pd.DataFrame({
        "feature": feats,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(TABLE/"permutation_importance.csv", index=False)

    top = imp.head(12).iloc[::-1]
    plt.figure(figsize=(9,6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Increase in model utility when feature is preserved")
    plt.title("Model-Agnostic Permutation Importance")
    plt.tight_layout()
    plt.savefig(FIG/"05_permutation_importance.png", dpi=200)
    plt.close()

    shap_status = {"available": False}
    try:
        import shap
        # Keep sample bounded so this remains practical on a laptop.
        sample = X.sample(min(1500, len(X)), random_state=42)
        if best_model_name == "Random Forest":
            explainer = shap.TreeExplainer(model)
            sv = explainer(sample)
        elif best_model_name == "Gradient Boosting":
            # HistGradientBoosting support varies by SHAP version; generic Explainer is robust.
            background = shap.sample(X, min(200, len(X)), random_state=42)
            explainer = shap.Explainer(model.predict, background)
            sv = explainer(sample)
        else:
            explainer = shap.Explainer(model.predict, shap.sample(X, min(200,len(X)), random_state=42))
            sv = explainer(sample)
        shap.plots.beeswarm(sv, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(FIG/"06_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        shap_status = {"available": True, "samples": len(sample)}
    except Exception as e:
        shap_status = {"available": False, "error": str(e)}

    (OUT/"shap_status.json").write_text(json.dumps(shap_status, indent=2), encoding="utf-8")
    return imp, shap_status

def representative_delta_plot(deltas, pairs):
    if len(deltas)==0 or len(pairs)==0: return
    # largest official non-pathological delta under 5 seconds
    candidates = pairs[pairs["official_lap_delta_s"].between(0.05,5.0)]
    if len(candidates)==0: candidates = pairs
    p = candidates.sort_values("official_lap_delta_s", ascending=False).iloc[0]
    d = deltas[deltas.pair_id == p.pair_id]
    if len(d)==0: return

    plt.figure(figsize=(10,5))
    plt.plot(d["distance_m"], d["cumulative_delta_s"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Distance around circuit (m)")
    plt.ylabel("Cumulative time delta (s)")
    plt.title(f"Example Performance Delta — {p['event']} {p['driver']}")
    plt.tight_layout()
    plt.savefig(FIG/"07_example_cumulative_delta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(d["distance_m"], d["reference_speed_kph"], label="Reference")
    plt.plot(d["distance_m"], d["comparison_speed_kph"], label="Comparison")
    plt.xlabel("Distance around circuit (m)")
    plt.ylabel("Speed (km/h)")
    plt.title(f"Example Speed Comparison — {p['event']} {p['driver']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG/"08_example_speed_trace.png", dpi=200)
    plt.close()

def write_report(features, pairs, stats, corr, loco, summary, abl, imp, shap_status):
    best = summary.iloc[0]
    topcorr = corr.head(5)
    topimp = imp.head(5)

    lines = []
    lines += [
        "# RaceCast Research — Empirical Results",
        "",
        "## Dataset",
        f"- Segment observations: **{len(features):,}**",
        f"- Same-driver lap comparison pairs: **{pairs['pair_id'].nunique():,}**",
        f"- Circuits: **{features['event'].nunique()}**",
        f"- Drivers represented in segment dataset: **{features['driver'].nunique()}**",
        "",
        "### Per-circuit processed dataset",
        stats.to_markdown(index=False),
        "",
        "## Cross-circuit model comparison",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Best mean held-out-circuit MAE: **{best['model']} = {best['MAE_mean']:.6f} s**.",
        "",
        "### Leave-one-circuit-out results",
        loco.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Strongest Spearman associations",
        topcorr.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Feature-group ablation",
        abl.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Model-agnostic feature importance",
        topimp.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## SHAP",
        f"SHAP generated: **{shap_status.get('available')}**",
    ]
    if shap_status.get("error"):
        lines.append(f"SHAP note: `{shap_status['error']}`")
    lines += [
        "",
        "## Methodological caution",
        "These values describe predictive associations in public telemetry. "
        "They do not establish physical causality or reproduce proprietary team telemetry.",
    ]
    (OUT/"paper_results.md").write_text("\n".join(lines), encoding="utf-8")

    machine = {
        "n_segments": len(features),
        "n_pairs": int(pairs["pair_id"].nunique()),
        "n_circuits": int(features["event"].nunique()),
        "n_drivers": int(features["driver"].nunique()),
        "best_model": str(best["model"]),
        "best_mean_loco_mae_s": float(best["MAE_mean"]),
        "best_mean_loco_rmse_s": float(best["RMSE_mean"]),
        "best_mean_loco_r2": float(best["R2_mean"]),
        "shap": shap_status,
    }
    (OUT/"results_summary.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

def main():
    print("1/6 Building segment-level research dataset...")
    features, pairs, deltas, stats = build_dataset()
    print(f"   {len(features):,} segment observations from {pairs['pair_id'].nunique():,} lap pairs")

    print("2/6 Running exploratory/statistical analysis...")
    eda_summary, corr = make_eda(features,pairs)

    print("3/6 Running leave-one-circuit-out model evaluation...")
    loco, summary, feats = leave_one_circuit_out(features)
    best_name = str(summary.iloc[0]["model"])
    print("   Best model:", best_name)

    print("4/6 Running ablation study...")
    abl = ablation(features,best_name)

    print("5/6 Generating explainability outputs...")
    imp, shap_status = explain(features,best_name,feats)
    representative_delta_plot(deltas,pairs)

    print("6/6 Writing paper-ready results...")
    write_report(features,pairs,stats,corr,loco,summary,abl,imp,shap_status)
    print("\nDONE.")
    print("Outputs:", OUT)
    print("Upload the entire outputs folder (or zip it) back to ChatGPT.")

if __name__ == "__main__":
    main()

