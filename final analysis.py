# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:06:37 2026

@author: enric
"""

"""
SCRIPT B — Backtest Results Analyzer
===================================

This script reads the artifacts produced by your Script A (tested/ directory):
- *_WF_split.parquet
- *_WF_M.parquet
- *_WF_results.parquet
- *_CPCV_split.parquet
- *_CPCV_M.parquet
- *_CPCV_results.parquet
- *_CPCV_counts.parquet (optional but recommended)

and produces (in analyzed/ directory):
- per-(pair,scheme) metrics table: *_metrics.xlsx
- optional CSCV outputs: *_cscv_results.xlsx
- global summaries:
  - summary_best_strategies.xlsx
  - model_summary_global.xlsx
  - model_best_cases.xlsx
  - model_robustness.xlsx

Key points implemented correctly:
- WF evaluation excludes the initial pure-train region using first_train_size from WF splits.
- CPCV evaluation assumes M_cpcv is already OOS-aggregated OR uses CPCV counts if provided:
  * if C_cpcv exists: M_avg = sum_store / count
  * if not: we treat M_cpcv as already averaged (your Script A can choose).
- Sharpe/Sortino computed on DAILY resampled equity returns (frequency-aware).
- DSR computed with sr_trials = sharpes of ALL columns (all strategy variants).
- CSCV/PBO computed from the returns matrix (time splits), for selection-bias diagnostics.
"""

import os
import re
import glob
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import norm, skew, kurtosis, spearmanr, ks_2samp


# =========================
# CONFIG
# =========================

RF_RATE = 0.00               # annual risk-free
PERIODS_PER_YEAR = 252       # daily periods per year
EULER_GAMMA = 0.5772156649


# =========================
# SMALL HELPERS
# =========================

def _safe_sheet_name(name: str) -> str:
    """Excel sheet names must be <= 31 chars and cannot contain: : \ / ? * [ ]"""
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    if len(name) <= 31:
        return name
    return name[:31]

def _ensure_dt_index(M: pd.DataFrame) -> pd.DataFrame:
    """Ensure M index is DatetimeIndex; if not, try to parse it."""
    if isinstance(M.index, pd.DatetimeIndex):
        return M
    idx = pd.to_datetime(M.index, errors="coerce", utc=True)
    if idx.isna().any():
        raise ValueError("M index is not DatetimeIndex and cannot be parsed to datetime.")
    M = M.copy()
    M.index = idx
    return M

def _coerce_numeric_df(df: pd.DataFrame, fillna: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if fillna is not None:
        out = out.fillna(fillna)
    return out

def _as_numpy(x) -> np.ndarray:
    return np.asarray(x)

def _frac(cond: pd.Series) -> float:
    if len(cond) == 0:
        return float("nan")
    return float(cond.mean())

def _annualize_sharpe(mu_daily: float, sigma_daily: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if not np.isfinite(mu_daily) or not np.isfinite(sigma_daily) or sigma_daily <= 0:
        return 0.0
    return float(mu_daily / sigma_daily * np.sqrt(periods_per_year))

def _annualize_sortino(mu_daily: float, downside_dev: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if not np.isfinite(mu_daily) or not np.isfinite(downside_dev) or downside_dev <= 0:
        return 0.0
    return float(mu_daily / downside_dev * np.sqrt(periods_per_year))


# =========================
# SPLITS LOADING
# =========================

def load_splits_parquet(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expected format: DataFrame with columns ['train_idx', 'test_idx'] where each cell contains
    a list-like or JSON string or object.

    Your Script A currently does:
        splits_wf.to_parquet(..., index=False)
    So splits_wf must already be a DataFrame.
    """
    df = pd.read_parquet(path)

    if not {"train_idx", "test_idx"}.issubset(df.columns):
        raise ValueError(f"{path}: splits parquet must contain train_idx and test_idx columns.")

    def parse_cell(v):
        # already list/ndarray
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.array(v, dtype=int)
        # pandas might load as object holding list
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            try:
                return np.array(list(v), dtype=int)
            except Exception:
                pass
        # json string
        if isinstance(v, str):
            v = v.strip()
            # try json
            try:
                obj = json.loads(v)
                return np.array(obj, dtype=int)
            except Exception:
                # try python literal-ish
                v2 = v.strip("()[]")
                if v2 == "":
                    return np.array([], dtype=int)
                parts = [p.strip() for p in v2.split(",") if p.strip() != ""]
                return np.array([int(p) for p in parts], dtype=int)
        # fallback
        return np.array([], dtype=int)

    splits = []
    for _, row in df.iterrows():
        tr = parse_cell(row["train_idx"])
        te = parse_cell(row["test_idx"])
        splits.append((tr, te))
    return splits

def wf_first_train_size(splits: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """Compute the first train-only prefix length for WF: min test idx of fold 1."""
    if not splits:
        return 0
    _, test0 = splits[0]
    if len(test0) == 0:
        return 0
    return int(np.min(test0))

def cpcv_test_count_from_splits(n_samples: int, splits: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """How many times each event appears in test across CPCV folds."""
    C = np.zeros(n_samples, dtype=int)
    for _, te in splits:
        C[te] += 1
    return C


# =========================
# EQUITY / DAILY SERIES
# =========================

def build_daily_equity_returns(
    event_returns: np.ndarray,
    timestamps: np.ndarray,
    freq: str = "1D",
) -> pd.Series:
    """
    Convert event returns into daily returns:
    - event_returns aligned to timestamps
    - sum within each day (days without trades -> 0)
    """
    s = pd.Series(event_returns, index=pd.to_datetime(timestamps, utc=True))
    daily = s.resample(freq).sum()
    daily = daily.fillna(0.0)
    return daily

def sharpe_from_daily(
    daily_returns: pd.Series,
    rf_annual: float = RF_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    r = pd.Series(daily_returns).astype(float)
    if len(r) < 2:
        return 0.0
    rf_daily = rf_annual / periods_per_year
    excess = r - rf_daily
    mu = float(excess.mean())
    sd = float(excess.std(ddof=1))
    return _annualize_sharpe(mu, sd, periods_per_year)

def sortino_from_daily(
    daily_returns: pd.Series,
    rf_annual: float = RF_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    r = pd.Series(daily_returns).astype(float)
    if len(r) < 2:
        return 0.0
    rf_daily = rf_annual / periods_per_year
    excess = r - rf_daily
    downside = np.minimum(excess.values, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    mu = float(excess.mean())
    return _annualize_sortino(mu, dd, periods_per_year)


# =========================
# PSR / DSR
# =========================

def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    T: int,
    skew_val: float = 0.0,
    kurt_val: float = 3.0,
) -> float:
    """
    Bailey & López de Prado (2012)
    T = number of observations used for sr_hat (daily points)
    """
    if T <= 1:
        return float("nan")

    num = (sr_hat - sr_benchmark) * math.sqrt(T - 1)
    den = math.sqrt(max(1e-12, 1.0 - skew_val * sr_hat + ((kurt_val - 1.0) / 4.0) * (sr_hat ** 2)))
    return float(norm.cdf(num / den))

def deflated_sharpe_ratio(
    sr_hat: float,
    sr_trials: np.ndarray,
    T: int,
    skew_val: float,
    kurt_val: float,
    sr0: float = 0.0,
) -> float:
    """
    Bailey & López de Prado (2014)
    sr_trials = Sharpe values of all trials (all strategies tested)
    """
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    N = len(sr_trials)

    if T <= 1:
        return float("nan")

    # fallback to PSR if only one trial
    if N <= 1:
        return probabilistic_sharpe_ratio(sr_hat, sr0, T, skew_val, kurt_val)

    var_sr = float(np.var(sr_trials, ddof=1)) if N > 1 else 0.0
    sigma_sr = math.sqrt(max(1e-12, var_sr))

    # expected max of N standard normals (approx)
    z1 = norm.ppf(1.0 - 1.0 / N)
    z2 = norm.ppf(1.0 - 1.0 / (N * math.e))
    z_max = (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2

    sr_star = sr0 + sigma_sr * z_max
    return probabilistic_sharpe_ratio(sr_hat, sr_star, T, skew_val, kurt_val)


# =========================
# CSCV / PBO
# =========================

def split_subsets(M: pd.DataFrame, S: int, drop_remainder: bool = False) -> List[pd.DataFrame]:
    T = len(M)
    base = T // S
    rem = T % S
    subsets = []
    start = 0
    for i in range(S):
        extra = 1 if (i < rem and not drop_remainder) else 0
        size = base + extra
        stop = start + size
        subsets.append(M.iloc[start:stop].copy())
        start = stop
    return subsets

def cscv_test_generator(subsets: List[pd.DataFrame]):
    import itertools
    S = len(subsets)
    half = S // 2
    idx = list(range(S))
    for combo in itertools.combinations(idx, half):
        J_train = pd.concat([subsets[i] for i in combo], axis=0)
        complement = [i for i in idx if i not in combo]
        J_test = pd.concat([subsets[i] for i in complement], axis=0)
        yield combo, J_train, J_test

def stats_subsets(J_train: pd.DataFrame, J_test: pd.DataFrame):
    # use mean return as selection criterion (can be Sharpe too, but keep simple)
    perf_IS = J_train.mean()
    perf_OOS = J_test.mean()

    best_IS = perf_IS.idxmax()
    rank_OOS = perf_OOS.rank(ascending=False)[best_IS]
    relative_rank = float(rank_OOS / (len(perf_OOS) + 1))

    eps = 1e-8
    relative_rank = min(max(relative_rank, eps), 1.0 - eps)
    logit = float(np.log(relative_rank / (1.0 - relative_rank)))
    return best_IS, float(perf_IS[best_IS]), float(perf_OOS[best_IS]), relative_rank, logit

def cscv_pipeline(M: pd.DataFrame, S: int = 8, drop_remainder: bool = False, max_iter: Optional[int] = None) -> pd.DataFrame:
    subsets = split_subsets(M, S=S, drop_remainder=drop_remainder)
    rows = []
    for k, (combo, J_tr, J_te) in enumerate(cscv_test_generator(subsets)):
        best_IS, perf_IS, perf_OOS, rel_rank, logit = stats_subsets(J_tr, J_te)
        rows.append({
            "combo": str(combo),             # store as string -> parquet/excel safe
            "best_IS": best_IS,
            "perf_IS": perf_IS,
            "perf_OOS": perf_OOS,
            "relative_rank_OOS": rel_rank,
            "logit": logit
        })
        if max_iter is not None and (k + 1) >= max_iter:
            break
    return pd.DataFrame(rows)

def evaluate_cscv(df_results: pd.DataFrame) -> Dict[str, object]:
    metrics: Dict[str, object] = {}

    rr = df_results["relative_rank_OOS"].astype(float)
    metrics["PBO"] = float(np.mean(rr > 0.5))

    rho, _ = spearmanr(df_results["perf_IS"], df_results["perf_OOS"])
    metrics["perf_degradation"] = float(rho)

    metrics["prob_loss"] = float(np.mean(df_results["perf_OOS"] < 0))

    dist_oos = rr.values
    dist_uniform = np.random.uniform(0, 1, size=len(dist_oos))
    stat, pval = ks_2samp(dist_oos, dist_uniform)
    metrics["stochastic_dom"] = {"ks_stat": float(stat), "pval": float(pval)}
    return metrics


# =========================
# FILE PARSING
# =========================

@dataclass
class BacktestPaths:
    base: str
    wf_split: Optional[str]
    wf_M: Optional[str]
    wf_results: Optional[str]
    cpcv_split: Optional[str]
    cpcv_M: Optional[str]
    cpcv_results: Optional[str]
    cpcv_counts: Optional[str]

def discover_backtests(tested_dir: str) -> List[BacktestPaths]:
    """
    Discover basenames by looking for *_WF_M.parquet or *_CPCV_M.parquet.
    Assumes naming:
        {basename}_WF_split.parquet
        {basename}_WF_M.parquet
        {basename}_WF_results.parquet
        {basename}_CPCV_split.parquet
        {basename}_CPCV_M.parquet
        {basename}_CPCV_results.parquet
        {basename}_CPCV_counts.parquet (optional)
    """
    wf_files = glob.glob(os.path.join(tested_dir, "*_WF_M.parquet"))
    cpcv_files = glob.glob(os.path.join(tested_dir, "*_CPCV_M.parquet"))

    bases = set()
    for p in wf_files:
        bases.add(os.path.basename(p).replace("_WF_M.parquet", ""))
    for p in cpcv_files:
        bases.add(os.path.basename(p).replace("_CPCV_M.parquet", ""))

    out = []
    for b in sorted(bases):
        wf_split = os.path.join(tested_dir, f"{b}_WF_split.parquet")
        wf_M = os.path.join(tested_dir, f"{b}_WF_M.parquet")
        wf_results = os.path.join(tested_dir, f"{b}_WF_results.parquet")

        cpcv_split = os.path.join(tested_dir, f"{b}_CPCV_split.parquet")
        cpcv_M = os.path.join(tested_dir, f"{b}_CPCV_M.parquet")
        cpcv_results = os.path.join(tested_dir, f"{b}_CPCV_results.parquet")
        cpcv_counts = os.path.join(tested_dir, f"{b}_CPCV_counts.parquet")

        out.append(BacktestPaths(
            base=b,
            wf_split=wf_split if os.path.exists(wf_split) else None,
            wf_M=wf_M if os.path.exists(wf_M) else None,
            wf_results=wf_results if os.path.exists(wf_results) else None,
            cpcv_split=cpcv_split if os.path.exists(cpcv_split) else None,
            cpcv_M=cpcv_M if os.path.exists(cpcv_M) else None,
            cpcv_results=cpcv_results if os.path.exists(cpcv_results) else None,
            cpcv_counts=cpcv_counts if os.path.exists(cpcv_counts) else None,
        ))
    return out

def parse_pair_from_base(base: str) -> str:
    """
    Your basenames look like: AUD_USD_LABELED (or similar).
    We'll try to return the first 2 tokens joined by '_' if present, else base.
    Customize if your base format differs.
    """
    toks = base.split("_")
    if len(toks) >= 2:
        return "_".join(toks[:2])
    return base


# =========================
# CORE ANALYSIS PER SCHEME
# =========================

def compute_strategy_metrics(
    M_eval: pd.DataFrame,
    rf_annual: float = RF_RATE,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """
    Compute per-strategy metrics using daily resampled returns (frequency-aware),
    then compute DSR using sr_trials across all strategies.
    """
    M_eval = _ensure_dt_index(M_eval)
    M_eval = _coerce_numeric_df(M_eval, fillna=0.0)

    ts = M_eval.index.to_numpy()

    # precompute daily series + sharpe for all -> sr_trials
    daily_map: Dict[str, pd.Series] = {}
    sharpe_map: Dict[str, float] = {}

    for col in M_eval.columns:
        r = M_eval[col].to_numpy(dtype=float)
        daily = build_daily_equity_returns(r, ts, freq="1D")
        daily_map[col] = daily
        sharpe_map[col] = sharpe_from_daily(daily, rf_annual=rf_annual, periods_per_year=periods_per_year)

    sr_trials = np.array(list(sharpe_map.values()), dtype=float)

    rows = []
    for col in M_eval.columns:
        r = M_eval[col].to_numpy(dtype=float)
        traded = (r != 0.0)
        n_trades = int(traded.sum())

        daily = daily_map[col]
        sr = sharpe_map[col]
        sor = sortino_from_daily(daily, rf_annual=rf_annual, periods_per_year=periods_per_year)

        mu_evt = float(np.mean(r[traded])) if n_trades > 0 else 0.0
        sd_evt = float(np.std(r[traded], ddof=1)) if n_trades > 1 else 0.0
        hit = float(np.mean(r[traded] > 0)) if n_trades > 0 else 0.0

        # daily skew / kurtosis for PSR/DSR
        if len(daily) > 3:
            sk = float(skew(daily.values, bias=False))
            ku = float(kurtosis(daily.values, fisher=False, bias=False))  # total kurtosis
        else:
            sk = 0.0
            ku = 3.0
        T_daily = int(len(daily))

        dsr = deflated_sharpe_ratio(
            sr_hat=sr,
            sr_trials=sr_trials,
            T=T_daily,
            skew_val=sk,
            kurt_val=ku,
            sr0=0.0,
        )

        rows.append({
            "strategy": col,
            "n_trades": n_trades,
            "hit_rate": hit,
            "mean_evt_ret": mu_evt,
            "std_evt_ret": sd_evt,
            "sharpe": float(sr),
            "sortino": float(sor),
            "dsr": float(dsr) if np.isfinite(dsr) else np.nan,
            "T_daily": T_daily,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["dsr", "sharpe"], ascending=False, na_position="last")


def evaluate_one_scheme(
    base: str,
    scheme: str,                 # "WF" or "CPCV"
    M_path: str,
    split_path: Optional[str],
    counts_path: Optional[str],
    analyzed_dir: str,
    do_cscv: bool = True,
    cscv_S: int = 8,
    cscv_max_iter: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Returns:
      - metrics_df: per-strategy metrics table
      - scheme_summary: dict with PBO/perf_degradation/prob_loss/ks etc (if CSCV enabled)
    """
    M = pd.read_parquet(M_path)
    M = _ensure_dt_index(M)
    M = _coerce_numeric_df(M, fillna=0.0)

    # Determine evaluation region
    if scheme.upper() == "WF":
        if split_path is None:
            raise ValueError(f"{base} WF: split file missing.")
        splits = load_splits_parquet(split_path)
        first_train = wf_first_train_size(splits)
        if first_train > 0 and first_train < len(M):
            M_eval = M.iloc[first_train:].copy()
        else:
            M_eval = M.copy()
    elif scheme.upper() == "CPCV":
        # If counts exist and M is a SUM matrix -> average it.
        # If your Script A already writes M as averaged, counts are optional and M_eval=M.
        M_eval = M.copy()
        if counts_path is not None and os.path.exists(counts_path):
            C = pd.read_parquet(counts_path)
            # support both shapes:
            # - DataFrame with one column "count" aligned by row order
            # - Series-like
            if isinstance(C, pd.DataFrame):
                if "count" in C.columns:
                    cnt = C["count"].to_numpy(dtype=float)
                else:
                    # if it contains counts per strategy, ignore here
                    # fallback: take first numeric column
                    num_cols = C.select_dtypes(include=[np.number]).columns.tolist()
                    if not num_cols:
                        cnt = None
                    else:
                        cnt = C[num_cols[0]].to_numpy(dtype=float)
            else:
                cnt = np.asarray(C, dtype=float)

            if cnt is not None and len(cnt) == len(M_eval):
                # if M is SUM over folds and cnt is #test appearances, average safely
                cnt_safe = np.where(cnt > 0, cnt, np.nan)
                M_eval = M_eval.divide(cnt_safe, axis=0).fillna(0.0)

    else:
        raise ValueError("scheme must be WF or CPCV")

    # Metrics (Sharpe/DSR etc)
    metrics_df = compute_strategy_metrics(M_eval)

    # CSCV (optional)
    scheme_summary: Dict[str, object] = {"base": base, "scheme": scheme}
    if do_cscv and (M_eval.shape[1] >= 3) and (len(M_eval) >= 200):
        df_cscv = cscv_pipeline(M_eval, S=cscv_S, max_iter=cscv_max_iter)
        cscv_metrics = evaluate_cscv(df_cscv)

        # save CSCV raw
        out_cscv_path = os.path.join(analyzed_dir, f"{base}_{scheme}_cscv_results.xlsx")
        with pd.ExcelWriter(out_cscv_path) as writer:
            df_cscv.to_excel(writer, sheet_name=_safe_sheet_name("cscv"), index=False)
        scheme_summary.update(cscv_metrics)

        # attach CSCV summary columns to each strategy row for convenience
        metrics_df["PBO"] = cscv_metrics["PBO"]
        metrics_df["perf_degradation"] = cscv_metrics["perf_degradation"]
        metrics_df["prob_loss"] = cscv_metrics["prob_loss"]
        metrics_df["ks_stat"] = cscv_metrics["stochastic_dom"]["ks_stat"]
        metrics_df["ks_pval"] = cscv_metrics["stochastic_dom"]["pval"]
    else:
        # fill with NaNs if skipped
        metrics_df["PBO"] = np.nan
        metrics_df["perf_degradation"] = np.nan
        metrics_df["prob_loss"] = np.nan
        metrics_df["ks_stat"] = np.nan
        metrics_df["ks_pval"] = np.nan

    # save per-scheme metrics
    out_metrics_path = os.path.join(analyzed_dir, f"{base}_{scheme}_metrics.xlsx")
    with pd.ExcelWriter(out_metrics_path) as writer:
        metrics_df.round(6).to_excel(writer, sheet_name=_safe_sheet_name("metrics"), index=False)

    return metrics_df, scheme_summary


# =========================
# GLOBAL SUMMARIES
# =========================

def summarize_best_strategies(all_metrics: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    For each (pair, scheme) choose best strategy:
      - max DSR if available
      - else max Sharpe
    """
    rows = []
    for (pair, scheme), g in all_metrics.groupby(["pair", "scheme"], sort=False):
        g2 = g.copy()
        if g2["dsr"].notna().any():
            best = g2.loc[g2["dsr"].idxmax()]
        else:
            best = g2.loc[g2["sharpe"].idxmax()]
        rows.append({
            "pair": pair,
            "scheme": scheme,
            "best_strategy": best["strategy"],
            "dsr": best.get("dsr", np.nan),
            "sharpe": best["sharpe"],
            "sortino": best["sortino"],
            "n_trades": best["n_trades"],
            "PBO": best.get("PBO", np.nan),
            "perf_degradation": best.get("perf_degradation", np.nan),
            "prob_loss": best.get("prob_loss", np.nan),
        })
    summary = pd.DataFrame(rows)

    out_path = os.path.join(out_dir, "summary_best_strategies.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        summary.round(6).to_excel(writer, sheet_name=_safe_sheet_name("best"), index=False)
    return summary

def summarize_models_global(all_metrics: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Global summary per "model" family.
    Here we define model_family by removing q suffix:
      e.g. xgb_reg__q70 -> xgb_reg
    Customize to match your naming scheme.
    """
    def model_family(strategy: str) -> str:
        # remove __qNN patterns
        return re.sub(r"__q\d{2,3}$", "", str(strategy))

    df = all_metrics.copy()
    df["model"] = df["strategy"].map(model_family)

    g = df.groupby("model", sort=False)

    summary = pd.DataFrame({
        "n_strategies": g.size(),
        "avg_sharpe": g["sharpe"].mean(),
        "avg_dsr": g["dsr"].mean(),
        "frac_dsr_gt_0_5": g["dsr"].apply(lambda s: float((s > 0.5).mean())),
        "avg_PBO": g["PBO"].mean(),
        "avg_perf_degradation": g["perf_degradation"].mean(),
        "avg_prob_loss": g["prob_loss"].mean(),
        "frac_sharpe_gt_1": g["sharpe"].apply(lambda s: float((s > 1.0).mean())),
    }).reset_index()

    out_path = os.path.join(out_dir, "model_summary_global.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        summary.round(6).to_excel(writer, sheet_name=_safe_sheet_name("global"), index=False)
    return summary

def summarize_models_best_cases(all_metrics: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Best case per model (highest Sharpe).
    """
    def model_family(strategy: str) -> str:
        return re.sub(r"__q\d{2,3}$", "", str(strategy))

    df = all_metrics.copy()
    df["model"] = df["strategy"].map(model_family)
    df = df.sort_values("sharpe", ascending=False)

    best = df.groupby("model", sort=False).head(1).copy()
    cols = ["model", "pair", "scheme", "strategy", "sharpe", "dsr", "PBO", "perf_degradation", "prob_loss", "n_trades"]
    best = best[cols]

    out_path = os.path.join(out_dir, "model_best_cases.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        best.round(6).to_excel(writer, sheet_name=_safe_sheet_name("best_cases"), index=False)
    return best

def summarize_models_robustness(
    all_metrics: pd.DataFrame,
    out_dir: str,
    sharpe_thr: float = 1.0,
    dsr_thr: float = 0.5,
    pbo_thr: float = 0.1,
    perfdeg_thr: float = -0.3,
) -> pd.DataFrame:
    """
    Robustness per model:
      - fraction of scenarios passing thresholds
    """
    def model_family(strategy: str) -> str:
        return re.sub(r"__q\d{2,3}$", "", str(strategy))

    df = all_metrics.copy()
    df["model"] = df["strategy"].map(model_family)

    rows = []
    for model, g in df.groupby("model", sort=False):
        rows.append({
            "model": model,
            "n_scenarios": len(g),
            "frac_sharpe_gt_thr": float((g["sharpe"] > sharpe_thr).mean()),
            "frac_dsr_gt_thr": float((g["dsr"] > dsr_thr).mean()),
            "frac_pbo_lt_thr": float((g["PBO"] < pbo_thr).mean()),
            "frac_perfdeg_gt_thr": float((g["perf_degradation"] > perfdeg_thr).mean()),
        })

    rob = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "model_robustness.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        rob.round(6).to_excel(writer, sheet_name=_safe_sheet_name("robustness"), index=False)
    return rob


# =========================
# MAIN DRIVER
# =========================

def run_analysis(
    tested_dir: str,
    analyzed_dir: str,
    do_cscv: bool = True,
    cscv_S: int = 8,
    cscv_max_iter: Optional[int] = None,
) -> None:
    os.makedirs(analyzed_dir, exist_ok=True)

    backtests = discover_backtests(tested_dir)
    if not backtests:
        print(f"No backtests found in {tested_dir}")
        return

    all_rows = []

    for bt in backtests:
        pair = parse_pair_from_base(bt.base)

        # ---- WF ----
        if bt.wf_M is not None and bt.wf_split is not None:
            print(f"\n[{bt.base}] Analyzing WF...")
            wf_metrics, wf_summary = evaluate_one_scheme(
                base=bt.base,
                scheme="WF",
                M_path=bt.wf_M,
                split_path=bt.wf_split,
                counts_path=None,
                analyzed_dir=analyzed_dir,
                do_cscv=do_cscv,
                cscv_S=cscv_S,
                cscv_max_iter=cscv_max_iter,
            )
            wf_metrics["pair"] = pair
            wf_metrics["scheme"] = "WF"
            all_rows.append(wf_metrics)

        # ---- CPCV ----
        if bt.cpcv_M is not None and bt.cpcv_split is not None:
            print(f"\n[{bt.base}] Analyzing CPCV...")
            cpcv_metrics, cpcv_summary = evaluate_one_scheme(
                base=bt.base,
                scheme="CPCV",
                M_path=bt.cpcv_M,
                split_path=bt.cpcv_split,
                counts_path=bt.cpcv_counts,
                analyzed_dir=analyzed_dir,
                do_cscv=do_cscv,
                cscv_S=cscv_S,
                cscv_max_iter=cscv_max_iter,
            )
            cpcv_metrics["pair"] = pair
            cpcv_metrics["scheme"] = "CPCV"
            all_rows.append(cpcv_metrics)

    if not all_rows:
        print("Nothing analyzed (missing files).")
        return

    all_metrics = pd.concat(all_rows, axis=0, ignore_index=True)

    # save merged metrics for convenience
    merged_path = os.path.join(analyzed_dir, "all_metrics.parquet")
    all_metrics.to_parquet(merged_path, index=False)
    print(f"\nSaved merged metrics to {merged_path}")

    # Global summaries
    summarize_best_strategies(all_metrics, analyzed_dir)
    summarize_models_global(all_metrics, analyzed_dir)
    summarize_models_best_cases(all_metrics, analyzed_dir)
    summarize_models_robustness(all_metrics, analyzed_dir)

    print("\nDone.")


if __name__ == "__main__":
    # CHANGE THESE PATHS
    TESTED_DIR = r"C:\Users\enric\Desktop\tesi\data\dukascopy\tested"
    ANALYZED_DIR = r"C:\Users\enric\Desktop\tesi\data\dukascopy\analyzed"

    run_analysis(
        tested_dir=TESTED_DIR,
        analyzed_dir=ANALYZED_DIR,
        do_cscv=True,
        cscv_S=8,
        cscv_max_iter=None,
    )