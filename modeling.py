
"""
Created on Tue Nov 18 15:25:38 2025

@author: enric

Modeling with Walk-Forward CV and Combinatorial Purged CV (CPCV)

Input  : labeled directory with event-level datasets (per pair/frequency).
Output : tested directory with:
    - <basename>_WF_M.csv      : OOS returns matrix (events x models) from Walk-Forward CV
    - <basename>_WF_cv_results.csv: per-model performance metrics from Walk-Forward CV
    - <basename>_CPCV_M.csv    : OOS returns matrix (events x models) from CPCV
    - <basename>_CPCV_cv_results.csv: per-model performance metrics from CPCV

IMPORTANT:
- The strategy is signal-based, not candle-by-candle. Plug
  in specific transaction cost, spread, slippage, and risk-management logic
  into the placeholders provided.
"""
import os
import glob
import itertools
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime
from sympy import EulerGamma, E
from scipy.stats import norm
import math


# Column templates 
start_s = pd.Timestamp(datetime(2007, 1, 1), tz="UTC")
end_s   = pd.Timestamp(datetime(2025, 9, 29), tz="UTC")


# === COST CONFIGURATION =====================================================

# First scenario: no costs 
USE_COSTS = False            # True =  fee + slippage, False = 0 costs

# Fixed fee (round-trip) expressed in basis points on the notional
# Esempio: 0.25 bps ~ 0.000025 = 0.0025%
FEE_BPS = 0.25               

# Slippage 
SLIPPAGE_BPS_BASE = 0.10     # 0.10 bps ~ 0.00001

# Dynamic slippage
USE_DYNAMIC_SLIPPAGE = True  # if False use only SLIPPAGE_BPS_BASE

# intraday profile  (slippage in bps) for hour of the day 
SLIPPAGE_BPS_BY_HOUR = {
    # Asia / late US (più illiquido)
    0: 0.18, 1: 0.18, 2: 0.18, 3: 0.18, 4: 0.18, 5: 0.18,
    6: 0.15,
    # London (liquido)
    7: 0.10, 8: 0.08, 9: 0.06, 10: 0.06, 11: 0.06, 12: 0.06,
    # London–NY overlap (più liquido)
    13: 0.05, 14: 0.05, 15: 0.05, 16: 0.06,
    # NY pomeriggio
    17: 0.08, 18: 0.10, 19: 0.12, 20: 0.14,
    # Late US / pre-Asia
    21: 0.16, 22: 0.18, 23: 0.18,
}


# How many folds for Walk-Forward CV
N_WF_SPLITS = 5

# CPCV configuration
CPCV_S = 6                 # number of contiguous blocks S in CSCV/CPCV
CPCV_MAX_COMBINATIONS = 12  # or set an integer to limit combinations


def bps_to_return(bps: float) -> float:
    """
    Convert basis points to 'return' 
    1 bp = 0.0001 = 0.01%
    """
    return bps * 1e-4


def get_slippage_bps_for_hour(hour: int) -> float:
    """
    Return the slippage value in bps for a specific hour(0-23).
    If the hour is not in the map, use SLIPPAGE_BPS_BASE.
    """
    return SLIPPAGE_BPS_BY_HOUR.get(hour, SLIPPAGE_BPS_BASE)


def get_slippage_perc_for_timestamp(ts: pd.Timestamp) -> float:
    """
    Return the slippage in 'return terms' (es. 0.00001)
    for a timestamp. The slippage can be fixed or dynamic.
    """
    if not USE_DYNAMIC_SLIPPAGE:
        return bps_to_return(SLIPPAGE_BPS_BASE)

    h = ts.hour
    slippage_bps = get_slippage_bps_for_hour(h)
    return bps_to_return(slippage_bps)

def get_fee_perc() -> float:
    """
    Fixed fee round-trip in 'return terms'.
    """
    return bps_to_return(FEE_BPS)


# Cross-validation split generators
def generate_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (expanding window) splits using TimeSeriesSplit.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_splits : int
        Number of splits (folds).

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_idx, test_idx) for each fold.
    """
    dummy_X = np.zeros((n_samples, 1))  # only for indexing
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tscv.split(dummy_X):
        splits.append((train_idx, test_idx))
    return splits


def _contiguous_runs(sorted_idx: np.ndarray) -> List[Tuple[int, int]]:
    """Returns the list of (start, end) included for each contiguous run in sorted_idx."""
    if len(sorted_idx) == 0:
        return []
    runs = []
    s = sorted_idx[0]
    prev = s
    for x in sorted_idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((s, prev))
            s = x
            prev = x
    runs.append((s, prev))
    return runs


def generate_cpcv_splits(
    n_samples: int,
    S: int = 8,
    drop_remainder: bool = False,
    max_combinations: Optional[int] = None,
    purge: int = 0,                 # fallback
    embargo: float = 0.0,            # fallback 

    # --- time-aware ---
    t0: Optional[pd.Series] = None,         
    t_end: Optional[pd.Series] = None,      
    purge_td: Optional[pd.Timedelta] = None,
    embargo_td: Optional[pd.Timedelta] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    CPCV with time-aware option:

    - If t0 and t_end are available -> purging/embargo work on the intervals [t0, t_end]
    - otherwise -> use the fallback logic purge/embargo on the indexes
    """

    if S <= 1:
        raise ValueError("S must be >= 2 for CPCV.")

    base_size = n_samples // S
    remainder = n_samples % S

    blocks = []
    start = 0
    for i in range(S):
        extra = 1 if (i < remainder and not drop_remainder) else 0
        size = base_size + extra
        stop = start + size
        blocks.append(np.arange(start, stop))
        start = stop

    if np.concatenate(blocks).shape[0] != n_samples:
        raise RuntimeError("Block creation mismatch.")

    #List of combinations
    half = S // 2
    combos = list(itertools.combinations(range(S), half))

    use_time_aware = (t0 is not None) and (t_end is not None)

    #Normalize input time-aware
    if use_time_aware:
        t0 = pd.to_datetime(t0, utc=True)
        t_end = pd.to_datetime(t_end, utc=True)

        if purge_td is None:
            purge_td = pd.Timedelta(0)
        if embargo_td is None:
            embargo_td = pd.Timedelta(0)

    splits = []

    for k, combo in enumerate(combos):
        train_blocks = [blocks[i] for i in combo]
        test_blocks  = [blocks[i] for i in range(S) if i not in combo]

        train_idx = np.concatenate(train_blocks)
        test_idx  = np.concatenate(test_blocks)

        train_idx = np.unique(train_idx)
        test_idx  = np.unique(test_idx)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # ---------------- TIME-AWARE PURGING/EMBARGO ----------------
        if use_time_aware:
            # divide test_idx in contiguous runs -> each run is a “test block” in time
            te_sorted = np.sort(test_idx)
            runs = _contiguous_runs(te_sorted)

            # train mask
            keep = np.ones(len(train_idx), dtype=bool)

            tr_t0 = t0.iloc[train_idx].to_numpy()
            tr_te = t_end.iloc[train_idx].to_numpy()

            for (a, b) in runs:
                # test block interval 
                te_block_start = t0.iloc[a]
                te_block_end   = t_end.iloc[b]

                # purging: remove data in train blocks -> [start - purge_td, end + purge_td]
                start_p = te_block_start - purge_td
                end_p   = te_block_end + purge_td

                overlap = (tr_t0 <= end_p) & (tr_te >= start_p)
                keep &= ~overlap

                # embargo: remove train data that starts immediately after the test within embargo_td
                if embargo_td > pd.Timedelta(0):
                    emb_start = te_block_end
                    emb_end   = te_block_end + embargo_td
                    embargo_hit = (tr_t0 > emb_start) & (tr_t0 <= emb_end)
                    keep &= ~embargo_hit

            train_idx = train_idx[keep]

        # ---------------- FALLBACK ----------------
        else:
            print("Time-Aware CPCV is not currently used")
            # PURGE by samples around test blocks
            if purge > 0:
                for tb in test_blocks:
                    if len(tb) == 0:
                        continue
                    b_start = tb[0]
                    b_end   = tb[-1]
                    purge_lower = max(b_start - purge, 0)
                    purge_upper = min(b_end + purge, n_samples - 1)

                    mask = ~((train_idx >= purge_lower) & (train_idx <= purge_upper))
                    train_idx = train_idx[mask]

            # EMBARGO by fraction of dataset after max(test)
            if embargo > 0:
                embargo_size = int(n_samples * embargo)
                embargo_start = min(test_idx.max() + 1, n_samples - 1)
                embargo_end = min(embargo_start + embargo_size, n_samples - 1)
                embargo_range = np.arange(embargo_start, embargo_end + 1)
                train_idx = train_idx[~np.isin(train_idx, embargo_range)]

        train_idx = np.unique(train_idx)
        test_idx  = np.unique(test_idx)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append((train_idx, test_idx))

        if max_combinations is not None and len(splits) >= max_combinations:
            break

    return splits


# Strategy
def build_event_returns_classification(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    payoff_col: str,
    side_col: str,
    fee_perc: float = 0.0,
    spread_perc: float = 0.0,
    slippage_perc: float = 0.0,
    prob_threshold: float = 0.5,
    position_size: float = 1,
    time_col: str = "timestamp",
    use_dynamic_slippage: bool = True,
    gross_is_log:bool = True) -> np.ndarray:

    n = len(test_idx)
    out = np.zeros(n, dtype=float)

    if payoff_col not in df.columns or side_col not in df.columns:
        return out

    # --- prob must be 1D length n ---
    if y_proba is None:
        prob = (np.asarray(y_pred).reshape(-1) == 1).astype(float)
    else:
        yp = np.asarray(y_proba)
        # if predict_proba returns (n,2), take class-1 proba
        if yp.ndim == 2 and yp.shape[1] >= 2:
            prob = yp[:, 1]
        else:
            prob = yp.reshape(-1)

    prob = np.asarray(prob).reshape(-1)
    if prob.shape[0] != n:
        raise ValueError(f"Prob length mismatch: prob={prob.shape[0]} vs test={n}")

    payoff = df.loc[df.index[test_idx], payoff_col].to_numpy(dtype=float, copy=False).reshape(-1)
    side_vals = df.loc[df.index[test_idx], side_col].to_numpy(dtype=float, copy=False).reshape(-1)

    if payoff.shape[0] != n or side_vals.shape[0] != n:
        raise ValueError("payoff/side length mismatch in build_event_returns_classification")

    take = prob >= prob_threshold
    gross = side_vals * payoff
    if gross_is_log:
        gross = float(np.expm1(gross))  # exp(log) - 1
    else:
        gross = float(gross)
    gross = np.where(take, gross, 0.0)
    
    if not use_dynamic_slippage:
        total_cost = fee_perc + slippage_perc
        out = gross - np.where(take, total_cost, 0.0)
        return out.astype(float, copy=False)
    
    times = pd.to_datetime(df.loc[df.index[test_idx], time_col], utc=True)
    dyn_slip = np.array([get_slippage_perc_for_timestamp(ts) for ts in times], dtype=float)
    total_cost = fee_perc + dyn_slip
    out = (gross - np.where(take, total_cost, 0.0)) * float(position_size)
    return out.astype(float, copy=False)


def build_event_returns_regression(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    payoff_col: str,
    side_col: str,
    fee_perc: float = 0.0,
    spread_perc: float = 0.0,
    slippage_perc: float = 0.0,
    decision_threshold: float = 0.0,
    position_size: float = 1,
    time_col: str = "timestamp",
    use_dynamic_slippage: bool = True,
    gross_is_log:bool = True) -> np.ndarray:

    n = len(test_idx)
    out = np.zeros(n, dtype=float)

    if payoff_col not in df.columns or side_col not in df.columns:
        return out

    pred = np.asarray(y_pred).reshape(-1)
    if pred.shape[0] != n:
        raise ValueError(f"Pred length mismatch: pred={pred.shape[0]} vs test={n}")

    payoff = df.loc[df.index[test_idx], payoff_col].to_numpy(dtype=float, copy=False).reshape(-1)
    side_vals = df.loc[df.index[test_idx], side_col].to_numpy(dtype=float, copy=False).reshape(-1)

    if payoff.shape[0] != n or side_vals.shape[0] != n:
        raise ValueError("payoff/side length mismatch in build_event_returns_classification")


    take = pred > decision_threshold
    gross = side_vals * payoff
    if gross_is_log:
        gross = float(np.expm1(gross))  # exp(log) - 1
    else:
        gross = float(gross)
    gross = np.where(take, gross, 0.0)

    if not use_dynamic_slippage:
        total_cost = fee_perc + slippage_perc
        out = gross - np.where(take, total_cost, 0.0)
        return out.astype(float, copy=False)

    times = pd.to_datetime(df.loc[df.index[test_idx], time_col], utc=True)
    dyn_slip = np.array([get_slippage_perc_for_timestamp(ts) for ts in times], dtype=float)
    total_cost = fee_perc + dyn_slip
    out = (gross - np.where(take, total_cost, 0.0)) * float(position_size)
    return out.astype(float, copy=False)


# Model definitions
def get_classification_models() -> Dict[str, Pipeline]:
    """
    Define classification models.

    Returns
    -------
    Dict[str, Pipeline]
        Mapping from model name to sklearn Pipeline.
    """
    models: Dict[str, Pipeline] = {}

    # Logistic Regression (benchmark)
    models["logit_cls"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


    # XGBoost Classifier 
    models["xgb_cls"] = Pipeline([
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0,           
        random_state=42,
        )),
    ])
    return models


def get_regression_models() -> Dict[str, Pipeline]:
    """
    Define regression models (direct payoff forecasting).

    Returns
    -------
    Dict[str, Pipeline]
        Mapping from model name to sklearn Pipeline.
    """
    models: Dict[str, Pipeline] = {}
    
    # Linear Regression (benchmark)
    models["lin_reg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])

    
    # XGBoost Regressor 
    models["xgb_reg"] = Pipeline([
        ("reg", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            n_jobs=0,            
            random_state=42,
        )),
    ])
   
    return models


def features(
    df: pd.DataFrame,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pair_name: str | None = None,
    corr: str | None = None,
    basename: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return Pair dataframe with features to run ML models

    """

    # Ensure timestamp is parsed if present
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)


    df = df[(df[time_col] >= start_s) & (df[time_col] <= end_s)]
    
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df_time_indexed = df.set_index(time_col)
    
    if "ret_1" in df.columns:
        # raw 1-min return already in df["ret_1"]
        df["ret_1_abs"] = df["ret_1"].abs()
    
        # rolling volatility of 1-min returns (short horizon)
        df["ret_1_roll_std_20"] = df["ret_1"].rolling(window=20, min_periods=10).std()
    
    if "ret_5" in df.columns and "ret_1" in df.columns:
        # medium-horizon return already in df["ret_5"]
        # risk-adjusted momentum: 5-min return scaled by short-term vol of ret_1
        df["ret_5_mom_std_ratio"] = df["ret_5"] / (df["ret_1_roll_std_20"] + 1e-8)
    
    # ratio return/volatility (1-min)
    if "ret_1" in df.columns and "vol_60" in df.columns:
        df["ret_1_to_vol_60"] = df["ret_1"] / (df["vol_60"] + 1e-8)
    
    #Volatility
    if "vol_60" in df.columns:
        # original 60-bar volatility already in df["vol_60"]
        df["vol_60_lag1"] = df["vol_60"].shift(1)
        df["vol_60_roc"] = (
            df["vol_60"] / df["vol_60_lag1"] - 1.0
        ).replace([np.inf, -np.inf], np.nan)
    
        # long-term volatility regime (300 bars) built from ret_1
        if "ret_1" in df.columns:
            df["vol_300"] = df["ret_1"].rolling(window=300, min_periods=60).std()
            df["vol_60_to_vol_300"] = df["vol_60"] / (df["vol_300"] + 1e-8)
    
    #Microstructure features 
    required_ohlc = {"open_bid", "high_bid", "low_bid", "close_bid",
                     "open_ask", "high_ask", "low_ask", "close_ask"}
    
    if required_ohlc.issubset(df.columns):
        # spread at close (SP)
        df["SP"] = df["close_bid"] - df["close_ask"]
    
        # close-to-close bid and ask movements (CBP, CAP)
        df["CBP"] = df["close_bid"] - df["close_bid"].shift(1)
        df["CAP"] = df["close_ask"] - df["close_ask"].shift(1)
    
        # intrabar ranges (VBP, VAP)
        df["VBP"] = df["high_bid"] - df["low_bid"]
        df["VAP"] = df["high_ask"] - df["low_ask"]
    
        # optional: simple spread proxy on open
        df["spread_proxy"] = df["open_ask"] - df["open_bid"]
    

    #Side-score 
    if "side_score" in df.columns:
        df["side_score_roll_mean_30"] = (
            df["side_score"].rolling(window=30, min_periods=10).mean()
        )
        df["side_score_roll_std_30"] = (
            df["side_score"].rolling(window=30, min_periods=10).std()
        )
    
    # Candidate feature set 
    candidate_features = [
        # momentum / returns
        "ret_1",
        "ret_1_abs",
        "ret_5",
        "ret_5_mom_std_ratio",
        "ret_1_to_vol_60",
    
        # volatility level and regime
        "vol_60",
        "vol_60_lag1",
        "vol_60_roc",
        "vol_60_to_vol_300",
    
        # side-score level and stability
        "side_score",
        "side_score_roll_mean_30",
        "side_score_roll_std_30",
    
        # microstructure
        "SP",
        "VBP",
        "VAP",
    ]
    candidate_features = sorted(candidate_features)

    feature_cols = [c for c in candidate_features if c in df.columns]

    if len(feature_cols) == 0:
        # No features to correlate
        corr_matrix = pd.DataFrame()
        return df, df_time_indexed, corr_matrix

    # Drop rows with NaN in these features before computing correlation
    df_feat = df[feature_cols].dropna(how="any")

    if df_feat.empty:
        corr_matrix = pd.DataFrame()
        return df, df_time_indexed, corr_matrix

    corr_matrix = df_feat.corr()

    #Optional: save per-pair correlation heatmap (and CSV)
    if pair_name is not None and corr is not None:
        os.makedirs(corr, exist_ok=True)

        # Save correlation matrix as CSV
        corr_csv_path = os.path.join(corr, f"{basename}_corr_features.csv")
        corr_matrix.to_csv(corr_csv_path, index=True)

        # Plot and save heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            cbar=True,
        )
        plt.title(f"Feature correlation heatmap - {basename}")
        plt.tight_layout()
        heatmap_path = os.path.join(corr, f"{basename}_corr_features.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

    return df, df_time_indexed, corr_matrix

def mean_corr_matrix(corr_matrices,
                     corr: str):
    """
    Return Average features correlation heatmap between different pairs
    """
    # Align matrices
    common_features = None
    for cm in corr_matrices.values():
        if common_features is None:
            common_features = set(cm.columns)
        else:
            common_features &= set(cm.columns)
    
    common_features = sorted(list(common_features))
    
    if not common_features:
        raise RuntimeError("No common features across pairs for correlation averaging.")
    
    # Stack and mean
    stack = []
    for pair, cm in corr_matrices.items():
        cm_sub = cm.loc[common_features, common_features]
        stack.append(cm_sub.values)
    
    # stack: shape (n_pairs, n_features, n_features)
    stack_arr = np.stack(stack, axis=0)
    mean_corr = stack_arr.mean(axis=0)
    
    mean_corr_df = pd.DataFrame(mean_corr, index=common_features, columns=common_features)
    
    # Average heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mean_corr_df,
        annot=False,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar=True,
    )
    plt.title("Average feature correlation across pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(corr, "mean_corr_features.png"), dpi=150)
    plt.close()
    
    mean_corr_df.to_csv(os.path.join(corr, "mean_corr_features.csv"), index=True)
    

def diagnose_backtest_pipeline(
    df_h: pd.DataFrame,
    splits,
    payoff_col: str,
    side_col: str,
    time_col: str,
    t_end_col: str,
    take_mask: pd.Series,
    name: str,
):
    print("\n" + "=" * 80)
    print(f"BACKTEST DIAGNOSTICS — {name}")
    print("=" * 80)

    # Payoff distribution
    p = df_h[payoff_col]
    print("\n[1] PAYOFF DISTRIBUTION")
    print(p.describe())
    print("P(payoff > 0):", float(np.mean(p > 0)))

    # 2) Long / Short balance
    print("\n[2] SIDE DISTRIBUTION (ALL EVENTS)")
    print(df_h[side_col].value_counts(normalize=True))

    print("\n[3] SIDE DISTRIBUTION (TAKEN TRADES)")
    taken_sides = df_h.loc[take_mask, side_col]
    print(taken_sides.value_counts(normalize=True))

    # 3) Trade-taking bias
    print("\n[4] TRADE-TAKING RATE BY SIDE")
    for s in [-1, 1]:
        m = (df_h[side_col] == s)
        rate = float(take_mask[m].mean())
        print(f"side={s:+d}: take_rate={rate:.3f}")

    # 4) Temporal/interval overlap check
    print("\n[5] TEMPORAL SPLIT CHECK (INDEX + INTERVAL)")
    t0 = pd.to_datetime(df_h[time_col], utc=True)
    te = pd.to_datetime(df_h[t_end_col], utc=True)

    def _runs_from_idx(idx: np.ndarray):
        idx = np.sort(np.unique(idx))
        if len(idx) == 0:
            return []
        runs = []
        s = idx[0]
        prev = s
        for x in idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((s, prev))
                s, prev = x, x
        runs.append((s, prev))
        return runs

    for i, (tr, te_idx) in enumerate(splits[:3], 1):
        tr = np.unique(tr)
        te_idx = np.unique(te_idx)

        # sanity: train/test index overlap
        set_overlap = len(np.intersect1d(tr, te_idx)) > 0

        # “range” (solo informativo)
        tr_start = t0.iloc[tr].min()
        tr_end_max = te.iloc[tr].max()
        te_start = t0.iloc[te_idx].min()
        te_end_max = te.iloc[te_idx].max()

        # overlap vero: train intervals vs test blocks (segmenti contigui in indice/tempo)
        runs = _runs_from_idx(te_idx)

        tr_t0 = t0.iloc[tr].to_numpy()
        tr_te = te.iloc[tr].to_numpy()

        any_interval_overlap = False
        overlap_count = 0

        for (a, b) in runs:
            # test block interval
            block_start = t0.iloc[a]
            block_end   = te.iloc[b]

            # train event overlaps this block if t0<=block_end and t_end>=block_start
            ov = (tr_t0 <= block_end) & (tr_te >= block_start)
            c = int(ov.sum())
            if c > 0:
                any_interval_overlap = True
                overlap_count += c

        print(
            f"Fold {i}: "
            f"train=[{tr_start}, {tr_end_max}] | "
            f"test=[{te_start}, {te_end_max}] | "
            f"IDX_OVERLAP={set_overlap} | "
            f"INTERVAL_OVERLAP={any_interval_overlap} "
            f"(train_events_overlapping_test_blocks={overlap_count})"
        )



@dataclass
class CPCVParams:
    S: int = CPCV_S
    drop_remainder: bool = False
    max_combinations: Optional[int] = CPCV_MAX_COMBINATIONS
    purge: int = 0
    embargo: float = 0.0

def _compute_fwd_log_ret(
    df_events: pd.DataFrame,
    df_time_indexed: pd.DataFrame,
    time_col: str,
    t_end_col: str,
    side_col: str,
    out_col: str = "fwd_log_ret",
) -> pd.Series:
    """
    Compute directional forward log-return for each event:
      - Long:  entry=open_ask(t0)  exit=close_bid(t_end)
      - Short: entry=open_bid(t0)  exit=close_ask(t_end)
    Return: side * (log(exit) - log(entry)), aligned to df_events rows.
    """

    req_evt = {side_col, t_end_col, "open_bid", "open_ask"}
    req_mkt = {"close_bid", "close_ask"}
    missing_evt = req_evt - set(df_events.columns)
    missing_mkt = req_mkt - set(df_time_indexed.columns)
    if missing_evt:
        raise ValueError(f"Missing event columns for payoff: {missing_evt}")
    if missing_mkt:
        raise ValueError(f"Missing market columns in df_time_indexed: {missing_mkt}")

    # Ensure datetime (tz-aware UTC strongly recommended)
    df_events[time_col] = pd.to_datetime(df_events[time_col], utc=True)
    df_events[t_end_col] = pd.to_datetime(df_events[t_end_col], utc=True)

    side = df_events[side_col].to_numpy(dtype=float)
    t_end = df_events[t_end_col]

    # entry at t0
    open_bid = df_events["open_bid"].to_numpy(dtype=float)
    open_ask = df_events["open_ask"].to_numpy(dtype=float)
    entry = np.where(side > 0, open_ask, open_bid)

    # exit at t_end (time-based reindex)
    exit_close_bid = df_time_indexed["close_bid"].reindex(t_end).to_numpy(dtype=float)
    exit_close_ask = df_time_indexed["close_ask"].reindex(t_end).to_numpy(dtype=float)
    exitp = np.where(side > 0, exit_close_bid, exit_close_ask)

    # validity
    ok = (
        np.isfinite(side) & (side != 0)
        & np.isfinite(entry) & (entry > 0)
        & np.isfinite(exitp) & (exitp > 0)
    )

    out = np.full(len(df_events), np.nan, dtype=float)
    out[ok] = side[ok] * (np.log(exitp[ok]) - np.log(entry[ok]))

    return pd.Series(out, index=df_events.index, name=out_col)


def run_models(
    df: pd.DataFrame,
    df_time_indexed: pd.DataFrame,
    tested: str,
    basename: str,
    time_col: str,
    cls_models: Dict[str, Pipeline],
    reg_models: Dict[str, Pipeline],
    use_walkforward: bool,
    use_cpcv: bool,
    cpcv_params: CPCVParams,
    fee_perc: float,
    spread_perc: float,
    slippage_perc: float,
    use_dynamic_slippage: bool,
    diag: bool = False,
    diag_model: str = "logit_cls",
):
    """
    Requires:
      - time_col (t0)
      - t_end
      - meta_y
      - side
      - open_bid/open_ask on event rows
    Uses df_time_indexed for close_bid/close_ask reindexed at t_end.
    """

    basename = Path(basename).stem.replace("_LABELED", "")

    meta_col   = "meta_y"
    t_end_col  = "t_end"
    side_col   = "side"
    weight_col = "weight"
    payoff_col = "fwd_log_ret"   # computed here if missing

    required = [time_col, meta_col, t_end_col, side_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{basename}: missing required columns {missing}")

    # ensure time types
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df[t_end_col] = pd.to_datetime(df[t_end_col], utc=True)

    # Compute payoff if not present
    if payoff_col not in df.columns:
        df[payoff_col] = _compute_fwd_log_ret(
            df_events=df,
            df_time_indexed=df_time_indexed,
            time_col=time_col,
            t_end_col=t_end_col,
            side_col=side_col,
            out_col=payoff_col,
        )

    # Filter valid events
    mask_evt = (
        df[meta_col].notna()
        & df[payoff_col].notna()
        & df[side_col].notna()
        & (df[side_col] != 0)
    )
    df_h = df.loc[mask_evt].copy().reset_index(drop=True)

    if len(df_h) < 200:
        print(f"{basename}: too few valid events ({len(df_h)}), skipping.")
        return None

    # Build X, y (avoid leakage)
    leak_cols = [meta_col, payoff_col, t_end_col, time_col]
    X = df_h.drop(columns=[c for c in leak_cols if c in df_h.columns], errors="ignore")

    y_cls = df_h[meta_col].astype(int)
    y_reg = df_h[payoff_col].astype(float)

    if weight_col in df_h.columns:
        sample_weight = df_h[weight_col].to_numpy(dtype=float)
    else:
        sample_weight = np.ones(len(df_h), dtype=float)

    # Drop rows with NaN in X
    mask_no_nan = ~X.isna().any(axis=1)
    X = X.loc[mask_no_nan].reset_index(drop=True)
    # --- drop any datetime-like columns from X (leakage + sklearn compatibility)
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if len(dt_cols) > 0:
        # print for debug once
        print(f"[{basename}] Dropping datetime columns from X: {dt_cols}")
        X = X.drop(columns=dt_cols)
    # if any object columns remain, try coercing to numeric or drop
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if len(obj_cols) > 0:
        # attempt numeric coercion; drop if still non-numeric
        for c in obj_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        # after coercion, drop columns that are entirely NaN (meaning non-numeric)
        all_nan_obj = [c for c in obj_cols if X[c].isna().all()]
        if all_nan_obj:
            print(f"[{basename}] Dropping non-numeric object columns from X: {all_nan_obj}")
            X = X.drop(columns=all_nan_obj)
    # drop obvious TBM / leakage patterns if present
    pattern_drop = [c for c in X.columns if c.startswith("t1_") or c.startswith("t_end") or c.startswith("label") or c.startswith("meta_")]
    if pattern_drop:
        print(f"[{basename}] Dropping leakage-pattern columns from X: {pattern_drop[:20]}{'...' if len(pattern_drop)>20 else ''}")
        X = X.drop(columns=pattern_drop)

    print(f"[{basename}] X dtypes summary:")
    print(X.dtypes.value_counts())
    
    y_cls = y_cls.loc[mask_no_nan].reset_index(drop=True)
    y_reg = y_reg.loc[mask_no_nan].reset_index(drop=True)
    df_h = df_h.loc[mask_no_nan].reset_index(drop=True)
    sample_weight = sample_weight[mask_no_nan.values]

    n_samples = len(X)
    print(f"[{basename}] Using {n_samples} events after NaN filter.")

    if n_samples < 200:
        print(f"{basename}: too few rows after NaN filtering.")
        return None

    # Splits
    splits_wf, splits_cpcv = [], []

    if use_walkforward:
        splits_wf = generate_walk_forward_splits(n_samples=n_samples)

    if use_cpcv:
        splits_cpcv = generate_cpcv_splits(
            n_samples=n_samples,
            S=cpcv_params.S,
            drop_remainder=cpcv_params.drop_remainder,
            max_combinations=cpcv_params.max_combinations,
    
            # time-aware
            t0=df_h[time_col],
            t_end=df_h[t_end_col],

            purge_td=pd.Timedelta(minutes=5),     
            embargo_td=pd.Timedelta(minutes=60),  
        )


    # Run CV
    def _run_scheme(scheme_name: str, splits):

        returns_store = {m: np.zeros(n_samples) for m in list(cls_models) + list(reg_models)}
        taken_store = {m: np.zeros(n_samples, dtype=bool) for m in cls_models}
        rows = []

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr_cls, y_te_cls = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
            sw_tr = sample_weight[train_idx]

            # --- CLS ---
            for name, pipe in cls_models.items():
                pipe.fit(X_tr, y_tr_cls, clf__sample_weight=sw_tr)

                y_pred = pipe.predict(X_te)
                try:
                    y_proba = pipe.predict_proba(X_te)[:, 1]
                except Exception:
                    y_proba = None

                fold_ret = build_event_returns_classification(
                    df=df_h,
                    test_idx=test_idx,
                    y_true=y_te_cls.values,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    payoff_col=payoff_col,
                    side_col=side_col,
                    fee_perc=fee_perc,
                    spread_perc=spread_perc,
                    slippage_perc=slippage_perc,
                    prob_threshold=0.5,
                    time_col=time_col,
                    use_dynamic_slippage=use_dynamic_slippage,
                )
                fold_ret = np.asarray(fold_ret).reshape(-1)
                if fold_ret.shape[0] != len(test_idx):
                    raise ValueError(f"{scheme_name} {name}: fold_ret shape {fold_ret.shape} != len(test_idx) {len(test_idx)}")
                returns_store[name][test_idx] = fold_ret

                taken_store[name][test_idx] = (fold_ret != 0)

                try:
                    auc = roc_auc_score(y_te_cls, y_proba) if y_proba is not None else np.nan
                except ValueError:
                    auc = np.nan

                rows.append({
                    "basename": basename,
                    "cv": scheme_name,
                    "model": name,
                    "task": "cls",
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "auc": auc,
                })

            # --- REG ---
            y_tr_reg = y_reg.iloc[train_idx].to_numpy()
            y_te_reg = y_reg.iloc[test_idx].to_numpy()

            for name, pipe in reg_models.items():
                pipe.fit(X_tr, y_tr_reg, reg__sample_weight=sw_tr)
                y_hat = pipe.predict(X_te)

                fold_ret = build_event_returns_regression(
                    df=df_h,
                    test_idx=test_idx,
                    y_true=y_te_reg,
                    y_pred=y_hat,
                    payoff_col=payoff_col,
                    side_col=side_col,
                    fee_perc=fee_perc,
                    spread_perc=spread_perc,
                    slippage_perc=slippage_perc,
                    decision_threshold=0.2,
                    time_col=time_col,
                    use_dynamic_slippage=use_dynamic_slippage,
                )
                fold_ret = np.asarray(fold_ret).reshape(-1)
                if fold_ret.shape[0] != len(test_idx):
                    raise ValueError(f"{scheme_name} {name}: fold_ret shape {fold_ret.shape} != len(test_idx) {len(test_idx)}")

                returns_store[name][test_idx] = fold_ret

        M = pd.DataFrame(returns_store, index=df_h[time_col])
        results = pd.DataFrame(rows)
        return M, results, taken_store, splits

    M_wf = pd.DataFrame()
    results_wf = pd.DataFrame()
    M_cpcv = pd.DataFrame()
    results_cpcv = pd.DataFrame()
    
    if use_walkforward:
        M_wf, results_wf, taken_wf, splits_wf = _run_scheme("WF", splits_wf)
        if diag:
            take_mask = taken_wf.get(diag_model)
            if take_mask is not None:
                diagnose_backtest_pipeline(
                    df_h=df_h,
                    splits=splits_wf,
                    payoff_col=payoff_col,
                    side_col=side_col,
                    time_col=time_col,
                    t_end_col="t1_vert",  # o quello che usi come fine evento
                    take_mask=pd.Series(take_mask, index=df_h.index),
                    name=f"{basename} | WF | {diag_model}",
                )

    if use_cpcv:
        M_cpcv, results_cpcv, taken_cpcv, splits_cpcv = _run_scheme("CPCV", splits_cpcv)
        if diag:
            take_mask = taken_cpcv.get(diag_model)
            if take_mask is not None:
                diagnose_backtest_pipeline(
                    df_h=df_h,
                    splits=splits_cpcv,
                    payoff_col=payoff_col,
                    side_col=side_col,
                    time_col=time_col,
                    t_end_col="t1_vert",
                    take_mask=pd.Series(take_mask, index=df_h.index),
                    name=f"{basename} | CPCV | {diag_model}",
                )

    return M_wf, results_wf, M_cpcv, results_cpcv

 

       
#Average Max Sharpe Ratio
def run_batch(sims, tried_strats, freq, T, mu_true, sigma, plot_fig = False):
    rng = np.random.default_rng()
    rets = rng.normal(loc=mu_true, scale = sigma, size=(sims, tried_strats, T))
    means = rets.mean(axis = 2)
    stds = rets.std(axis = 2, ddof = 1)
    stds = np.where(stds == 0, np.nan, stds)
    srs = means / stds * math.sqrt(freq)
    srs = np.nan_to_num(srs, nan = 0.0)
    
    if plot_fig:
        # Plot returns
        plt.plot(rets[0].T)
        plt.title(f"Returns of {tried_strats} strategies (Simulation 0)")
        plt.xlabel("Time steps (e.g. days)")
        plt.ylabel("Cumulative return")
        plt.grid(True, alpha = 0.3)
        plt.show()
        
        #Plot cum returns
        sim0 = rets[0]
        cum_rets = (1 + sim0).cumprod(axis=1)-1
        plt.figure(figsize=(8,5))
        plt.plot(cum_rets.T)
        plt.title(f"Cumulative returns of {tried_strats} strategies (Simulation 0)")
        plt.xlabel("Time steps (e.g. days)")
        plt.ylabel("Cumulative return")
        plt.grid(True, alpha = 0.3)
        plt.show()

        #max sharpe plot
        best_idx = np.argmax(srs[0])
        plt.figure(figsize=(8,5))
        plt.plot(cum_rets.T, color = "gray", alpha = 0.4)

        #paint the best strategy in red
        plt.plot(cum_rets[best_idx], color = "red", linewidth = 2.5, label=f"Best SR ({srs[0][best_idx]:.2f})") 
        plt.title(f"Cumulative return of {tried_strats} strategies (Simulation 0)\n"
                  f"Best strategy: #{best_idx} (SR={srs[0][best_idx]:.2f})")
        plt.xlabel("Time steps (e.g. days)")
        plt.ylabel("Cumulative return")
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.show()
        
    return srs.max(axis = 1)

def generate_max_sr_plot(sims, tried_strats, freq, T, mu_true, sigma, years):
    # (B) Sweep over different numbers of tried strategies and store average max Sharpe
    # Your requested list (includes 1..10, then specific larger values; duplicate 1000 kept on purpose)
    sweep_strats = list(range(1, 11)) + [50, 100, 150, 200, 300, 400]
    avg_max_by_n = []

    for n in sweep_strats:
        # Turn off internal plotting for speed
        max_sr = run_batch(sims, n, freq, T, mu_true, sigma)
        avg_max = float(max_sr.mean())
        avg_max_by_n.append(avg_max)
        print(f"n_strats={n:4d} -> E[max SR] = {avg_max:.4f}")

    # === theoretical curve ===
    gamma = float(EulerGamma)  # Euler–Mascheroni constant
    e = float(E)
    n_theory = np.arange(1, 1001)
    y_theory = (1 - gamma) * norm.ppf(1 - 1 / n_theory) + gamma * norm.ppf(1 - 1 / (n_theory * e))

    # === upper bound curve ===
    upper_bound = np.sqrt(2 * np.log(n_theory))

    # === combined plot ===
    plt.figure(figsize=(9, 5))

    # Simulated data (discrete points)
    plt.plot(sweep_strats, avg_max_by_n, 'o-', label="Simulated E[max Sharpe]", color='C0')

    # Theoretical curve (smooth line)
    plt.plot(n_theory, y_theory, '--',
             label=r"Theory: $(1-\gamma)Z^{-1}(1-\frac{1}{n}) + \gamma Z^{-1}(1-\frac{1}{ne})$",
             color='C1')

    # Upper bound curve
    plt.plot(n_theory, upper_bound, '-.',
             label=r"Upper Bound: $\sqrt{2\ln n}$", color='C2')

    # Aesthetics
    plt.title(f"Empirical vs Theoretical E[max Sharpe]\n({sims} experiments, {years} years, {freq}/yr, true SR=0)")
    plt.xlabel("Number of strategies tried (n)")
    plt.ylabel("E[max Sharpe]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()










    

def calculate_minimum_backtest_length(tried_strats, expected_max_sr):
    gamma = float(EulerGamma)
    e = float(E)
    minbtl = (((1 - gamma) * norm.ppf(1 - 1 / tried_strats)
               + gamma * norm.ppf(1 - 1 / (tried_strats * e)))
              / expected_max_sr) ** 2
    return minbtl


def generate_minbtl_plot(expected_max_sr):
    gamma = float(EulerGamma)
    e = float(E)
    tried_strats = np.arange(1, 1001)
    minbtl = (((1 - gamma) * norm.ppf(1 - 1 / tried_strats)
               + gamma * norm.ppf(1 - 1 / (tried_strats * e)))
              / expected_max_sr) ** 2

    upperbound = 2 * np.log(tried_strats) / (expected_max_sr ** 2)
    
    #Theoretical curve (smooth line)
    plt.figure(figsize=(9,5))
    plt.plot(tried_strats, minbtl, "-", label = r"MinBTL", color = "C1")
    plt.plot(tried_strats, upperbound, "-", label = r"Upper Bound: $\frac{2ln(n)}{E(max_n)^2}$", color = "C3")
    
    #Aesthetics
    plt.title(f"MinBTL per Expected Maximum SR = {expected_max_sr}")
    plt.xlabel("Number of strategies tried {n}")
    plt.ylabel("Minimum years of test")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()






# Main loop 
def main():
    """
    Steps:
    1. List all LABELED files.
    2. For each file:
       a. Load the dataset.
       b. For each horizon H in HORIZONS:
          - Run Walk-Forward CV (WF) models.
          - Run CPCV models.
    3. Save all matrices and results into tested directory.
    """
    
    BASE_DIR = Path("C:/Users/enric/Desktop/tesi/gaps.py").resolve().parent
    DATA_DIR = BASE_DIR / "data"
    
    #This dictionary contains input and output folders, 
    #time is used to recognise the time column while suffix is the suffix used to recognise the datasets.
    
    folders = [
        {
            "labeled": DATA_DIR / "dukascopy" / "labeled",
            "tested": DATA_DIR / "dukascopy" / "tested",
            "corr": DATA_DIR / "dukascopy" / "corr",
        },
        
        {
            "labeled": DATA_DIR / "ibkr" / "labeled",
            "tested": DATA_DIR / "ibkr" / "tested",
            "corr": DATA_DIR / "ibkr" / "corr",
        }
    ]
    
    
    params = CPCVParams(S=CPCV_S,
              drop_remainder=False,
               max_combinations=CPCV_MAX_COMBINATIONS,
               purge= 0,
               embargo= 0)
    
    #We want to compute the max average sharpe ratio obtained by a random strategy 
    #We must consider the number of different strategies tested, the frequency and the number of years
    #Parameters
    n_events = 1700 #Just for reference, WF with 5 split and CPCV with 6 splits 
    #generate splits with approximately 1700 events(3 years of data) 
    years = 3
    sims = 10000
    mu_true = 0.0
    sigma = 0.01
    freq = n_events/years
    T = int(freq*years)
    
    
    cv = 2 #Number of Cross validation models used: CPCV and WF
    ml_models = 2 #Number of ML models for each CV model, Linear/Logistic reg and XGBoost
    #strat = 7 #In previous analysis we used 7 different vertical barriers for tbm
    tried_strats =  cv * ml_models   if [ cv, ml_models] !=0 else 0
    max_sharpe = run_batch(sims, tried_strats, freq, T, mu_true , sigma, plot_fig=False)
    
    #Average max sharpe ratio
    if sims == 1:
        print(f"Max Sharpe of {tried_strats} strategies: {round(max_sharpe.mean(),4)}")
    else:
        print(f"Average Max Sharpe in {sims} simulations for {tried_strats} strategies: {round(max_sharpe.mean(),4)}")
        #plt.plot(max_sharpe)
        #plt.title(f"Max Sharpe - {sims} simulations")
    generate_maxsr_plot_variable = False
    if generate_maxsr_plot_variable:
        generate_max_sr_plot(sims, tried_strats, freq, T, mu_true, sigma, years)
    
    
    #Parameters for MinBTL
    expected_max_sr = round(max_sharpe.mean(),4)
    generate_minbtl_plot_variable = False
    if generate_minbtl_plot_variable:
        generate_minbtl_plot(expected_max_sr)
        
    minbtl = calculate_minimum_backtest_length(tried_strats, expected_max_sr)
    print(f"MinBTL for {tried_strats} tried strategies with expected maximum SR of {expected_max_sr}: {round(minbtl,4)} years")


    """
    for folder in folders:
        
        labeled = folder["labeled"]
        tested = folder["tested"]
        corr = folder["corr"]
        os.makedirs(tested, exist_ok=True)
        
        labeled_files = glob.glob(os.path.join(labeled, "*.parquet"))
        if not labeled_files:
            print(f"No parquet files found in {labeled}.")
            return
    
        print(f"Found {len(labeled_files)} LABELED files.")
        for path in labeled_files:
            basename = os.path.splitext(os.path.basename(path))[0]
            print(f"\nProcessing file: {basename}")
    
            df = pd.read_parquet(path)
            
            time_col = "timestamp"
    
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], utc=True)
            
            # Normalize t1 columns: tutte in UTC
            t1_cols = [c for c in df.columns if c.startswith("t1_")]
            
            for col in t1_cols:
                df[col] = pd.to_datetime(df[col], utc=True)
    
            df, df_time_indexed, corr_matrix = features(
            df=df,
            corr = corr,
            time_col="timestamp",
            start=start_s,
            end=end_s,
            pair_name=path,
            basename=basename
        )   
            

            
            
            
            corr_matrices = {}

            if not corr_matrix.empty:
                corr_matrices[path] = corr_matrix
                mean_corr_matrix(corr_matrices,
                                 corr = corr)
            
            #print(df["timestamp"].dtype)
            #print(df["t1_H1"].dtype)   # o un altro orizzonte sicuramente presente
            #print(df_time_indexed.index.dtype)
            print(df)
                 # --- costi da usare in questo run ---
            if USE_COSTS:
                fee_perc = get_fee_perc()
                # se vuoi comunque un minimo slippage statico oltre al dinamico
                slippage_perc = bps_to_return(SLIPPAGE_BPS_BASE)
            else:
                fee_perc = 0.0
                slippage_perc = 0.0
         
            spread_perc = 0.0  # lo spread è già nel payoff via bid/ask
    
    
            # Walk-Forward CV CV and CPCV
            print("Running Walk-Forward CV and CPCV...")
            M_wf, results_wf, M_cpcv, results_cpcv = run_models(
                df=df,
                df_time_indexed=df_time_indexed,
                tested=tested,
                basename=basename,
                time_col="timestamp",
                cls_models=get_classification_models(),
                reg_models=get_regression_models(),
                use_walkforward=True,
                use_cpcv=True,
                cpcv_params=params,
                fee_perc=fee_perc,
                spread_perc=spread_perc,
                slippage_perc=slippage_perc,
                diag=True,
                diag_model="logit_cls",
                use_dynamic_slippage=USE_COSTS and USE_DYNAMIC_SLIPPAGE,
            )
"""
if __name__ == "__main__":
    main()

'''

    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(20)
    



    purge_summary = pd.concat(summary_list, ignore_index=True)
    pivot = purge_summary.pivot(index="pair", columns="horizon", values="p99_minutes")
    print(pivot)

'''


'''        

'''   

    
