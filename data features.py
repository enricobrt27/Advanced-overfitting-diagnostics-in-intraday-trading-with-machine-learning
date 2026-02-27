import os, glob, re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
from data_cleaning import drop_duplicate_timestamps, detect_missing_periods, detect_invalid_blocks, ensure_datetime_utc
from sklearn.metrics import roc_auc_score, log_loss

TZ = "UTC"                           #assume/convert to UTC
EXPECTED_FREQ = pd.Timedelta(minutes=1) 
start_s = pd.Timestamp(datetime(2007, 1, 1), tz="UTC")
end_s   = pd.Timestamp(datetime(2025, 9, 30), tz="UTC")

# Gap thresholds 
MIN_SHORT_GAP = pd.Timedelta(minutes=1)
MAX_SHORT_GAP = pd.Timedelta(days=2)
MIN_LONG_GAP  = pd.Timedelta(days=2)        # to ignore weekends
MAX_LONG_GAP  = pd.Timedelta(days=10)

# Columns expected 
OHLC_COLS = ["open_ask", "high_ask", "low_ask", "close_ask","open_bid", "high_bid", "low_bid", "close_bid"]


def bid_ask_checks(df: pd.DataFrame,
                   bid_prefix="bid",
                   ask_prefix="ask",
                   ) -> pd.DataFrame:
    
    #Drop rows where ask < bid for each OHLC and non negative spread.
    #Input: open_bid, high_bid, low_bid, close_bid e correspondent _ask columns.
    
    df = df.copy()
    cond = pd.Series(True, index=df.index)
    for fld in ["open","high","low","close"]:
        b = f"{fld}_{bid_prefix}"
        a = f"{fld}_{ask_prefix}"
        if b in df.columns and a in df.columns:
            eps = 1e-10
            cond &= (df[a] + eps) >= df[b]
    removed = (~cond).sum()
    if removed:
        print(f"Removed {removed} rows where ask < bid (any OHLC).")
    df = df.loc[cond].reset_index(drop=True)
    
    if "close_bid" in df.columns and "close_ask" in df.columns:
        spread_ok = (df["close_ask"] - df["close_bid"]) > 0
        rem2 = (~spread_ok).sum()
        if rem2:
            print(f"Removed {rem2} rows with non-positive close spread.")
        df = df.loc[spread_ok].reset_index(drop=True)
    return df


def parse_pair_side(fname: str) -> Tuple[str, str]:
    """
    Extract (PAIR, SIDE) from names like:
    EUR_USD_BID_2007-01-01_2025-09-30_CLEAN.parquet  -> ("EUR_USD","BID")
    """
    
    m = re.search(r"([A-Z]{3}_[A-Z]{3})_(BID|ASK)", fname.upper())
    if not m:
        raise ValueError(f"Filename not recognized: {fname}")
    return m.group(1), m.group(2)


def join_bid_ask_file(bid_path: str, ask_path: str,
                      time_col="timestamp") -> pd.DataFrame:
    """
    Read two files CLEAN (bid/ask), align by timestamp (inner) and rename columns.
    """
    
    df_bid = pd.read_parquet(bid_path)
    df_ask = pd.read_parquet(ask_path)
    df_bid = ensure_datetime_utc(df_bid, time_col)
    df_ask = ensure_datetime_utc(df_ask, time_col)

    # Rename adding suffixes _bid/_ask
    rename_ohlc = {"open":"open", "high":"high", "low":"low", "close":"close"}
    df_bid = df_bid.rename(columns={c: f"{c}_bid" for c in rename_ohlc if c in df_bid.columns})
    df_ask = df_ask.rename(columns={c: f"{c}_ask" for c in rename_ohlc if c in df_ask.columns})

    keep_cols_bid = [time_col] + [c for c in df_bid.columns if c.endswith("_bid")]
    keep_cols_ask = [time_col] + [c for c in df_ask.columns if c.endswith("_ask")]
    df_bid = df_bid[keep_cols_bid]
    df_ask = df_ask[keep_cols_ask]

    df = df_bid.merge(df_ask, on=time_col, how="inner").sort_values(time_col).reset_index(drop=True)
    df = bid_ask_checks(df, bid_prefix="bid", ask_prefix="ask")
    return df

def join_folder(cleaned: str, joined: str, time_col: str):
    os.makedirs(joined, exist_ok=True)
    files = glob.glob(os.path.join(cleaned, "*_CLEAN.parquet"))
    if not files:
        print(f"No CLEAN files found in {cleaned}")
        return

    # Pair map-> {BID: path, ASK: path}
    index = {}
    for f in files:
        base = os.path.basename(f)
        pair, side = parse_pair_side(base)
        index.setdefault(pair, {})[side] = f

    for pair, sides in index.items():
        if "BID" in sides and "ASK" in sides:
            print(f"Joining {pair} ...")
            df_join = join_bid_ask_file(sides["BID"], sides["ASK"])
            
            #Remove duplicates again
            df_join = drop_duplicate_timestamps(df_join, time_col)
            
            # Detect gaps
            short_gaps = detect_missing_periods(df_join, time_col, MIN_SHORT_GAP, MAX_SHORT_GAP)
            long_gaps  = detect_missing_periods(df_join, time_col, MIN_LONG_GAP,  MAX_LONG_GAP)
    
            # Detect invalid blocks (NaN/zero OHLC runs)
            invalid_blocks = detect_invalid_blocks(df_join, time_col, OHLC_COLS, min_block=MIN_SHORT_GAP)
            
            # Summary
            print(f"Rows after cleaning: {len(df_join)}")
            print(f"Short gaps (1m..2d): {len(short_gaps)}")
            print(f"Long gaps  (≥3d):    {len(long_gaps)}")
            print(f"Invalid blocks:      {len(invalid_blocks)}")
        
            out_path = os.path.join(joined, f"{pair}_BIDASK_JOINED.parquet")
            out_short = os.path.join(joined, f"{pair}_short_gaps.parquet")
            out_long = os.path.join(joined, f"{pair}_long_gaps.parquet")
            out_invalid = os.path.join(joined, f"{pair}_invalid_blocks.parquet")
            
            df_join.reset_index(drop=True).to_parquet(out_path, index=False)
            short_gaps.to_parquet(out_short, index=False)
            long_gaps.to_parquet(out_long, index=False)
            invalid_blocks.to_parquet(out_invalid, index=False)
            print(" Saved: df_join, short_gaps, long_gaps, invalid_blocks\n")
        else:
            print(f"Skipping {pair}: missing {'BID' if 'BID' not in sides else 'ASK'} file")

    print(f"\nAll files in folder {cleaned} have been processed successfully.")

# FEATURES + LABELS 

@dataclass
class TBParams:
    horizon_bars: int = 60        # vertical barrier in bars (e.g., 60 minutes)
    up_mult: float = 1.0          # multiplier for upper horizontal barrier
    dn_mult: float = 1.0          # multiplier for lower horizontal barrier
    vol_col: str = "vol_60"       # rolling vol column name (precomputed)
    ewma_col: str = "ewma_50"
    price_col: str = "close_mid"  # mid-price column
    time_col: str = "timestamp"        # timestamp column
    min_abs_logret: float = 0.0   # optional: set small moves to 0 label
    
def compute_spread_stats_for_file(
    path: str,
    time_col: str = "timestamp",
    max_spread_bps: float = 200.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute spread statistics for a single *_BIDASK_JOINED.parquet file.

    - Use close_bid / close_ask
    - Compute spread (price) and spread_bps = spread / mid * 1e4
    - Remove only for the statistics:
        * spread <= 0
        * spread_bps <= 0
        * spread_bps >= max_spread_bps
        * NaN

    This is diagnostic only; it does not modify inputs.
    Return a DataFrame containing spread statistics.
    """
    base = os.path.basename(path)
    pair = base.replace("_BIDASK_JOINED.parquet", "")

    df = pd.read_parquet(path)
    df = ensure_datetime_utc(df, time_col)

    required = {"close_bid", "close_ask"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: missing required columns {required - set(df.columns)}")

    # Check ASK/BID
    ask_le_bid = (df["close_ask"] <= df["close_bid"]).sum()

    # Compute spread and mid
    df["spread"] = df["close_ask"] - df["close_bid"]
    df["mid"] = 0.5 * (df["close_ask"] + df["close_bid"])
    df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4

    # Optional diagnostics
    if verbose:
        print(f"\n=== {pair} ===")
        print(df["spread"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
        print(df["spread_bps"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
        print("ASK <= BID:", ask_le_bid)
        print("NaN spread:", df["spread"].isna().sum())
        print("NaN spread_bps:", df["spread_bps"].isna().sum())

    # Filter ONLY for statistics
    total_n = len(df)

    mask_basic = (
        (df["spread"] > 0)
        & df["spread"].notna()
        & df["spread_bps"].notna()
    )
    mask_bps = df["spread_bps"] > 0
    mask_bps &= df["spread_bps"] < max_spread_bps

    mask_valid = mask_basic & mask_bps
    df_valid = df.loc[mask_valid].copy()
    valid_n = len(df_valid)
    outliers_removed = int(total_n - valid_n)

    if df_valid.empty:
        raise ValueError(f"{path}: no valid spread rows after filtering")

    stats = {}

    # Spread statistics (price)
    s = df_valid["spread"]
    stats["spread_min"]     = s.min()
    stats["spread_p01"]     = s.quantile(0.01)
    stats["spread_p05"]     = s.quantile(0.05)
    stats["spread_median"]  = s.quantile(0.50)
    stats["spread_mean"]    = s.mean()
    stats["spread_p95"]     = s.quantile(0.95)
    stats["spread_p99"]     = s.quantile(0.99)
    stats["spread_max"]     = s.max()
    stats["spread_n"]       = int(valid_n)
    stats["spread_outliers_removed"] = outliers_removed

    # Statistics for spread_bps
    sb = df_valid["spread_bps"]
    stats["spread_bps_min"] = sb.min()
    stats["spread_bps_p01"] = sb.quantile(0.01)
    stats["spread_bps_p05"] = sb.quantile(0.05)
    stats["spread_bps_median"] = sb.quantile(0.50)
    stats["spread_bps_mean"] = sb.mean()
    stats["spread_bps_p95"] = sb.quantile(0.95)
    stats["spread_bps_p99"] = sb.quantile(0.99)
    stats["spread_bps_max"] = sb.max()
    stats["spread_bps_n"]   = int(valid_n)
    stats["spread_bps_outliers_removed"] = outliers_removed

    out = pd.DataFrame(stats, index=[pair])
    return out

def compute_spread_stats_for_folder(
    input_joined: str,
    time_col: str = "timestamp",
    max_spread_bps: float = 200.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply compute_spread_stats_for_file to all the *_BIDASK_JOINED.parquet files
    in a folder and concatenate all the results.
    """
    pattern = os.path.join(input_joined, "*_BIDASK_JOINED.parquet")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")

    rows = []
    for f in sorted(files):
        print(f"Computing spread stats for {os.path.basename(f)} ...")
        try:
            row = compute_spread_stats_for_file(
                f,
                time_col=time_col,
                max_spread_bps=max_spread_bps,
                verbose=verbose,
            )
            rows.append(row)
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    if not rows:
        raise RuntimeError("No valid spread stats computed for any file.")

    stats_all = pd.concat(rows, axis=0)
    return stats_all


def validate_pipeline_diagnostics(
    df_sess: pd.DataFrame,
    events: pd.DatetimeIndex | pd.Series,
    tb: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    label_col: str = "label",
    t1_vert_col: str = "t1_vert",
    t_end_col: str = "t_end",
    side_col: str = "side",
    side_score_col: str = "side_score",
    horizon_bars: int | None = None,
    bar_minutes: int = 1,
    pair_name: str | None = None,
    print_daily_tables: bool = True,
    max_rows_table: int = 12,
) -> None:
    """
    Print a comprehensive diagnostic report for:
      - Event selection (CUSUM events)
      - TBM labeling quality
      - Primary side stability
      - Meta-labeling sanity checks

    Assumptions:
      - df_sess is session-filtered and indexed by datetime OR has time_col.
      - events contains event timestamps (t0).
      - tb is indexed by t0 and contains at least [label_col] and ideally [t_end_col] and [t1_vert_col].
      - side/meta columns can be either in df_sess (at t0 rows) or added later; function uses df_sess.reindex(tb.index).

    This is diagnostic only; it does not modify inputs.
    """
    
    
    def _print_hdr(title: str):
        line = "=" * len(title)
        print("\n" + title + "\n" + line)

    def _quantiles(x: pd.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        x = pd.Series(x).dropna().astype(float)
        if x.empty:
            return None
        return x.quantile(list(qs))

    # --- Normalize df_sess index ---
    x = df_sess.copy()
    if time_col in x.columns:
        x = x.set_index(time_col)
    x = x.sort_index()
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("df_sess must be indexed by a DatetimeIndex or provide time_col with datetimes")

    # --- Normalize events ---
    if isinstance(events, pd.Series):
        # allow a series of timestamps
        ev = pd.DatetimeIndex(events.dropna().values)
    else:
        ev = pd.DatetimeIndex(events)
    ev = ev.sort_values()

    # --- Normalize tb ---
    tb_local = tb.copy()
    if not isinstance(tb_local.index, pd.DatetimeIndex):
        raise TypeError("tb must be indexed by t0 as a DatetimeIndex")

    # Restrict to common t0 domain
    t0 = tb_local.index
    if len(t0) == 0:
        print(" tb is empty: nothing to diagnose.")
        return

    # Pair label
    tag = f" [{pair_name}]" if pair_name else ""

    # Basic sanity
    _print_hdr(f"0) INPUT SANITY{tag}")
    print(f"df_sess rows: {len(x):,}")
    print(f"events (t0) count: {len(ev):,}")
    print(f"tb rows: {len(tb_local):,}")
    print(f"tb columns: {list(tb_local.columns)}")

    # Event selection diagnostics
    _print_hdr(f" EVENT SELECTION DIAGNOSTICS{tag}")

    # Events/day
    ev_days = pd.Series(1, index=ev).groupby(ev.normalize()).sum()
    if ev_days.empty:
        print(" No events to analyze.")
    else:
        med = float(ev_days.median())
        q1 = float(ev_days.quantile(0.25))
        q3 = float(ev_days.quantile(0.75))
        iqr = q3 - q1
        print(f"Events/day: median={med:.1f}, IQR={iqr:.1f} (Q1={q1:.1f}, Q3={q3:.1f})")
        if print_daily_tables:
            print("\nEvents/day (head):")
            print(ev_days.head(max_rows_table).to_string())

    # Δt distribution
    if len(ev) >= 2:
        dt = pd.Series(ev[1:].values - ev[:-1].values).astype("timedelta64[ns]")
        dt_min = dt / np.timedelta64(1, "m")
        q = _quantiles(dt_min)
        print("\nΔt between events (minutes) quantiles:")
        if q is None:
            print("  (not available)")
        else:
            print(q.to_string())
    else:
        print("\nΔt between events: not enough events.")

    # Concurrency (approx): overlap if H known AND tb has end times
    # We compute mean concurrency over the session timeline (df_sess index).
    _print_hdr(f" CONCURRENCY (OVERLAP) APPROX{tag}")
    end_col = None
    if t_end_col in tb_local.columns:
        end_col = t_end_col
    elif "t1" in tb_local.columns:
        end_col = "t1"

    if end_col is None:
        print(" Cannot compute concurrency: tb has no end column (t_end or t1).")
    else:
        # Map intervals to positions over x.index
        idx = x.index
        ordinal = pd.Series(np.arange(len(idx)), index=idx)

        # Keep only intervals fully in idx
        t1 = tb_local[end_col]
        valid = t0.isin(idx) & t1.isin(idx)
        t0v = t0[valid]
        t1v = t1[valid]
        if len(t0v) == 0:
            print(" No intervals align to df_sess index for concurrency.")
        else:
            s = ordinal.loc[t0v].to_numpy()
            e = ordinal.loc[t1v].to_numpy()
            # Ensure e>=s
            ok = e >= s
            s = s[ok]; e = e[ok]
            conc = np.zeros(len(idx), dtype=float)
            for si, ei in zip(s, e):
                conc[int(si):int(ei)+1] += 1.0
            # Mean concurrency over times where at least one event active
            active = conc > 0
            mean_conc = float(np.mean(conc[active])) if active.any() else 0.0
            p50 = float(np.median(conc[active])) if active.any() else 0.0
            p90 = float(np.quantile(conc[active], 0.90)) if active.any() else 0.0
            print(f"Mean concurrency (active times): {mean_conc:.2f}")
            print(f"Concurrency p50={p50:.2f}, p90={p90:.2f}")
            if horizon_bars is not None:
                print(f"(Note) horizon_bars={horizon_bars}, bar_minutes={bar_minutes}")

    # TBM diagnostics
    _print_hdr(f" TBM LABELING DIAGNOSTICS{tag}")

    if label_col not in tb_local.columns:
        print(f" tb missing '{label_col}' column.")
    else:
        labs = tb_local[label_col].astype(float)

        # Label % overall
        pct_p = float((labs == 1).mean() * 100)
        pct_m = float((labs == -1).mean() * 100)
        pct_0 = float((labs == 0).mean() * 100)
        print(f"Label distribution overall: +1={pct_p:.1f}%, -1={pct_m:.1f}%, 0={pct_0:.1f}%")

        # Label % by day
        daily = pd.DataFrame({"label": labs}, index=tb_local.index)
        byday = daily.groupby(daily.index.normalize())["label"].value_counts(normalize=True).unstack(fill_value=0.0) * 100
        for c in [-1.0, 0.0, 1.0]:
            if c not in byday.columns:
                byday[c] = 0.0
        byday = byday[[-1.0, 0.0, 1.0]].rename(columns={-1.0: "%-1", 0.0: "%0", 1.0: "%+1"})
        if print_daily_tables:
            print("\nLabel % by day (head):")
            print(byday.head(max_rows_table).round(1).to_string())

    # t_end - t0 distribution
    _print_hdr(f" TBM HOLDING TIME DIAGNOSTICS{tag}")
    if (t_end_col in tb_local.columns):
        dt_hold = (tb_local[t_end_col] - tb_local.index)
        hold_min = dt_hold / np.timedelta64(1, "m")
        qh = _quantiles(hold_min)
        print("Holding time (t_end - t0) minutes quantiles:")
        if qh is None:
            print("  (not available)")
        else:
            print(qh.to_string())
        # Quick "closes too fast" signal
        if not hold_min.dropna().empty:
            pct_fast = float((hold_min <= 1.0).mean() * 100)
            print(f"% events closing within <= 1 minute: {pct_fast:.1f}%")
    else:
        print(f" tb missing '{t_end_col}' column; cannot compute holding time distribution.")

    # No-hit case check: label=0 should imply t_end == t1_vert (and not equal t0)
    _print_hdr(f" NO-HIT CASE CONSISTENCY{tag}")
    if (label_col in tb_local.columns) and (t_end_col in tb_local.columns) and (t1_vert_col in tb_local.columns):
        z = tb_local[tb_local[label_col] == 0].copy()
        if z.empty:
            print("No label=0 cases (ok if horizon is long / barriers tight).")
        else:
            eq_vert = (z[t_end_col] == z[t1_vert_col]).mean() * 100
            eq_t0 = (z[t_end_col] == z.index).mean() * 100
            print(f"Among label=0: % with t_end == t1_vert: {eq_vert:.1f}% (expected high)")
            print(f"Among label=0: % with t_end == t0:      {eq_t0:.1f}% (expected ~0)")
            if print_daily_tables:
                print("\nSample label=0 rows (head):")
                print(z[[t1_vert_col, t_end_col, label_col]].head(min(max_rows_table, len(z))).to_string())
    else:
        print(f" Need columns '{label_col}', '{t_end_col}', '{t1_vert_col}' to verify no-hit consistency.")

    # Side diagnostics
    _print_hdr(f" PRIMARY SIDE DIAGNOSTICS{tag}")

    # Side NaN %
    if side_col in x.columns:
        side_on_t0 = x[side_col].reindex(tb_local.index)
        pct_nan = float(side_on_t0.isna().mean() * 100)
        print(f"% NaN side at events (dead zone): {pct_nan:.1f}%")
    else:
        print(f" df_sess missing '{side_col}' column.")

    # Corr(side_score, label) sanity check
    if (side_score_col in x.columns) and (label_col in tb_local.columns):
        ss = x[side_score_col].reindex(tb_local.index).astype(float)
        lab = tb_local[label_col].astype(float)
        tmp = pd.concat([ss.rename("side_score"), lab.rename("label")], axis=1).dropna()
        if len(tmp) < 20:
            print("Not enough data to compute corr(side_score, label).")
        else:
            corr = float(tmp["side_score"].corr(tmp["label"]))
            print(f"Corr(side_score, label) at events (sanity, not optimization): {corr:.3f}")
    else:
        print(f" Need '{side_score_col}' in df_sess and '{label_col}' in tb.")

    # Future-dependence check (heuristic): verify that side_score does not use future by checking if it
    # changes when shifted by 1 bar at t0. This is not a proof, but a red flag detector.
    _print_hdr(f" LOOK-AHEAD HEURISTIC CHECK{tag}")
    if side_score_col in x.columns:
        ss_full = x[side_score_col].astype(float)
        # Compare ss(t0) vs ss(t0-1) distributional difference (should be different but not pathological)
        ss_t0 = ss_full.reindex(tb_local.index)
        ss_prev = ss_full.shift(1).reindex(tb_local.index)
        tmp = pd.concat([ss_t0.rename("ss_t0"), ss_prev.rename("ss_prev")], axis=1).dropna()
        if len(tmp) < 20:
            print("Not enough data for heuristic shift check.")
        else:
            # If ss is accidentally using current bar close that includes future relative to t0 definition,
            # you often see ss_t0 ~ ss_prev with weird patterns or discontinuities.
            diff = (tmp["ss_t0"] - tmp["ss_prev"]).abs()
            print("Abs(side_score(t0) - side_score(t0-1)) quantiles:")
            print(_quantiles(diff).to_string())
            # Not a strict pass/fail; you must inspect your feature engineering for shift usage.
            print("Reminder: all rolling features used at t0 should be computed with past-only data; if using close at t0, ensure that's allowed by your execution model.")
    else:
        print(f" df_sess missing '{side_score_col}' column.")



    _print_hdr(f"DONE{tag}")


def _to_utc(dt: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    """
    Convert datetime-like to tz-aware UTC consistently.
    - If tz-naive: localize as UTC
    - If tz-aware: convert to UTC
    """
    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is None:
            return dt.tz_localize("UTC")
        return dt.tz_convert("UTC")

    # Series
    s = pd.to_datetime(dt, errors="coerce")
    if hasattr(s.dt, "tz") and s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def validate_labeled_parquet(
    path: str,
    *,
    time_col: str = "t0",
    horizon_bars: int | None = None,
    bar_minutes: int = 1,
    pair_name: str | None = None,
):
    df = pd.read_parquet(path)

    if time_col not in df.columns:
        raise KeyError(f"{time_col} not found in {path}")

    # Ensure timestamp column is datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    # Force UTC tz-aware index
    idx = pd.DatetimeIndex(df[time_col])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.drop(columns=[time_col]).copy()
    df.index = idx
    df.index.name = "t0"

    # Force UTC on tb datetime columns if present
    for c in ["t_end", "t1_vert", "t1"]:
        if c in df.columns:
            df[c] = _to_utc(df[c])

    # Reconstruct events and tb
    events = df.index

    required_tb_cols = ["label"]
    for c in ["t_end", "t1_vert", "t1"]:
        if c in df.columns:
            required_tb_cols.append(c)

    tb = df[required_tb_cols].copy()
    tb.index.name = "t0"

    validate_pipeline_diagnostics(
        df_sess=df,
        events=events,
        tb=tb,
        horizon_bars=horizon_bars,
        bar_minutes=bar_minutes,
        pair_name=pair_name or path,
    )

def print_cusum_filter_stats(
    idx_all: pd.DatetimeIndex,
    elig_mask: pd.Series,
    events: pd.DatetimeIndex,
    *,
    pair: str = "",
):
    n_all = len(idx_all)
    n_elig = int(elig_mask.sum())
    n_events = len(events)

    print(f"\n[CUSUM FILTER STATS]{' ' + pair if pair else ''}")
    print(f"Total bars (full): {n_all:,}")
    print(f"Eligible bars (session+runway): {n_elig:,}  ({n_elig/n_all*100:.1f}%)")
    print(f"CUSUM events: {n_events:,}  ({(n_events/max(n_elig,1))*100:.2f}% of eligible bars)")

    if n_events >= 2:
        dt = (events[1:] - events[:-1]) / np.timedelta64(1, "m")
        qs = np.quantile(dt, [0.1, 0.25, 0.5, 0.75, 0.9])
        print("Δt events (min) q10/q25/q50/q75/q90:", np.round(qs, 2))



def add_mid_and_returns(df: pd.DataFrame,
                        time_col="timestamp") -> pd.DataFrame:
    
    #Adds mid-ohlc and log-returns on close_mid price.
    #Input columns: *_bid and *_ask OHLC.
    
    df = df.copy()
    for fld in ["open","high","low","close"]:
        b, a = f"{fld}_bid", f"{fld}_ask"
        if b in df.columns and a in df.columns:
            df[f"{fld}_mid"] = 0.5*(df[b] + df[a])
    # log-price & returns 
    if "close_mid" in df.columns:
        df["log_mid"] = np.log(df["close_mid"].replace(0, np.nan))
        df["ret_1"] = df["log_mid"].diff(1)
        df["ret_5"] = df["log_mid"].diff(5)
    #rolling volatility on 1 minute returns (std dei ret_1)
    df["vol_60"] = df["ret_1"].rolling(60).std(ddof=1)
    #ewma
    df["ewma_50"] = df["ret_1"].ewm(span=50, adjust=False).std(bias=False)
    # winsorize returns (optional)
    #df["ret_1_w"] = zscore_winsorize(df["ret_1"], z=5.0)
    return df


def ewma_vol(series: pd.Series, span: int = 50) -> pd.Series:
    #Estimate the volatility using EWMA on the returns
    #Return a time series of estimated standard deviations.
    x = series.astype(float)
    return x.ewm(span=span, adjust=False).std()


def session_mask_utc(
    idx: pd.DatetimeIndex,
    start_h: int = 8,
    end_h: int = 18,
) -> pd.Series:
    """
    True if timestamp is between [start_h, end_h) in UTC.
    idx must be tz-aware in UTC.
    """
    # hour in 0..23
    h = idx.hour
    return (h >= start_h) & (h < end_h)


def session_mask_h(
    idx: pd.DatetimeIndex,
    horizon_bars: int,
    bar_minutes: int = 1,
    session_start_h: int = 8,
    session_end_h: int = 18,
) -> pd.Series:
    """
    True if:
      - timestamp in [08:00, 18:00)
      - e t + H*bar_minutes <= 18:00 of the same day.
    """
    H_td = pd.Timedelta(minutes=horizon_bars * bar_minutes)
    day = idx.normalize()
    session_start = day + pd.Timedelta(hours=session_start_h)
    session_end = day + pd.Timedelta(hours=session_end_h)

    in_session = (idx >= session_start) & (idx < session_end)
    runway_ok = (idx + H_td) <= session_end
    return in_session & runway_ok


def cusum_events(
    r: pd.Series,
    sigma: pd.Series,
    k: float = 3.0,
    min_spacing: pd.Timedelta | None = None,
    sigma_floor: float | None = None,
    return_direction: bool = False,
) -> pd.DatetimeIndex | pd.DataFrame:
    """
    Symmetric CUSUM with time-varying threshold:
        h_t = k * sigma_t

    Parameters
    ----------
    r : pd.Series
        Log-returns r_t (e.g., df['ret_1']), indexed by timestamp.
    sigma : pd.Series
        Volatility estimate sigma_t (e.g., EWMA std), aligned to r index.
    k : float
        Multiplier for threshold (e.g., 3.0).
    min_spacing : pd.Timedelta | None
        Optional minimum time between emitted events.
    sigma_floor : float | None
        Optional floor for sigma to avoid tiny thresholds (e.g., set to training p05).
        If None, no floor is applied.
    return_direction : bool
        If True, returns DataFrame with event_time and direction (+1/-1).

    Returns
    -------
    pd.DatetimeIndex or pd.DataFrame
        Event timestamps (and optionally trigger direction).
    """

    # Align and clean
    r = r.astype(float)
    sigma = sigma.astype(float)
    x = pd.concat([r, sigma], axis=1, keys=["r", "sigma"]).dropna()
    if x.empty:
        if return_direction:
            return pd.DataFrame({"event_time": [], "direction": []})
        return pd.DatetimeIndex([], name="event_time")

    # Optional sigma floor to prevent micro-noise event bursts
    if sigma_floor is not None:
        x["sigma"] = x["sigma"].clip(lower=float(sigma_floor))

    # Threshold series
    h = k * x["sigma"]
    h = h.replace([np.inf, -np.inf], np.nan).dropna()

    # Keep only timestamps where both r and h are valid
    x = x.loc[h.index]
    if x.empty:
        if return_direction:
            return pd.DataFrame({"event_time": [], "direction": []})
        return pd.DatetimeIndex([], name="event_time")

    s_pos, s_neg = 0.0, 0.0
    ev_times, ev_dir = [], []
    last_t = None

    for t, row in x.iterrows():
        rt = row["r"]
        ht = h.at[t]

        # Skip if ht invalid or non-positive
        if not np.isfinite(ht) or ht <= 0:
            continue

        s_pos = max(0.0, s_pos + rt)
        s_neg = min(0.0, s_neg + rt)
        trig = 0
        if s_pos > ht:
            trig = +1; s_pos = 0.0; s_neg = 0.0
        elif s_neg < -ht:
            trig = -1; s_pos = 0.0; s_neg = 0.0

        if trig != 0:
            if min_spacing is None or last_t is None or (t - last_t) >= min_spacing:
                ev_times.append(t)
                ev_dir.append(trig)
                last_t = t

    if return_direction:
        return pd.DataFrame({"event_time": ev_times, "direction": ev_dir})
    return pd.DatetimeIndex(ev_times, name="event_time")


def cusum_test(
    x: pd.Series,
    h: float,
):
    x = x.dropna() 
    print(x.abs().quantile([0.5, 0.9, 0.99]))


def triple_barrier(
    df: pd.DataFrame,
    event_times: pd.DatetimeIndex,
    params: TBParams,
    session_end_h: int = 18,         # end UTC session
    enforce_session_cap: bool = True # if False, fallback to H-bar TBM  
) -> pd.DataFrame:

    """Define TBM events and assign -1/0/1 labels."""
    
    df = df[[params.time_col, params.price_col, params.vol_col]].dropna().copy()
    df = df.sort_values(params.time_col).reset_index(drop=True)
    df.set_index(params.time_col, inplace=True)

    ev = event_times.intersection(df.index)
    if len(ev) == 0:
        return (
            pd.DataFrame(columns=["t1_vert", "t_end", "label"])
              .astype({"label": "int8"})
        )

    p = df[params.price_col].astype(float)
    vol = df[params.vol_col].astype(float)
    log_p = np.log(p.replace(0.0, np.nan))

    idx = df.index
    idx_pos = {t: i for i, t in enumerate(idx)}
    n = len(df)

    H = int(params.horizon_bars)
    upm, dnm = float(params.up_mult), float(params.dn_mult)

    out_t1_vert, out_t_end, out_lab = [], [], []

    for t0 in ev:
        i = idx_pos[t0]
        if i >= n - 1 or np.isnan(log_p.iat[i]) or np.isnan(vol.iat[i]):
            out_t1_vert.append(t0)
            out_t_end.append(t0)
            out_lab.append(0)
            continue

        # candidate for H bars
        j_bar = min(n - 1, i + H)

        # 2) candidate for "session end"
        if enforce_session_cap:
            day = t0.normalize()
            t_session_end = day + pd.Timedelta(hours=session_end_h)

            # last timestamp <= session_end
            # searchsorted(..., side="right") -> first index > session_end
            j_sess = idx.searchsorted(t_session_end, side="right") - 1

            # j_sess can end before i if t0 is close to the start of the dataset 
            j_sess = max(j_sess, i)

            # real j_end: min between horizon (bar) and session cap
            j_end = min(j_bar, j_sess)
        else:
            j_end = j_bar

        # t1_vert = effectve timestamp of the vertical barrier 
        t1_vert = idx[j_end]

        # Horizontal Barriers in log-space
        v = max(vol.iat[i], 1e-12)
        log_up = log_p.iat[i] + upm * v
        log_dn = log_p.iat[i] - dnm * v

        label = 0
        j_hit = None

        for j in range(i + 1, j_end + 1):
            lpj = log_p.iat[j]
            if np.isnan(lpj):
                continue
            if lpj >= log_up:
                label = +1
                j_hit = j
                break
            if lpj <= log_dn:
                label = -1
                j_hit = j
                break

        t_end = idx[j_hit] if j_hit is not None else t1_vert

        out_t1_vert.append(t1_vert)
        out_t_end.append(t_end)
        out_lab.append(label)

    tb = pd.DataFrame(
        {"t1_vert": out_t1_vert, "t_end": out_t_end, "label": out_lab},
        index=ev
    )
    tb.index.name = "t0"
    tb["label"] = tb["label"].astype("int8")
    return tb



def primary_side_fn(df: pd.DataFrame,
                    time_col: str = "timestamp",
                    price_col: str = "close_mid",
                    ret_col: str = "ret_1",        # 1 minute log-return in the pipeline
                    vol_col: str = "vol_60",       # rolling std of ret
                    lookback: int = 60,            # magnitude of the momentum 
                    z_floor: float = 0.5,          # dead banda morta for classification
                    clip_z: float = 3.0            # continuous score limit
                   ) -> pd.DataFrame:
    """
    Deterministic primary signal using only past data.
    Returns a DataFrame aligned on df[time_col] with:
      - 'side_score': continuous z-scored momentum (for regression / ranking)
      - 'side': discrete signal in {+1, -1} with a dead-zone |z| < z_floor -> NaN
    No look-ahead: all rolling ops use only history.
    """
    x = df.copy()
    # Momentum on log-price
    if "log_mid" not in x.columns and price_col in x.columns:
        x["log_mid"] = np.log(x[price_col].replace(0, np.nan))
    mom = x["log_mid"].diff(lookback)

    #Scaled volatility: takes vol_col (rolling std dei ret_1) * sqrt(lookback) ~ std del mom
    vol = x[vol_col] * np.sqrt(lookback)
    vol = vol.replace(0.0, np.nan)

    # Z-score of the momentum 
    z = (mom / vol).replace([np.inf, -np.inf], np.nan)
    z = z.clip(-clip_z, clip_z)

    # Discrete signal with dead band 
    side = np.sign(z)
    side[np.abs(z) < z_floor] = np.nan   # no trade when signal is weak

    out = pd.DataFrame({
        time_col: x[time_col],
        "side_score": z,   # continuous -> for regression
        "side": side       # discrete -> for classificazione/meta-labeling
    })
    return out.set_index(time_col)

def primary_side_mean_reversion(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    price_col: str = "close_mid",
    vol_col: str = "vol_60",
    lookback: int = 20,      
    z_floor: float = 0.5,
    clip_z: float = 3.0,
    use_log: bool = True,
) -> pd.DataFrame:
    """
    Mean-reversion primary side:
      - Computes z-scored momentum over 'lookback'
      - Takes the opposite sign (contrarian)
    """

    x = df.copy()

    if use_log:
        if "log_mid" not in x.columns and price_col in x.columns:
            x["log_mid"] = np.log(x[price_col].replace(0, np.nan))
        px = x["log_mid"]
    else:
        px = x[price_col].astype(float)

    # Past-only momentum
    mom = px.diff(lookback)

    # Scale by recent volatility (std of 1-min returns) * sqrt(lookback)
    vol = x[vol_col].astype(float) * np.sqrt(lookback)
    vol = vol.replace(0.0, np.nan)

    z = (mom / vol).replace([np.inf, -np.inf], np.nan).clip(-clip_z, clip_z)

    # Contrarian: reverse sign
    z_mr = -z

    side = np.sign(z_mr)
    side[np.abs(z_mr) < z_floor] = np.nan

    out = pd.DataFrame(
        {time_col: x[time_col], "side_score": z_mr, "side": side}
    ).set_index(time_col)

    return out




def compute_sample_weights(
    tb_df: pd.DataFrame,
    rets: pd.Series
) -> pd.Series:
    """
    Compute sample weights following López de Prado (2018) / Arian et al. (2024).

    tb_df: DataFrame con index = t0 e colonna 't1'
    rets:  Serie di log-returns r_{t-1,t}, indicizzata per timestamp
    """

    # Align t0 e t1
    t0 = tb_df.index
    
    if "t_end" in tb_df.columns:
        t1 = tb_df["t_end"]
    elif "t1" in tb_df.columns:
        t1 = tb_df["t1"]
    else:
        raise KeyError("tb_df must contain either 't_end' or 't1'")

    # Order rets and make sure it is float
    rets = rets.sort_index().astype(float)

    # Keep only events whose t0 and t1 are in the rets domain
    valid = t0.isin(rets.index) & t1.isin(rets.index)
    t0 = t0[valid]
    t1 = t1[valid]
    if len(t0) == 0:
        return pd.Series(dtype=float)

    # Map time → entire index
    all_times = rets.index.to_list()
    ordinal = {t: i for i, t in enumerate(all_times)}
    rets_array = rets.to_numpy()

    # Entire intervals [s_i, e_i] for each event
    intervals = []
    for ti0, ti1 in zip(t0, t1):
        s = ordinal.get(ti0, None)
        e = ordinal.get(ti1, None)
        if s is None or e is None or e < s:
            intervals.append(None)
        else:
            intervals.append((s, e))

    valid_events = [iv is not None for iv in intervals]
    t0 = t0[valid_events]
    intervals = [iv for iv in intervals if iv is not None]

    I = len(intervals)
    if I == 0:
        return pd.Series(dtype=float)

    # Concurrency c_t
    T = len(all_times)
    concurrency = np.zeros(T, dtype=float)
    for (s, e) in intervals:
        concurrency[s:e+1] += 1.0
    concurrency[concurrency == 0] = np.nan

    # Raw weights ~w_i
    raw_weights = np.zeros(I, dtype=float)
    for idx, (s, e) in enumerate(intervals):
        r_slice = rets_array[s:e+1]
        c_slice = concurrency[s:e+1]
        ok = ~np.isnan(c_slice)
        if not ok.any():
            continue
        r_adj = r_slice[ok] / c_slice[ok]
        raw_weights[idx] = np.abs(r_adj.sum())

    if np.all(raw_weights == 0):
        w = np.ones(I, dtype=float)
    else:
        raw_sum = raw_weights.sum()
        w = raw_weights * (I / raw_sum)
    
    p99 = np.percentile(w, 99)
    w = np.clip(w, 0, p99)

    s = w.sum()
    if s > 0:
        w = w * (I / s)

    return pd.Series(w, index=t0)


def feature_folder(
    joined: str,
    labeled: str,
    tb_params: Optional[TBParams] = None,
    cusum_h: float = 3.0,
    cusum_min_spacing: str | None = "2min",
    horizon: int | None = None):

    """
    Build event-driven feature sets:
      - Input:  *_BIDASK_JOINED.parquet  (joined, cleaned BID/ASK OHLC)
      - Output: *_LABELED.parquet       (one row per event t0 with features + TBM labels)

    Notes:
    Features are computed on the full time series, then sliced at event times (t0).
    CUSUM defines event timestamps only; TBM assigns {-1,0,+1} labels at those t0.
    """

    os.makedirs(labeled, exist_ok=True)
    files = glob.glob(os.path.join(joined, "*_BIDASK_JOINED.parquet"))
    if not files:
        print(f"No JOINED files found in {joined}")
        return

    if tb_params is None:
        tb_params = TBParams()
        
    spacing_td = pd.Timedelta(cusum_min_spacing) if cusum_min_spacing else None

    for f in files:
        base = os.path.basename(f)
        print(f"\nFeaturizing {base} ...")

        #Load & time index
        df = pd.read_parquet(f)
        if tb_params.time_col not in df.columns:
            if "date" in df.columns:
                df = df.rename(columns={"date": tb_params.time_col})
            else:
                raise KeyError(
                    f"Time column '{tb_params.time_col}' not found in {base}. "
                    f"Available columns: {df.columns.tolist()}"
                )
        df = ensure_datetime_utc(df, tb_params.time_col)
        df = df[(df[tb_params.time_col] >= start_s) & (df[tb_params.time_col] <= end_s)]
        df = df.sort_values(tb_params.time_col).reset_index(drop=True)

        #Mid, log-returns, rolling vol (vol_60)
        df = add_mid_and_returns(df, time_col=tb_params.time_col)

        if "log_mid" not in df.columns or "ret_1" not in df.columns:
            print("⚠️ Missing log_mid/ret_1, skipping file.")
            continue
        
        df = df.set_index(tb_params.time_col).sort_index()
        
        rets = df["ret_1"].astype(float)
        
        if "ewma_50" in df.columns:
            sigma = df["ewma_50"].astype(float)
        else:
            sigma = rets.ewm(span=50, adjust = False).std(bias=False)
        
        sigma_floor = sigma.quantile(0.05)
        
        elig = session_mask_h(
            idx=df.index,
            horizon_bars=tb_params.horizon_bars,
            bar_minutes=1,          # 1-min bars
            session_start_h=8,
            session_end_h=18,
        )

        rets_el = rets.loc[elig]
        sigma_el = sigma.loc[elig]
        
        
        if rets_el.empty or sigma_el.empty:
            print("⚠️ No eligible timestamps (session+runway). Saving basic features only.")
            df_tmp = df.loc[session_mask_utc(df.index, 8, 18)].reset_index()
            side_df = primary_side_mean_reversion(
                df_tmp,
                time_col=tb_params.time_col,
                lookback=15,
                z_floor=0.5,
            )
            df_tmp = df_tmp.merge(side_df.reset_index(), on=tb_params.time_col, how="left")
            out_path = os.path.join(
                labeled, base.replace("_BIDASK_JOINED.parquet", "_LABELED.parquet")
            )
            df_tmp.to_parquet(out_path, index=False)
            print(f"✅ Saved (no eligible timestamps): {out_path}")
            continue
            
        events = cusum_events(
            r=rets_el,
            sigma=sigma_el,
            k=cusum_h,              # k (es. 3.0)
            min_spacing=spacing_td,
            sigma_floor=sigma_floor,
            return_direction=False
        )
        if len(events) == 0:
            print("⚠️ No CUSUM events found (eligible domain); saving basic features only.")
            df_tmp = df.loc[session_mask_utc(df.index, 8, 18)].reset_index()
            side_df = primary_side_mean_reversion(
                df_tmp,
                time_col=tb_params.time_col,
                lookback=15,
                z_floor=0.5,
            )
            df_tmp = df_tmp.merge(side_df.reset_index(), on=tb_params.time_col, how="left")
            out_path = os.path.join(
                labeled, base.replace("_BIDASK_JOINED.parquet", "_LABELED.parquet")
            )
            df_tmp.to_parquet(out_path, index=False)
            print(f"✅ Saved (no eligible events): {out_path}")
            continue
        
        print(f"Found {len(events)} CUSUM events (generated inside session+runway domain).")
 
        print_cusum_filter_stats(
            idx_all=df.index,
            elig_mask=elig,
            events=events,
            pair=base.replace("_BIDASK_JOINED.parquet",""),
        )

    
        # dataset session-only
        df_sess = df.loc[session_mask_utc(df.index, 8, 18)].copy()
        
        # Primary side on session
        df_reset = df_sess.reset_index()
        side_df = side_df = primary_side_mean_reversion(
            df_reset,
            time_col=tb_params.time_col,
            lookback=15,
            z_floor=0.5,
        )

        df_reset = df_reset.merge(side_df.reset_index(), on=tb_params.time_col, how="left")
        df_sess = df_reset.set_index(tb_params.time_col).sort_index()
        
        # TBM su sessione con eventi già validi
        tb = triple_barrier(df_sess.reset_index(), events, tb_params)
        
        if tb.empty:
            print("⚠️ No TB labels generated.")
            continue
        
        df_sess.loc[tb.index, "t1_vert"] = tb["t1_vert"].values
        df_sess.loc[tb.index, "t_end"]   = tb["t_end"].values
        df_sess.loc[tb.index, "label"]   = tb["label"].astype(float).values
        
        # Sample weights: t_end as interval end
        rets = df_sess["ret_1"].astype(float)
        
        weights = compute_sample_weights(tb, rets)
        
        df_sess.loc[weights.index, "weight"] = weights.values
        
        # Save only event rows (t0) 
        df_out = df_sess.loc[tb.index].reset_index()

        
        out_path = os.path.join(
            labeled, base.replace("_BIDASK_JOINED.parquet", "_LABELED.parquet")
        )
        df_out.to_parquet(out_path, index=False)
        print(f"✅ Saved LABELED events: {out_path}")
        validate_pipeline_diagnostics(
            df_sess=df_sess,          
            events=tb.index,          
            tb=tb,                    
            horizon_bars=horizon,     
            bar_minutes=1,            
            pair_name=base.replace("_BIDASK_JOINED.parquet", ""),
        )
        

def main():    
    
    BASE_DIR = Path("C:/Users/enric/Desktop/tesi/gaps.py").resolve().parent
    DATA_DIR = BASE_DIR / "data"
    
    #This dictionary contains input and output folders, 
    #time is used to recognise the time column while suffix is the suffix used to recognise the datasets.
    
    folders = [
        {
            "cleaned":  DATA_DIR / "dukascopy" / "cleaned",
            "joined": DATA_DIR / "dukascopy" / "joined",
            "labeled": DATA_DIR / "dukascopy" / "labeled",
            "time" : "timestamp",
            "suffix": "*_CLEAN.parquet",
        },
        {
            "cleaned":  DATA_DIR / "ibkr" / "cleaned",
            "joined": DATA_DIR / "ibkr" / "joined",
            "labeled": DATA_DIR / "ibkr" / "labeled",
            "time" : "timestamp",
            "suffix": "*_CLEAN.parquet",
        }
    ]
    
    
    #Feature engineering + CUSUM + Triple Barrier + meta labeling
    params = TBParams(
        horizon_bars=60,          
        up_mult=5.0,
        dn_mult=5.0,
        vol_col="vol_60",
        price_col="close_mid",
        time_col="timestamp",
        min_abs_logret=0.0
    )

    
    import cProfile
    import pstats
    
    with cProfile.Profile() as pr:
        for folder in folders:
            
            cleaned = folder["cleaned"]
            joined = folder["joined"]
            labeled = folder["labeled"]
            time_col = folder["time"]
            
            join_folder(cleaned, joined, time_col)
            
            feature_folder(
                joined=joined,
                labeled=labeled,
                tb_params=params,
                cusum_h=5.0,                 #z-score amount
                cusum_min_spacing="15min",   # minimum distance between events
                horizon=60
            )

  
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(20)


        
if __name__ == "__main__":
    main() 
    
    
"""
    # Meta-label diagnostics
    _print_hdr(f" META-LABEL DIAGNOSTICS{tag}")

    # meta_y balance
    if meta_y_col in x.columns:
        my = x[meta_y_col].reindex(tb_local.index)
        myv = my.dropna().astype(float)
        if myv.empty:
            print(" meta_y is all NaN at events.")
        else:
            p1 = float((myv == 1).mean() * 100)
            p0 = float((myv == 0).mean() * 100)
            print(f"meta_y class balance (non-NaN): 1={p1:.1f}%, 0={p0:.1f}%")
    else:
        # allow meta_y passed only as tb column
        if meta_y_col in tb_local.columns:
            myv = tb_local[meta_y_col].dropna().astype(float)
            p1 = float((myv == 1).mean() * 100) if not myv.empty else np.nan
            p0 = float((myv == 0).mean() * 100) if not myv.empty else np.nan
            print(f"meta_y class balance (from tb): 1={p1:.1f}%, 0={p0:.1f}%")
        else:
            print(f"meta_y not found in df_sess or tb as '{meta_y_col}'.")

    # Relationship meta_y vs |side_score|
    _print_hdr(f"meta_y vs |side_score|{tag}")
    if (side_score_col in x.columns):
        ss = x[side_score_col].reindex(tb_local.index).astype(float)
        # meta_y from df_sess if present, else tb
        if meta_y_col in x.columns:
            my = x[meta_y_col].reindex(tb_local.index)
        elif meta_y_col in tb_local.columns:
            my = tb_local[meta_y_col]
        else:
            my = None

        if my is None:
            print(" Cannot compute meta_y vs |side_score|: meta_y missing.")
        else:
            tmp = pd.concat([my.rename("meta_y"), ss.abs().rename("|side_score|")], axis=1).dropna()
            if len(tmp) < 30:
                print("Not enough labeled data for meta_y vs |side_score| check.")
            else:
                # Bin by |side_score|
                tmp["bin"] = pd.qcut(tmp["|side_score|"], q=5, duplicates="drop")
                grp = tmp.groupby("bin")["meta_y"].mean() * 100
                print("P(meta_y=1) by |side_score| quintile (in %):")
                print(grp.round(1).to_string())
    else:
        print(f"df_sess missing '{side_score_col}' column.")
        

"""