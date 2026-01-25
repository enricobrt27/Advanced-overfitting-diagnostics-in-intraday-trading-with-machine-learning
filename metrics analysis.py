# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:36:51 2025

@author: enric


Overfitting Diagnostics

Input  : tested directory (OOS returns matrices M from Script 4)
Output : analyzed directory

For each M file:
- Run CSCV (Combinatorial Symmetric Cross-Validation) to estimate:
    * PBO (Probability of Backtest Overfitting)
    * Performance degradation (Spearman correlation IS vs OOS)
    * Probability of loss (OOS < 0)
    * KS test vs Uniform for relative OOS ranks
- Compute Sharpe and Sortino ratios per strategy (column),
  using daily equity-curve returns.
- Save:
    * CSCV raw results per file
    * Aggregated metrics per file
"""

import os
import glob
import math
from typing import List, Dict

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, ks_2samp, skew, kurtosis, norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EULER_GAMMA = 0.5772156649
RF_RATE = 0.0384
PERIODS_PER_YEAR = 252

# --------------------------
# Directories
# --------------------------

tested = r"C:/Users/enric/Desktop/dataset tesi/dataset dukascopy/tested prova 2"  # input
analyzed = r"C:/Users/enric/Desktop/dataset tesi/dataset dukascopy/analyzed"      # output
analyzed_costs = r"C:/Users/enric/Desktop/dataset tesi/dataset dukascopy/analyzedfee" 
os.makedirs(analyzed, exist_ok=True)


# --------------------------------------------------------------------
# Helper functions (CSCV)
# --------------------------------------------------------------------

def split(M: pd.DataFrame, s: int, drop_remainder: bool = False) -> List[pd.DataFrame]:
    """
    Split the T x N returns matrix M into s contiguous subsets along the time axis.
    Returns a list of DataFrames, each a chunk of M.
    """
    T = len(M)
    half = s // 2
    base = T // s
    rem = T % s
    subsets = []
    start = 0

    for i in range(s):
        extra = 1 if i < rem and not drop_remainder else 0
        size = base + extra
        stop = start + size
        subsets.append(M.iloc[start:stop].copy())
        start += size

    combinations = math.comb(s, half)
    # print(f"\nSplitting into {s} subsets: CSCV will consider {combinations} combinations.")
    return subsets


def cscv_test_generator(subsets: List[pd.DataFrame]):
    """
    Generate (combo_indices, J_train, J_test) pairs:
    - For each combination of half subsets, build J_train as the concat of chosen subsets,
      and J_test as the concat of remaining subsets.
    """
    import itertools

    s = len(subsets)
    half = s // 2
    indices = list(range(s))

    for combo in itertools.combinations(indices, half):
        J_train = pd.concat([subsets[i] for i in combo], axis=0, ignore_index=True)
        complement = [i for i in indices if i not in combo]
        J_test = pd.concat([subsets[i] for i in complement], axis=0, ignore_index=True)
        yield combo, J_train, J_test


def stats_subsets(J_train: pd.DataFrame, J_test: pd.DataFrame):
    """
    Given training and test matrices (rows = time, cols = strategies),
    compute:
    - best_IS: index of the best strategy in-sample (max mean return)
    - perf_IS: in-sample mean of that strategy
    - perf_OOS: out-of-sample mean of that strategy
    - relative_rank: rank of that strategy OOS, / (N+1)
    - logit: log(relative_rank / (1 - relative_rank))
    """
    perf_IS = J_train.mean()
    perf_OOS = J_test.mean()

    best_IS = perf_IS.idxmax()
    rank_OOS = perf_OOS.rank(ascending=False)[best_IS]
    relative_rank = rank_OOS / (len(perf_OOS) + 1)

    # Avoid division by zero or infinite logit
    eps = 1e-8
    relative_rank = min(max(relative_rank, eps), 1 - eps)

    logit = np.log(relative_rank / (1 - relative_rank))

    return best_IS, perf_IS[best_IS], perf_OOS[best_IS], relative_rank, logit


def cscv_pipeline(M: pd.DataFrame, S: int = 8, max_iter: int = None) -> pd.DataFrame:
    """
    Run CSCV on matrix M:
    - Split M into S subsets.
    - For each combination of half subsets:
        * choose best IS strategy
        * evaluate its OOS rank and compute logit
    Return a DataFrame with results per combo.
    """
    results = []
    subsets = split(M, S)
    gen = cscv_test_generator(subsets)

    for idx, (combo, J_train, J_test) in enumerate(gen):
        best_IS, perf_IS, perf_OOS, rel_rank, logit = stats_subsets(J_train, J_test)
        results.append({
            "combo": combo,
            "best_IS": best_IS,
            "perf_IS": perf_IS,
            "perf_OOS": perf_OOS,
            "relative_rank_OOS": rel_rank,
            "logit": logit
        })
        if max_iter is not None and idx + 1 >= max_iter:
            break

    df_results = pd.DataFrame(results)
    return df_results


def evaluate_cscv(df_results: pd.DataFrame) -> Dict:
    """
    Compute summary metrics from CSCV output:
    - PBO: probability that relative rank OOS > 0.5
    - perf_degradation: Spearman correlation perf_IS vs perf_OOS
    - prob_loss: probability that perf_OOS < 0
    - stochastic_dom: KS test vs Uniform(0,1)
    """
    metrics = {}

    # 1. PBO
    metrics["PBO"] = float(np.mean(df_results["relative_rank_OOS"] > 0.5))

    # 2. Performance degradation (Spearman)
    rho, _ = spearmanr(df_results["perf_IS"], df_results["perf_OOS"])
    metrics["perf_degradation"] = float(rho)

    # 3. Probability of loss
    metrics["prob_loss"] = float(np.mean(df_results["perf_OOS"] < 0))

    # 4. KS test vs Uniform
    dist_oos = df_results["relative_rank_OOS"].values
    dist_uniform = np.random.uniform(0, 1, size=len(dist_oos))
    stat, pval = ks_2samp(dist_oos, dist_uniform)
    metrics["stochastic_dom"] = {
        "ks_stat": float(stat),
        "pval": float(pval)
    }

    return metrics


def plot_logit_histogram(df_results: pd.DataFrame, out_path: str, bins: int = 20):
    """
    Plot histogram of CSCV logit values; save to file.
    """
    metrics = evaluate_cscv(df_results)
    pbo = round(metrics["PBO"], 3)

    plt.figure(figsize=(10, 5))
    plt.hist(df_results["logit"], bins=bins, alpha=0.7, edgecolor="black")
    proxy = Line2D([0], [0], linestyle='-', label=f'PBO ≈ {pbo}')
    logit0 = plt.axvline(0, color="red", linestyle="--", label="Logit = 0")
    plt.xlabel("Logit values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Logit Values (CSCV)")
    plt.legend(handles=[proxy, logit0], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------------------------------------------------------
# Risk / Performance metrics (equity-curve based)
# --------------------------------------------------------------------

def build_daily_equity_returns(
    event_returns: np.ndarray,
    timestamps: np.ndarray,
    freq: str = "1D"
) -> pd.Series:
    """
    Converte i ritorni per evento in ritorni giornalieri (o altra frequenza).

    - event_returns: array di ritorni per trade/evento.
    - timestamps: array di timestamp corrispondenti (stessa lunghezza).
    - freq: frequenza di resampling, di default '1D' (giornaliera).

    Giorni senza trade -> ritorno 0.
    """
    s = pd.Series(event_returns, index=pd.to_datetime(timestamps))
    daily = s.resample(freq).sum()
    daily = daily.fillna(0.0)
    return daily


def sharpe_from_equity_curve(
    daily_returns: pd.Series,
    rf_annual: float = RF_RATE,
    periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """
    Sharpe ratio annualizzato calcolato sui ritorni giornalieri dell'equity curve.
    """
    r = pd.Series(daily_returns).astype(float)
    if len(r) < 2:
        return 0.0

    rf_daily = rf_annual / periods_per_year
    excess = r - rf_daily

    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma <= 0:
        return 0.0

    return float(mu / sigma * np.sqrt(periods_per_year))


def sortino_from_equity_curve(
    daily_returns: pd.Series,
    rf_annual: float = RF_RATE,
    periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """
    Sortino ratio annualizzato calcolato sui ritorni giornalieri dell'equity curve.
    """
    r = pd.Series(daily_returns).astype(float)
    if len(r) < 2:
        return 0.0

    rf_daily = rf_annual / periods_per_year
    excess = r - rf_daily

    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd <= 0:
        return 0.0

    avg_excess = excess.mean()
    return float(avg_excess / dd * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    T: int,
    skew_val: float = 0.0,
    kurt_val: float = 3.0
) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012),
    applicato a Sharpe annualizzato usando T osservazioni (es. giorni).
    """
    if T <= 1:
        return np.nan

    num = (sr_hat - sr_benchmark) * np.sqrt(T - 1)
    den = np.sqrt(1 - skew_val * sr_hat + ((kurt_val - 1.0) / 4.0) * (sr_hat ** 2))

    if den <= 0:
        return np.nan

    return float(norm.cdf(num / den))


def deflated_sharpe_ratio(
    sr_hat: float,
    sr_trials: np.ndarray,
    T: int,
    skew_val: float,
    kurt_val: float,
    sr0: float = 0.0
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014), versione generale.

    sr_hat     : Sharpe annualizzato della strategia da valutare.
    sr_trials  : array degli Sharpe annualizzati di TUTTE le strategie testate (backtest).
    T          : numero di osservazioni (es. giorni) per la strategia.
    skew_val, kurt_val : skewness e kurtosis (totale, non 'excess') dei rendimenti (daily).
    sr0        : Sharpe 'di base' (tipicamente 0).
    """
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[~np.isnan(sr_trials)]
    N = len(sr_trials)

    # Se non ci sono vere "multiple trials", ritorna al PSR standard
    if N <= 1:
        return probabilistic_sharpe_ratio(sr_hat, sr0, T, skew_val, kurt_val)

    var_sr = np.var(sr_trials, ddof=1)
    sigma_sr = np.sqrt(var_sr)

    # Expected max di N standard normal (approssimazione)
    z_max = ((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / N) +
             EULER_GAMMA * norm.ppf(1.0 - 1.0 / (N * np.e)))

    sr_star = sr0 + sigma_sr * z_max

    return probabilistic_sharpe_ratio(sr_hat, sr_star, T, skew_val, kurt_val)


# --------------------------------------------------------------------
# Main analysis loop
# --------------------------------------------------------------------

def list_M_files(trained_dir: str) -> List[str]:
    """Return list of *_M.parquet (or *_M.csv) files from TRAINED directory."""
    pattern = os.path.join(trained_dir, "*_H*_M.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        # fallback eventuale a csv, se necessario
        pattern_csv = os.path.join(trained_dir, "*_H*_M.csv")
        files = sorted(glob.glob(pattern_csv))

    return files


def process_M_file(path: str, S: int = 8, max_iter: int = None):
    """
    Per una singola matrice di rendimenti OOS M:
    - esegue CSCV
    - calcola metriche riassuntive (PBO, perf_degradation, prob_loss, KS)
    - calcola Sharpe / Sortino / DSR per ciascuna strategia (colonna)
      usando ritorni giornalieri dell'equity curve
    - salva i risultati in ANALYZED
    """
    fname = os.path.basename(path)
    base_id = os.path.splitext(fname)[0]  # es. "EURUSD_1m_H5_M"
    print(f"\n=== Analyzing M file: {fname} ===")

    # Legge M; assume indice come eventi (timestamp o simile)
    if path.lower().endswith(".parquet"):
        M = pd.read_parquet(path)
    else:
        M = pd.read_csv(path, index_col=0)

    # Garantiamo numerico
    M = M.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if len(M) < 100:
        print(f"  Not enough observations in M ({len(M)} rows). Skipping CSCV.")
        return

    print(f"  M has shape {M.shape[0]} x {M.shape[1]}")

    # 1) CSCV
    df_cscv = cscv_pipeline(M, S=S, max_iter=max_iter)
    metrics_cscv = evaluate_cscv(df_cscv)

    # 2) Sharpe / Sortino / DSR per strategia
    #    TUTTE le strategie condividono gli stessi timestamps
    if isinstance(M.index, pd.DatetimeIndex):
        ts = M.index.to_numpy()
    else:
        ts = pd.to_datetime(M.index).to_numpy()

    # Pre-calcolo equity curve giornaliera e Sharpe per tutte le strategie
    daily_map = {}
    sharpe_map = {}

    for col in M.columns:
        r = M[col].values
        daily = build_daily_equity_returns(r, ts)  # ritorni giornalieri
        daily_map[col] = daily
        sharpe_map[col] = sharpe_from_equity_curve(daily, rf_annual=RF_RATE, periods_per_year=PERIODS_PER_YEAR)

    all_sharpes = np.array(list(sharpe_map.values()))

    perf_rows = []
    for col in M.columns:
        r = M[col].values
        daily = daily_map[col]

        sr = sharpe_map[col]
        sor = sortino_from_equity_curve(daily, rf_annual=RF_RATE, periods_per_year=PERIODS_PER_YEAR)

        # mean/std sui rendimenti per evento (coerente con "mean_ret"/"std_ret" già usati)
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))

        # skew/kurtosis calcolati sui ritorni giornalieri dell'equity curve
        if len(daily) > 3:
            sk_val = float(skew(daily, bias=False))
            ku_val = float(kurtosis(daily, fisher=False, bias=False))  # kurtosi totale
        else:
            sk_val = 0.0
            ku_val = 3.0

        T_daily = len(daily)

        dsr_val = deflated_sharpe_ratio(
            sr_hat=sr,
            sr_trials=all_sharpes,
            T=T_daily,
            skew_val=sk_val,
            kurt_val=ku_val,
            sr0=0.0
        )

        perf_rows.append({
            "strategy": col,
            "dsr": dsr_val,
            "sharpe": sr,
            "sortino": sor,
            "mean_ret": mu,
            "std_ret": sigma,
            "n_obs": int(len(r)),
        })

    df_perf = pd.DataFrame(perf_rows)

    # 3) Salvataggio risultati
    # 3.1 CSCV raw
    out_cscv_path = os.path.join(analyzed, f"{base_id}_cscv_results.csv")
    df_cscv.to_csv(out_cscv_path, index=False)
    print(f"  Saved CSCV raw results to {out_cscv_path}")

    # 3.2 Metriche di overfitting + performance metriche
    out_metrics_path = os.path.join(analyzed, f"{base_id}_overfit_metrics.csv")
    df_metrics = df_perf.copy()

    df_metrics["PBO"] = metrics_cscv["PBO"]
    df_metrics["perf_degradation"] = metrics_cscv["perf_degradation"]
    df_metrics["prob_loss"] = metrics_cscv["prob_loss"]
    df_metrics["ks_stat"] = metrics_cscv["stochastic_dom"]["ks_stat"]
    df_metrics["ks_pval"] = metrics_cscv["stochastic_dom"]["pval"]

    df_metrics.to_csv(out_metrics_path, index=False)
    print(f"  Saved overfitting metrics to {out_metrics_path}")

    # 3.3 Istogramma dei logit
    out_plot_path = os.path.join(analyzed, f"{base_id}_logit_hist.png")
    plot_logit_histogram(df_cscv, out_plot_path)
    print(f"  Saved logit histogram plot to {out_plot_path}")


# --------------------------------------------------------------------
# Summary tables
# --------------------------------------------------------------------

def parse_pair_scheme_horizon_from_name(fname: str):
    """
    Esempio nome: AUD_USD_CPCV_H10_M_overfit_metrics.csv
    Restituisce: pair='AUD_USD', scheme='CPCV', horizon=10
    """
    base = fname.replace("_overfit_metrics.csv", "")
    parts = base.split("_")
    # ..._PAIR_..._SCHEME_H{H}_M
    scheme = parts[-3]                # es. 'CPCV' o 'WF'
    horizon_str = parts[-2]           # es. 'H10'
    horizon = int(horizon_str[1:])    # rimuovi 'H'
    pair = "_".join(parts[:-3])       # tutto quello che rimane prima dello scheme
    return pair, scheme, horizon


def build_summary_tables(analyzed_dir: str):
    """
    Legge tutti i *_overfit_metrics.csv e costruisce:
    - un CSV 'summary_best_strategies.csv' con una riga per (pair, scheme, horizon),
      scegliendo la strategia con DSR massimo (fallback Sharpe se DSR mancante);
    - per ogni scheme (WF / CPCV) crea file Excel con pivot pair x horizon
      per alcune metriche chiave, arrotondate a 3 decimali.
    """
    pattern = os.path.join(analyzed_dir, "*_overfit_metrics.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Nessun file *_overfit_metrics.csv trovato in {analyzed_dir}.")
        return

    summary_rows = []

    for path in files:
        fname = os.path.basename(path)
        pair, scheme, horizon = parse_pair_scheme_horizon_from_name(fname)

        df = pd.read_csv(path)

        # Scegli la strategia "migliore":
        # 1) massimizza DSR se presente e non tutto NaN
        # 2) altrimenti fallback al massimo Sharpe
        if "dsr" in df.columns and df["dsr"].notna().any():
            idx_best = df["dsr"].idxmax()
        else:
            idx_best = df["sharpe"].idxmax()

        row = df.loc[idx_best]

        summary_rows.append({
            "pair": pair,
            "scheme": scheme,
            "horizon": horizon,
            "best_strategy": row["strategy"],
            "dsr": row.get("dsr", np.nan),
            "sharpe": row["sharpe"],
            "sortino": row["sortino"],
            "mean_ret": row["mean_ret"],
            "std_ret": row["std_ret"],
            "PBO": row["PBO"],
            "perf_degradation": row["perf_degradation"],
            "prob_loss": row["prob_loss"],
            "ks_stat": row["ks_stat"],
            "ks_pval": row["ks_pval"],
        })

    summary_df = pd.DataFrame(summary_rows)

    # Arrotondiamo tutte le colonne numeriche a 3 decimali
    num_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[num_cols] = summary_df[num_cols].round(3)

    summary_path = os.path.join(analyzed_dir, "summary_best_strategies.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Salvato riassunto best-strategies in {summary_path}")

    # Costruisci pivot per metrica e scheme
    metrics_to_pivot = ["dsr", "sharpe", "mean_ret", "PBO", "perf_degradation", "prob_loss"]

    for scheme in summary_df["scheme"].unique():
        df_s = summary_df[summary_df["scheme"] == scheme]

        out_xlsx = os.path.join(analyzed_dir, f"summary_tables_{scheme}.xlsx")
        with pd.ExcelWriter(out_xlsx) as writer:
            for metric in metrics_to_pivot:
                if metric not in df_s.columns:
                    continue
                pivot = df_s.pivot(
                    index="pair",
                    columns="horizon",
                    values=metric
                ).sort_index(axis=1)
                # arrotonda anche le pivot a 3 decimali
                pivot = pivot.round(3)
                pivot.to_excel(writer, sheet_name=metric)
        print(f"Salvate tabelle riassuntive per scheme={scheme} in {out_xlsx}")


# ============================================================
#  ANALISI COMPARATIVA DEI MODELLI ML
#  (da aggiungere dopo build_summary_tables)
# ============================================================

def load_all_overfit_metrics(analyzed_dir: str) -> pd.DataFrame:
    """
    Legge tutti i file *_overfit_metrics.csv in analyzed_dir
    e concatena in un unico DataFrame, aggiungendo:
    - pair, scheme, horizon (da nome file)
    - model (uguale a 'strategy' per ora)
    """
    pattern = os.path.join(analyzed_dir, "*_overfit_metrics.csv")
    files = sorted(glob.glob(pattern))

    all_rows = []
    for path in files:
        fname = os.path.basename(path)
        pair, scheme, horizon = parse_pair_scheme_horizon_from_name(fname)

        df = pd.read_csv(path)
        df["pair"] = pair
        df["scheme"] = scheme
        df["horizon"] = horizon

        # qui assumiamo che 'strategy' identifichi il modello ML (logit_cls, svr_reg, ecc.)
        df["model"] = df["strategy"]

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    return all_df


def summarize_models_global(all_df: pd.DataFrame, out_dir: str):
    """
    Tabella A (global summary per modello), salvata in .xlsx con 3 decimali.

    Colonne tenute:
    - model
    - n_strategies
    - avg_sharpe
    - avg_dsr
    - frac_dsr_gt_0_5
    - avg_PBO
    - avg_perf_degradation
    - avg_prob_loss
    - frac_sharpe_gt_1
    """
    if all_df.empty:
        print("summarize_models_global: nessun dato in all_df.")
        return

    g = all_df.groupby("model")

    def frac(cond: pd.Series) -> float:
        return float(cond.mean()) if len(cond) > 0 else np.nan

    summary = pd.DataFrame({
        "n_strategies": g.size(),
        "avg_sharpe": g["sharpe"].mean(),
        "avg_dsr": g["dsr"].mean(),
        "frac_dsr_gt_0_5": g.apply(lambda x: frac(x["dsr"] > 0.5)),
        "avg_PBO": g["PBO"].mean(),
        "avg_perf_degradation": g["perf_degradation"].mean(),
        "avg_prob_loss": g["prob_loss"].mean(),
        "frac_sharpe_gt_1": g.apply(lambda x: frac(x["sharpe"] > 1.0)),
    }).reset_index()

    # arrotonda tutte le colonne numeriche a 3 decimali
    num_cols = summary.select_dtypes(include=[np.number]).columns
    summary[num_cols] = summary[num_cols].round(3)

    out_path = os.path.join(out_dir, "model_summary_global.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        summary.to_excel(writer, sheet_name="global_summary", index=False)

    print(f"Salvata sintesi globale per modello in {out_path}")

def summarize_models_best_cases(all_df: pd.DataFrame, out_dir: str):
    """
    Tabella B:
    Per ogni modello ML seleziona il 'best case' in termini di Sharpe.

    Colonne tenute:
    - model, pair, scheme, horizon, strategy
    - sharpe, dsr, PBO, perf_degradation, prob_loss
    - mean_ret, std_ret, n_obs (se presente)
    """
    if all_df.empty:
        print("summarize_models_best_cases: nessun dato in all_df.")
        return

    df_sorted = all_df.sort_values("sharpe", ascending=False)
    best = df_sorted.groupby("model").head(1).copy()

    cols = [
        "model", "pair", "scheme", "horizon", "strategy",
        "sharpe", "dsr", "PBO", "perf_degradation", "prob_loss",
        "mean_ret", "std_ret", "n_obs"
    ]
    cols_present = [c for c in cols if c in best.columns]
    best = best[cols_present]

    # arrotonda numeriche a 3 decimali
    num_cols = best.select_dtypes(include=[np.number]).columns
    best[num_cols] = best[num_cols].round(3)

    out_path = os.path.join(out_dir, "model_best_cases.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        best.to_excel(writer, sheet_name="best_cases", index=False)

    print(f"Salvati best-case per modello in {out_path}")



def summarize_models_robustness(all_df: pd.DataFrame, out_dir: str,
                                sharpe_thr: float = 1.0,
                                dsr_thr: float = 0.5,
                                pbo_thr: float = 0.1,
                                perfdeg_thr: float = -0.3):
    """
    Tabella C: robustezza per modello.

    Colonne tenute:
    - model
    - n_scenarios
    - frac_sharpe_gt_thr
    - frac_dsr_gt_thr
    - frac_pbo_lt_thr
    - frac_perfdeg_gt_thr
    (i conteggi assoluti li togliamo per snellire la tabella)
    """
    if all_df.empty:
        print("summarize_models_robustness: nessun dato in all_df.")
        return

    g = all_df.groupby("model")

    rows = []
    for model, df_m in g:
        n_total = len(df_m)
        if n_total == 0:
            continue

        frac_sharpe = float((df_m["sharpe"] > sharpe_thr).mean())
        frac_dsr = float((df_m["dsr"] > dsr_thr).mean())
        frac_pbo = float((df_m["PBO"] < pbo_thr).mean())
        frac_perf = float((df_m["perf_degradation"] > perfdeg_thr).mean())

        rows.append({
            "model": model,
            "n_scenarios": n_total,
            "frac_sharpe_gt_thr": frac_sharpe,
            "frac_dsr_gt_thr": frac_dsr,
            "frac_pbo_lt_thr": frac_pbo,
            "frac_perfdeg_gt_thr": frac_perf,
        })

    df_rob = pd.DataFrame(rows)

    # arrotonda numeriche a 3 decimali
    num_cols = df_rob.select_dtypes(include=[np.number]).columns
    df_rob[num_cols] = df_rob[num_cols].round(3)

    out_path = os.path.join(out_dir, "model_robustness.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        df_rob.to_excel(writer, sheet_name="robustness", index=False)

    print(f"Salvati indicatori di robustezza per modello in {out_path}")



def summarize_models_cost_impact(analyzed_no_costs: str,
                                 analyzed_with_costs: str,
                                 out_dir: str):
    """
    Tabella D: impatto dei costi per modello.

    Colonne tenute:
    - model
    - n_common_strategies
    - avg_sharpe_no_costs
    - avg_sharpe_with_costs
    - avg_sharpe_diff
    - avg_dsr_no_costs
    - avg_dsr_with_costs
    - avg_dsr_diff
    - avg_PBO_no_costs
    - avg_PBO_with_costs
    - avg_PBO_diff
    """
    df_no = load_all_overfit_metrics(analyzed_no_costs)
    df_wc = load_all_overfit_metrics(analyzed_with_costs)

    if df_no.empty or df_wc.empty:
        print("summarize_models_cost_impact: una delle due directory non contiene dati.")
        return

    key_cols = ["pair", "scheme", "horizon", "strategy"]

    cols_keep = key_cols + ["model", "sharpe", "dsr", "PBO"]
    df_no = df_no[cols_keep].rename(columns={
        "sharpe": "sharpe_no_costs",
        "dsr": "dsr_no_costs",
        "PBO": "PBO_no_costs"
    })
    df_wc = df_wc[cols_keep].rename(columns={
        "sharpe": "sharpe_with_costs",
        "dsr": "dsr_with_costs",
        "PBO": "PBO_with_costs"
    })

    merged = pd.merge(
        df_no,
        df_wc.drop(columns=["model"]),
        on=key_cols,
        how="inner",
    )

    if merged.empty:
        print("summarize_models_cost_impact: nessuna strategia in comune tra le due directory.")
        return

    merged["sharpe_diff"] = merged["sharpe_with_costs"] - merged["sharpe_no_costs"]
    merged["dsr_diff"] = merged["dsr_with_costs"] - merged["dsr_no_costs"]
    merged["PBO_diff"] = merged["PBO_with_costs"] - merged["PBO_no_costs"]

    g = merged.groupby("model")

    cost_impact = pd.DataFrame({
        "avg_sharpe_no_costs": g["sharpe_no_costs"].mean(),
        "avg_sharpe_with_costs": g["sharpe_with_costs"].mean(),
        "avg_sharpe_diff": g["sharpe_diff"].mean(),
        "avg_dsr_no_costs": g["dsr_no_costs"].mean(),
        "avg_dsr_with_costs": g["dsr_with_costs"].mean(),
        "avg_dsr_diff": g["dsr_diff"].mean(),
        "avg_PBO_no_costs": g["PBO_no_costs"].mean(),
        "avg_PBO_with_costs": g["PBO_with_costs"].mean(),
        "avg_PBO_diff": g["PBO_diff"].mean(),
        "n_common_strategies": g.size(),
    }).reset_index()

    # arrotonda numeriche a 3 decimali
    num_cols = cost_impact.select_dtypes(include=[np.number]).columns
    cost_impact[num_cols] = cost_impact[num_cols].round(3)

    out_path = os.path.join(out_dir, "model_cost_impact.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        cost_impact.to_excel(writer, sheet_name="cost_impact", index=False)

    print(f"Salvato impatto dei costi per modello in {out_path}")



# ------------------------------------------------------------
# Esempio di integrazione nel main esistente
# (puoi adattarlo alla tua pipeline)
# ------------------------------------------------------------

def main():
    M_files = list_M_files(tested)
    if not M_files:
        print(f"No *_M.csv files found in {tested}.")
        return

    print(f"Found {len(M_files)} M files to analyze.")
    for path in M_files:
        process_M_file(path, S=8, max_iter=None)

    # 1) tabelle riassuntive per pair/scheme/horizon
    build_summary_tables(analyzed)

    # 2) analisi comparativa tra modelli ML
    all_metrics_df = load_all_overfit_metrics(analyzed)
    if all_metrics_df.empty:
        print("Nessun *_overfit_metrics.csv trovato, salto analisi modelli.")
        return

    summarize_models_global(all_metrics_df, analyzed)
    summarize_models_best_cases(all_metrics_df, analyzed)
    summarize_models_robustness(all_metrics_df, analyzed)
    
    summarize_models_cost_impact(
         analyzed_no_costs=analyzed,
         analyzed_with_costs=analyzed_costs,
         out_dir=analyzed
     )


if __name__ == "__main__":
    main()
