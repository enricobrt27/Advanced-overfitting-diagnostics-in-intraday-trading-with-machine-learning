Intraday FX Event-Driven ML Framework

This project implements a fully event-driven machine learning framework for intraday FX trading strategies.

It combines:

-Volatility-adjusted CUSUM event detection
-Triple-Barrier Method (TBM) labeling
-Primary signal generation (momentum & mean-reversion)
-Walk-forward & combinatorial purged cross-validation
-Regression and classification models
-Realistic transaction cost modeling (spread + fees + dynamic slippage)
-Stability diagnostics and overfitting checks

The objective is not to maximize in-sample Sharpe, but to rigorously test whether a statistically robust predictive edge exists after realistic costs.


Data & Universe

Instruments tested:

EUR/USD
AUD/USD
NZD/USD
GBP/USD
EUR/GBP
USD/CAD
USD/JPY

-Intraday frequency (1-minute bars)
-Log-returns used throughout the pipeline
-Spread is already embedded in price series.
-Additional costs (fees + slippage) are applied explicitly.



10. Key Findings
-Strong Ranking Signal
-Decile 10 consistently outperforms decile 1
-Spearman ~ 0.98–1.00 across folds
-Weak Post-Cost Edge
-Many folds positive gross
-After realistic costs, net edge often marginal
-Sharpe < 0.6 in most configurations
-No Clear Persistent Arbitrage

Observed performance may be Micro-edge or sampling noise within realistic bounds

11. Current Status

Regression model shows strongest structure.

Classification and meta-labeling:

Work technically

But do not materially outperform regression ranking

The framework is now suitable for:

Robustness research

Feature experimentation

Regime decomposition

Further regularization experiments

Multi-asset extension

12. Next Possible Improvements (Without Overfitting)

Feature orthogonalization

Dimensionality reduction (PCA on features)

Regime-conditioned models (ex-ante defined)

Probabilistic calibration tests

Cross-asset pooled training

Turnover-aware threshold optimization

13. Philosophy of the Project

This project prioritizes:

Statistical integrity

Realistic cost modeling

Strict out-of-sample validation

Overfitting diagnostics

Transparency over performance inflation

It is designed as a research framework rather than a production trading system.

14. Conclusion

The system demonstrates:

Consistent predictive ranking power

Strong structural monotonicity

Controlled experimental setup

But does not yet demonstrate statistically undeniable post-cost edge across regimes.

It represents a rigorous attempt to evaluate whether intraday FX micro-structure inefficiencies can survive realistic trading frictions.







## Data Sources

The analysis focuses on FX data obtained from two major providers:

- **Dukascopy**: provides free historical tick-level bid/ask data. While extremely valuable for research, its historical minute-bar feed does not guarantee continuity and often exhibits short micro-gaps, irregular tick density, and invalid price blocks.
- **Interactive Brokers (IBKR)**: provides a more stable and execution-oriented FX feed. Minute bars are highly regular during trading hours, with gaps concentrated around deterministic rollover windows.

Both historical datasets and live-recorded data were analyzed to validate the observed behaviors. 

---

## Requirements

- Python **3.10** or later

### Main dependencies
- `pandas`
- `numpy`
- `matplotlib`

### Optional dependency
Required only if cleaned datasets are stored in **Parquet** format instead of CSV:
- `pyarrow`

---

## Usage

1. Place raw datasets in the appropriate `data/**/raw/` folders.

2. Run the cleaning pipeline:

    python data_cleaning.py

3. Run the join and feature pipeline:

    python data features.py

3. Run the modeling pipeline:

    python modeling.py

3. Run the final analysis pipeline to get the results:

    python final analysis.py


---



## Repository Structure

The repository expects the following folder structure:

```text
your-repo/
├─ data/
│  ├─ dukascopy/
│    ├─ raw/
│    ├─ cleaned/
│    ├─ joined/
│    ├─ labeled/
│    ├─ tested/
│    └─ analyzed/
│    ├─ tested_costs/
│    └─ analyzed_costs/
│
├─ data_cleaning.py
├─ data features.py
├─ modeling.py
├─ final analysis.py
├─ gaps.py
└─ README.md



All paths are built relative to the location of "gaps.py" using:

```python
BASE_DIR = Path("C:/Users/enric/gaps.py").resolve().parent
DATA_DIR = BASE_DIR / "data"