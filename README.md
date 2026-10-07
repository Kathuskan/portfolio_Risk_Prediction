# SAIFA — Risk across tails and timescales

A single-file Python pipeline that reads real ETF data, fits risk models, evaluates forecasts, and shows metrics and plots. No notebook, PDF generator, ZIP builder or automatic package installer is involved.

## Run

Open this `Saifa` folder in VS Code. Select the project's `.venv` Python interpreter.

On this Mac, the existing environment is ready:

```bash
.venv/bin/python main.py
```

With the environment activated, the single command is:

```bash
python main.py
```

The command prints progress and evaluation tables in the terminal and opens three Matplotlib plot windows. PNG copies are saved in `outputs/figures/`. No HTML or PDF is generated.

Close the plot windows when finished to return to the terminal. If your environment has no graphical display, use `--no-show` and open the saved PNG files in VS Code.

For automated execution without opening plot windows:

```bash
python main.py --no-show
```

### First setup on another computer

Use Python 3.11 (the pinned packages support Python 3.9–3.12). Create and activate an environment, then install the dependencies once:

```bash
python -m venv .venv
```

Mac/Linux: `source .venv/bin/activate`.
Windows Command Prompt: `.venv\Scripts\activate.bat`.
In PowerShell you can instead use `.venv\Scripts\python.exe` in place of `python`, without activation.

```bash
python -m pip install -r requirements.txt
python main.py
```

The frozen data is included, so normal analysis does not require Internet access. No packages are installed during a normal run. A package-server timeout is a network failure, not evidence of an incompatible package version.

## Files to understand

```text
Saifa/
  main.py              All code: data, Haar, copulas, risk, evaluation and plots
  requirements.txt     Numerical dependencies only
  README.md            This guide
  data/
    raw_prices.csv     Frozen adjusted prices
    data_manifest.json Source, dates, adjustment settings and checksum
  outputs/
    figures/           Three PNG plots
    tables/            Exact calculations and evidence as CSV
    selected_models.json
    run_summary.json   Input hash, periods, package versions and passed checks
```

All Python code is in `main.py`, arranged in numbered sections: study choices, optional download, Haar returns, risk measures, copulas, evaluation, plots and the pipeline. Small functions keep repeated mathematics in one place. Read `main()` to follow the execution order, then the named functions above it. There are no classes or framework layers.

## What happens in one run?

1. **Check the data.** Verify its SHA-256 hash, columns, dates, missing values, positive prices and return arithmetic. Split the observations chronologically.
2. **Fit and evaluate.** Fit on 2016–2021; compare forecasts on 2022–2023; select the lowest validation quantile loss at each horizon. Refit on 2016–2023 and evaluate 2024 onward. Never select a model using final-test results.
3. **Compare dependence assumptions.** Compare horizon-specific Student-t risk with Gaussian and with transferred daily dependence, holding horizon margins fixed. Compare 20,000 with 40,000 simulated scenarios to identify unstable small differences.
4. **Print evaluation metrics.** Display all models, sample counts, breaches, expected breaches, rates, quantile loss, VaR and ES. Mark the choices made on validation.
5. **Show plots.** Save the figures and display them directly in Matplotlib windows. All empirical numbers and figures are recalculated by the same command.

## How to read the metrics

These are **risk forecast evaluation measures**, not classification accuracy. Do not describe `100% - breach rate` as model accuracy.

| Metric | Meaning | Interpretation |
|---|---|---|
| N | Number of non-overlapping evaluated holding periods | Long horizons have fewer observations |
| Breaches | Actual loss exceeded the predicted 95% VaR | Count alongside N |
| Expected | `0.05 * N` | Nominal expected exceptions |
| Breach % | `100 * breaches / N` | Target is 5%, not zero |
| Calibration gap | Breach percentage minus 5 | Percentage-point deviation; descriptive |
| QL_x100 | Mean 95% quantile/pinball loss multiplied by 100 | Lower is better within the same horizon |
| VaR95 | Predicted 95th loss percentile | Positive values represent losses |
| ES95 | Mean loss in the worst 5% | Tail severity estimate; not formally backtested here |
| Selected | Chosen using validation quantile loss | Not necessarily the final-test winner |

Pinball loss is `(0.95 - I[loss < VaR]) * (loss - VaR)`. It penalises forecasts that are too small and unnecessarily conservative forecasts. A model cannot win just by producing fewer breaches.

The terminal uses percentage VaR/ES and percentage breach rates. Most raw CSV columns use decimal returns/losses. Columns explicitly marked `pp` or `percent` are already scaled. Compare scores within a horizon; different horizons have different natural loss scales.

## Why these choices?

- **Assets and weights:** SPY, QQQ, GLD and TLT, initially 25% each, cover equity, gold and long-term Treasuries without optimisation. SPY and QQQ overlap. Equal weights are a transparent design choice, not a claim of optimality.
- **Real data:** Yahoo Finance adjusted closing prices from 4 January 2016 to 5 October 2026, 2,704 rows. The manifest records acquisition and provenance. Provider histories can be revised. We use the newer frozen local snapshot and never silently refresh it.
- **Log returns:** `log(P[t]/P[t-1])` adds through time for each asset. Portfolio simple loss is `-sum(weight * expm1(cumulative_asset_log_return))`. Initial weights are held without rebalancing inside each horizon. Costs are excluded.
- **Haar and horizons:** At 1, 4 and 16 trading days, the causal Haar scaling coefficient times `sqrt(h)` equals the cumulative log return exactly. The code checks this against direct sums and PyWavelets. This is an interpretable horizon representation, not a claim of extra predictive power over equivalent aggregation. It is not a full wavelet-band model.
- **HS benchmark:** Historical Simulation directly uses historical portfolio losses.
- **Gaussian versus Student-t:** Same empirical margins and Kendall-based correlation. Gaussian has zero asymptotic tail dependence for nondegenerate correlation; t can have joint extremes but imposes symmetric upper and lower tails. Neither is assumed to win.
- **t degrees of freedom:** Choose 4, 8, 16 or 32 by training copula pseudo-log-likelihood while holding correlation fixed. All candidate scores and boundary flags are saved. This is a coarse grid estimate, not unrestricted maximum likelihood.
- **Daily-t diagnostic:** Transfer daily correlation and degrees of freedom to longer-horizon margins. It is not an additional selection candidate. Differences include changes in both correlation and tail shape.
- **95% risk:** More expected tail observations than a 99% analysis, but long-horizon evidence is still sparse.
- **Non-overlapping evaluation:** Avoid mechanically repeating returns across neighbouring outcomes. Non-overlap does not guarantee independence. Training uses overlapping windows; an approximate non-overlapping sample equivalent is saved.
- **Simulation:** Fixed seeds 42/43 and common inputs aid reproducibility. A 40,000-scenario comparison tests sensitivity; fixed seeds alone do not establish accuracy. A model gap is marked sensitive if doubling scenarios changes it by at least 25% of its original magnitude.

Dates, assets, weights and horizons are fixed design decisions. Fit diagnostics estimate parameters. Validation scores support selection. Final-test scores assess the frozen choice. Do not change the design to make final-test results look better.

## Where is the evidence?

| Output in `outputs/tables/` | Question it answers |
|---|---|
| `data_quality.csv`, `period_coverage.csv` | What data and dates were actually used? |
| `parameter_fits.csv` | Why was this t degrees-of-freedom value chosen? |
| `estimation_samples.csv` | How many rows were fitted, did nu reach a boundary, was correlation repaired? |
| `dependence.csv` | What happened to each of the six asset pairs at each horizon? |
| `metrics.csv` | Which model was selected, and how did every model perform? |
| `forecasts.csv` | What was known at fitting cutoff, and which later dates formed each outcome? |
| `risk_comparison.csv`, `risk_impact.csv` | How much do the dependence assumptions change risk estimates? |
| `monte_carlo_check.csv` | How sensitive are the estimates to simulation count? |

The dependence figure uses the pre-test refit sample. The empirical 10% co-exceedance measure differs from asymptotic fitted tail dependence. Changes are descriptive; there is no formal significance test. Model-to-model differences do not establish true-risk underestimation. A t copula is not automatically more accurate.

## What you can and cannot conclude

A selected model may lose to HS on the final test; report that honestly. Few exceptions at 16 days limit conclusions. Empirical margins cannot extrapolate beyond observed marginal extremes. Static fits do not update volatility or adapt to regime changes. A VaR breach test does not validate ES. No causality, general statistical superiority, optimised portfolio or profitable strategy is claimed.

For a live explanation, work through one four-day return by hand, then explain ranks, dependence fitting, simulated losses and the validation score. The numbered sections in `main.py` expose each formula and show how calculated results become plots.

## Optional fresh download

Normal runs reuse `data/raw_prices.csv`. Do not overwrite it if you need the original results.
To acquire a new snapshot in a separate project copy:

```bash
python -m pip install yfinance==1.2.0
python main.py --download
```

The downloader refuses to overwrite an existing frozen CSV. If no data file exists, `main.py` calls it; yfinance must then be installed and Yahoo must be reachable. Source pages: [SPY](https://finance.yahoo.com/quote/SPY/history/), [QQQ](https://finance.yahoo.com/quote/QQQ/history/), [GLD](https://finance.yahoo.com/quote/GLD/history/), [TLT](https://finance.yahoo.com/quote/TLT/history/).

## AI disclosure

Generative AI materially assisted design, implementation, debugging and documentation. The team remains responsible for reviewing and explaining the calculations and claims. No completed human review is asserted.
