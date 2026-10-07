"""Data -> Haar wavelets -> copulas -> risk -> evaluation -> plots.
Run: python main.py (or add --no-show to save plots without opening windows).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from scipy import stats
import pywt

# 1. Fixed study choices
TICKERS = ['SPY', 'QQQ', 'GLD', 'TLT']
HORIZONS = [1, 4, 16]
MODELS = ['HS', 'Gaussian', 'Student-t']
WEIGHTS = np.full(4, 0.25)
NU_GRID = [4, 8, 16, 32]


# 2. Optional public-data download (normal runs use the included CSV)
def download():
    import yfinance as yf
    target = ROOT / 'data/raw_prices.csv'
    if target.exists():
        raise FileExistsError('Frozen data already exists. Preserve it; download into a separate project to refresh.')
    yf.set_tz_cache_location(str(ROOT / 'data/.yfinance_cache'))
    prices = []
    for ticker in TICKERS:
        frame = yf.download(ticker, start='2016-01-01', end='2026-10-06',
                            auto_adjust=False, actions=False, interval='1d',
                            keepna=True, progress=False, threads=False, timeout=30)
        if frame is None or frame.empty or 'Adj Close' not in frame:
            raise ValueError(f'{ticker}: no adjusted prices. Supply real Yahoo adjusted-price CSVs; no synthetic fallback.')
        series = frame['Adj Close']
        if isinstance(series, pd.DataFrame):
            series = series[ticker]
        prices.append(series.rename(ticker))
    frame = pd.concat(prices, axis=1).sort_index()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = 'Date'
    target.parent.mkdir(exist_ok=True)
    frame.to_csv(target, float_format='%.12g')
    manifest = dict(source='Yahoo Finance via yfinance',
                    source_urls=[f'https://finance.yahoo.com/quote/{t}/history/' for t in TICKERS],
                    tickers=TICKERS, requested_start='2016-01-01', exclusive_end='2026-10-06',
                    actual_first=str(frame.index.min().date()), actual_last=str(frame.index.max().date()),
                    observations=len(frame), missing=frame.isna().sum().to_dict(),
                    adjustment='Yahoo Adj Close; auto_adjust=False; actions=False; interval=1d',
                    downloaded_utc=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (target.parent / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(frame.shape, frame.index.min(), frame.index.max(), frame.isna().sum().to_dict())


# 3. Haar holding-period returns
def haar_returns(daily, horizon):
    """Causal undecimated Haar approximation, rescaled to holding-period log returns."""
    approximation = daily.copy()
    level = int(np.log2(horizon))
    for j in range(level):
        approximation = (approximation + approximation.shift(2**j)) / np.sqrt(2)
    result = (np.sqrt(horizon) * approximation).dropna()
    direct = daily.rolling(horizon).sum().dropna()
    np.testing.assert_allclose(result, direct, atol=1e-12)
    # Independent library check on a complete block, without any boundary padding.
    if level:
        block = daily.iloc[:horizon].to_numpy()
        expected = pywt.wavedec(block, 'haar', level=level, axis=0)[0][0]
        np.testing.assert_allclose(approximation.iloc[horizon-1], expected, atol=1e-12)
    return result


# 4. Portfolio loss and risk measures
def portfolio_loss(log_returns):
    return -np.expm1(np.asarray(log_returns)) @ WEIGHTS


def risk(losses):
    """Linear empirical VaR; ES averages exactly 5% mass in the worst tail."""
    ordered = np.sort(np.asarray(losses))
    var = float(np.quantile(ordered, 0.95, method='linear'))
    mass = len(ordered) * 0.05
    whole = int(np.floor(mass))
    fraction = mass - whole
    total = ordered[-whole:].sum() if whole else 0.0
    if fraction > 1e-12:
        total += fraction * ordered[-whole-1]
    es = float(total / mass)
    assert np.isfinite([var, es]).all() and es >= var - 1e-12
    return var, es


# 5. Copula estimation and simulation
def fit_copula(sample):
    """Kendall correlation and grid pseudo-likelihood; training observations only."""
    u = stats.rankdata(np.asarray(sample), axis=0, method='average') / (len(sample) + 1)
    assert np.all((u > 0) & (u < 1))
    tau = np.eye(4)
    for i in range(4):
        for j in range(i):
            tau[i, j] = tau[j, i] = stats.kendalltau(u[:, i], u[:, j]).statistic
    corr = np.sin(np.pi * tau / 2)
    original = corr.copy()
    eigenvalues, vectors = np.linalg.eigh(corr)
    if eigenvalues.min() < 1e-8:
        corr = (vectors * np.maximum(eigenvalues, 1e-8)) @ vectors.T
        diagonal = np.sqrt(np.diag(corr))
        corr = corr / np.outer(diagonal, diagonal)
    repair = float(np.abs(corr - original).max())
    assert np.linalg.eigvalsh(corr).min() > 0
    np.testing.assert_allclose(np.diag(corr), 1)
    if repair > 0.01:
        raise ValueError('Material correlation repair: investigate instead of proceeding.')
    scores = []
    for nu in NU_GRID:
        z = stats.t.ppf(u, df=nu)
        score = np.sum(stats.multivariate_t.logpdf(z, shape=corr, df=nu)
                       - stats.t.logpdf(z, df=nu).sum(axis=1))
        scores.append(float(score))
    nu = NU_GRID[int(np.argmax(scores))]
    pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            rho = corr[i, j]
            tail = 2 * stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), df=nu+1)
            finite = np.mean((u[:, i] <= 0.1) & (u[:, j] <= 0.1)) / 0.1
            pairs.append(dict(pair=f'{TICKERS[i]} / {TICKERS[j]}', tau=tau[i, j],
                              rho=rho, nu=nu, lambda_lower=float(tail), finite_q10=float(finite)))
    return dict(corr=corr, nu=nu, scores=scores, repair=repair, pairs=pairs)


def simulate(sample, fitted, model, n=20000):
    """Same seeds give paired normal inputs and stable prefixes for 20k/40k checks."""
    normal = np.random.default_rng(42).standard_normal((n, 4))
    z = normal @ np.linalg.cholesky(fitted['corr']).T
    if model == 'Gaussian':
        uniform = stats.norm.cdf(z)
    else:
        nu = fitted['nu']
        # ONE scale per row, shared by the four assets: essential for a t copula.
        scale = np.sqrt(np.random.default_rng(43).chisquare(nu, size=n) / nu)
        latent = z / scale[:, None]
        np.testing.assert_allclose(latent * scale[:, None], z, atol=1e-12)
        uniform = stats.t.cdf(latent, df=nu)
    assert np.isfinite(uniform).all() and np.all((uniform >= 0) & (uniform <= 1))
    sample = np.asarray(sample)
    scenarios = np.column_stack([np.quantile(sample[:, i], uniform[:, i], method='linear')
                                 for i in range(4)])
    losses = portfolio_loss(scenarios)
    assert np.isfinite(losses).all()
    return risk(losses)


# 6. Out-of-sample evaluation
def evaluate(daily, horizon, thresholds, cutoff, phase):
    """Non-overlapping outcomes, reset at the evaluation-period boundary."""
    n = len(daily) // horizon
    blocks = daily.iloc[:n*horizon].to_numpy().reshape(n, horizon, 4).sum(axis=1)
    losses = portfolio_loss(blocks)
    summaries, observations = [], []
    assert pd.Timestamp(cutoff) < daily.index.min()
    for model, (var, es) in thresholds.items():
        score = (0.95 - (losses < var)) * (losses - var)
        summaries.append(dict(phase=phase, horizon=horizon, model=model, n=n,
                              breaches=int((losses > var).sum()), expected_breaches=0.05*n,
                              breach_rate=float(np.mean(losses > var)),
                              quantile_loss=float(score.mean()), VaR95=var, ES95=es))
        for k, loss in enumerate(losses):
            observations.append(dict(phase=phase, horizon=horizon, model=model,
                                     fit_cutoff=cutoff,
                                     holding_start=str(daily.index[k*horizon].date()),
                                     holding_end=str(daily.index[(k+1)*horizon-1].date()),
                                     VaR95=var, ES95=es, loss=float(loss)))
    return summaries, observations

# 7. Plots
def make_plots(risk_table, dependence, forecasts, folder, show=True):
    import matplotlib.pyplot as plt
    # Exactly three figures, generated from the same analysis.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    final_dep = dependence
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for pair, frame in final_dep.groupby('pair', sort=False):
        axes[0].plot(frame.horizon, frame.lambda_lower, marker='o', label=pair)
        axes[1].plot(frame.horizon, frame.finite_q10, marker='o', label=pair)
    for ax in axes:
        ax.set_xticks(HORIZONS)
        ax.set_xlabel('Holding period (trading days)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
    axes[0].set_title('Fitted t-copula asymptotic tail')
    axes[1].set_title('Empirical co-exceedance at q = 0.10')
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(folder / 'dependence.png', dpi=300)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for model in MODELS + ['Daily-t']:
        frame = risk_table.query('phase == "test" and model == @model')
        for ax, measure in zip(axes, ['VaR95', 'ES95']):
            ax.plot(frame.horizon, 100*frame[measure], marker='o', label=model)
            ax.set_xticks(HORIZONS)
            ax.set_xlabel('Holding period (trading days)')
            ax.set_ylabel('Portfolio loss (%)')
            ax.set_title(measure)
            ax.grid(alpha=0.15)
    axes[1].legend(fontsize=8)
    fig.savefig(folder / 'risk.png', dpi=300)
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    for i, model in enumerate(MODELS):
        frame = forecasts.query('phase == "test" and horizon == 4 and model == @model')
        dates = pd.to_datetime(frame.holding_end)
        if i == 0:
            ax.plot(dates, frame.loss*100, color='#72808d', linewidth=0.8, label='Realised loss')
        ax.plot(dates, frame.VaR95*100, label=f'{model} VaR95', linestyle=['--', ':', '-.'][i])
    ax.axhline(0, linewidth=0.5, color='black')
    ax.set_ylabel('Four-day portfolio loss (%)')
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(folder / 'test_losses.png', dpi=300)
    if show:
        plt.show()
    else:
        plt.close('all')


# 8. Run the complete pipeline
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true', help='Save PNG plots without opening plot windows.')
    parser.add_argument('--download', action='store_true', help='Download data only; refuses to overwrite frozen data.')
    args = parser.parse_args()
    if args.download:
        download()
        return
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    OUT = ROOT / 'outputs'
    TABLES = OUT / 'tables'
    FIGURES = OUT / 'figures'
    for directory in [TABLES, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    print('STEP 1/5: Check frozen data and chronological periods', flush=True)
    # 1. Load the immutable input and inspect units/coverage before estimating anything.
    raw_path = ROOT / 'data/raw_prices.csv'
    if not raw_path.exists():
        download()
    manifest = json.loads((ROOT / 'data/data_manifest.json').read_text())
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == manifest['sha256'], 'Frozen data checksum changed.'
    prices = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    assert list(prices.columns) == TICKERS
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    assert prices.index.min() >= pd.Timestamp('2016-01-01')
    assert prices.index.max() < pd.Timestamp('2026-10-06')
    assert np.isclose(WEIGHTS.sum(), 1)
    quality = pd.DataFrame({'ticker': TICKERS, 'price_rows': len(prices),
                            'missing_prices': prices.isna().sum().values,
                            'first_date': str(prices.index.min().date()),
                            'last_date': str(prices.index.max().date())})
    quality.to_csv(TABLES / 'data_quality.csv', index=False)
    # Stop rather than bridge missing prices and mislabel a multi-session move as daily.
    assert not prices.isna().any().any(), 'Missing prices require explicit investigation.'
    assert np.isfinite(prices).all().all() and (prices > 0).all().all()
    assert prices.index.to_series().diff().dt.days.max() <= 4, 'Unexpected calendar gap: investigate.'
    returns = np.log(prices / prices.shift(1)).dropna()
    assert np.isfinite(returns).all().all()
    train = returns.loc['2016':'2021']
    validation = returns.loc['2022':'2023']
    refit = returns.loc['2016':'2023']
    test = returns.loc['2024':'2026-10-05']
    assert len(train) > 1000 and len(validation) > 400 and len(test) > 100
    assert train.index.max() < validation.index.min() and refit.index.max() < test.index.min()
    coverage = []
    for name, frame in [('training', train), ('validation', validation), ('refit', refit), ('test', test)]:
        coverage.append(dict(period=name, n=len(frame), first=str(frame.index.min().date()),
                             last=str(frame.index.max().date())))
    pd.DataFrame(coverage).to_csv(TABLES / 'period_coverage.csv', index=False)
    np.testing.assert_allclose(portfolio_loss(np.log1p([[0.1, -0.1, 0.02, 0.0]])), [-0.005])
    print(pd.DataFrame(coverage).to_string(index=False))


    print('\nSTEP 2/5: Fit on training; select on validation; refit and test', flush=True)
    # 2. Validation rule is fixed here before any scores are calculated.
    # Minimum mean 95% pinball loss per horizon; numerical ties favour simpler models.
    selected = {}
    fit_rows, pair_rows, sample_rows, risk_rows, backtest_rows, forecast_rows, mc_rows = [], [], [], [], [], [], []
    for phase, fitting, evaluation in [('validation', train, validation), ('test', refit, test)]:
        daily_fit = None
        for h in HORIZONS:
            sample = haar_returns(fitting, h)
            fitted = fit_copula(sample)
            if h == 1:
                daily_fit = fitted
            sample_rows.append(dict(phase=phase, horizon=h, overlapping_rows=len(sample),
                                    nonoverlapping_equivalent=len(fitting)//h,
                                    nu=fitted['nu'], grid_boundary=fitted['nu'] in [4, 32],
                                    correlation_repair=fitted['repair']))
            for nu, score in zip(NU_GRID, fitted['scores']):
                fit_rows.append(dict(phase=phase, horizon=h, nu=nu, pseudo_loglik=score,
                                     chosen=nu == fitted['nu']))
            for pair in fitted['pairs']:
                pair_rows.append(dict(phase=phase, horizon=h, **pair))
            thresholds = {'HS': risk(portfolio_loss(sample))}
            for model in ['Gaussian', 'Student-t', 'Daily-t']:
                params = daily_fit if model == 'Daily-t' else fitted
                thresholds[model] = simulate(sample, params, model)
                if phase == 'test':
                    doubled = simulate(sample, params, model, n=40000)
                    mc_rows.append(dict(horizon=h, model=model,
                                        VaR20k=thresholds[model][0], ES20k=thresholds[model][1],
                                        VaR40k=doubled[0], ES40k=doubled[1],
                                        delta_VaR_pp=100*(doubled[0]-thresholds[model][0]),
                                        delta_ES_pp=100*(doubled[1]-thresholds[model][1])))
            for model, values in thresholds.items():
                risk_rows.append(dict(phase=phase, horizon=h, model=model, VaR95=values[0], ES95=values[1]))
            summaries, observations = evaluate(evaluation, h,
                                               {m: thresholds[m] for m in MODELS},
                                               str(fitting.index.max().date()), phase)
            backtest_rows.extend(summaries)
            forecast_rows.extend(observations)
            if phase == 'validation':
                best = summaries[0]
                for candidate in summaries[1:]:
                    if candidate['quantile_loss'] < best['quantile_loss'] - 1e-12:
                        best = candidate
                selected[h] = best['model']
            print(phase, h, 'nu=', fitted['nu'], 'validation choice=', selected[h])
        # Save choices before final-test fitting starts; they are never updated on test results.
        if phase == 'validation':
            (OUT / 'selected_models.json').write_text(json.dumps(selected, indent=2))

    risk_table = pd.DataFrame(risk_rows)
    backtest = pd.DataFrame(backtest_rows)
    backtest['calibration_gap_pp'] = 100 * (backtest['breach_rate'] - 0.05)
    backtest['selected_on_validation'] = [m == selected[h] for h, m in zip(backtest.horizon, backtest.model)]
    dependence = pd.DataFrame(pair_rows)
    forecasts = pd.DataFrame(forecast_rows)
    mc = pd.DataFrame(mc_rows)
    for name, table in [('parameter_fits', pd.DataFrame(fit_rows)), ('dependence', dependence),
                        ('estimation_samples', pd.DataFrame(sample_rows)), ('risk_comparison', risk_table),
                        ('metrics', backtest), ('forecasts', forecasts), ('monte_carlo_check', mc)]:
        table.to_csv(TABLES / f'{name}.csv', index=False, float_format='%.12g')


    print('\nSTEP 3/5: Compare tail and horizon assumptions; check simulation sensitivity', flush=True)
    # 3. Compare model assumptions without interpreting any model as the true risk.
    impact = []
    for h in HORIZONS:
        risks = risk_table.query('phase == "test" and horizon == @h').set_index('model')
        checks = mc.query('horizon == @h').set_index('model')
        for baseline in ['Gaussian', 'Daily-t']:
            for measure in ['VaR95', 'ES95']:
                difference = risks.loc['Student-t', measure] - risks.loc[baseline, measure]
                base = risks.loc[baseline, measure]
                mc_name = 'delta_VaR_pp' if measure == 'VaR95' else 'delta_ES_pp'
                shift = (checks.loc['Student-t', mc_name] - checks.loc[baseline, mc_name]) / 100
                # Predeclared diagnostic, not a confidence interval: a >=25% change in the gap is sensitive.
                sensitive = abs(shift) >= 0.25 * max(abs(difference), 1e-12)
                impact.append(dict(horizon=h, baseline=baseline, measure=measure,
                                   difference_pp=100*difference,
                                   relative_percent=100*difference/base if abs(base) > 1e-8 else None,
                                   gap_shift_pp=100*shift, mc_sensitive=bool(sensitive)))
    impact = pd.DataFrame(impact)
    impact.to_csv(TABLES / 'risk_impact.csv', index=False, float_format='%.12g')


    print('\nSTEP 4/5: Show evaluation measures', flush=True)
    columns = ['horizon', 'model', 'selected_on_validation', 'n', 'breaches',
               'expected_breaches', 'breach_rate', 'calibration_gap_pp', 'quantile_loss', 'VaR95', 'ES95']
    for phase in ['validation', 'test']:
        display = backtest.loc[backtest.phase == phase, columns].copy()
        display['breach_rate'] *= 100
        display['VaR95'] *= 100
        display['ES95'] *= 100
        display['quantile_loss'] *= 100
        display.rename(columns={'horizon':'days', 'selected_on_validation':'selected',
                                'expected_breaches':'expected', 'breach_rate':'breach_%',
                                'quantile_loss':'QL_x100', 'VaR95':'VaR_%', 'ES95':'ES_%'}, inplace=True)
        print('\n' + phase.upper())
        print(display.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    print('\nRead this: lower quantile loss is better; breach target is 5%. Compare models within each horizon.')
    print('A selected model may lose to HS in the final test. Selection is not changed by the test results.')
    final_dep = dependence.query('phase == "test"').drop(columns='phase')
    print('\nMean pre-test dependence estimates (descriptive, not a significance test):')
    print(final_dep.groupby('horizon')[['lambda_lower', 'finite_q10']].mean().round(4).to_string())
    print('\nES effect of dependence assumptions (Student-t minus baseline):')
    print(impact.query('measure == "ES95"').to_string(index=False, float_format=lambda value:f'{value:.4f}'))
    boundary = pd.DataFrame(sample_rows).query('grid_boundary')
    if not boundary.empty:
        print('\nCaution: nu reached a grid boundary. See parameter_fits.csv for all scores.')
    if impact.mc_sensitive.any():
        print('Caution: some model gaps are Monte Carlo-sensitive. Do not over-interpret these small differences.')
    print('\nSTEP 5/5: Save and display the three plots', flush=True)
    versions = {name:importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'matplotlib', 'PyWavelets']}
    summary = dict(data_sha256=manifest['sha256'], periods=coverage, selected_on_validation=selected,
                   checks='Data, chronology, Haar identity, correlation, compounding and ES assertions passed.',
                   python=platform.python_version(), packages=versions)
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nDone. Tables:', TABLES)
    print('Plots:', FIGURES)
    if not args.no_show:
        print('Close the plot windows when finished to return to the terminal.')
    make_plots(risk_table, final_dep, forecasts, FIGURES, show=not args.no_show)


if __name__ == '__main__':
    main()
