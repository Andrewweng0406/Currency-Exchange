"""Recheck saved research predictions and interval coverage without fitting models or opening a database."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtesting.metrics import classification_metrics
from app.models.training import _weights_from_scores
from scripts.analyze_history import block_interval, clean, model_summary, monthly_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError('Preserve the original research directory; choose a separate output directory')
    manifest = json.loads((args.input_dir/'manifest.json').read_text())
    inputs = ['research_features.csv','predictions.csv','model_metrics.csv','strategy_summary.csv']
    for name in inputs:
        actual = hashlib.sha256((args.input_dir/name).read_bytes()).hexdigest()
        if actual != manifest[name]:
            raise ValueError(f'Research input fingerprint mismatch: {name}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.input_dir/'research_features.csv', parse_dates=['date'])
    # The original XGBoost pipeline returns float32. CSV's shortest decimal
    # representation round-trips through float32, not through a float64 parse.
    predictions = pd.read_csv(args.input_dir/'predictions.csv', parse_dates=['date','label_end'],
                              dtype={'xgboost':'float32'})
    previous = pd.read_csv(args.input_dir/'model_metrics.csv')
    metrics, _ = model_summary(predictions)
    keys = ['horizon','model']
    columns = ['brier','coin_ci_low','coin_ci_high','delta_ci_low','delta_ci_high']
    before = previous.set_index(keys)[columns].sort_index()
    after = metrics.set_index(keys)[columns].sort_index()
    np.testing.assert_allclose(before, after, atol=1e-12, rtol=0, equal_nan=True)
    monthly, strategies = monthly_comparison(features, predictions)
    prior_strategies = pd.read_csv(args.input_dir/'strategy_summary.csv').set_index('strategy')
    after_strategies = strategies.set_index('strategy')
    np.testing.assert_allclose(prior_strategies, after_strategies, atol=1e-10, rtol=0, equal_nan=True)

    # Independently rebuild the year-by-year ensemble from the saved component
    # probabilities, selecting weights from already-matured earlier folds only.
    names = ['coin_flip','historical_rate','logistic','xgboost','time_series']
    allocations = []
    for horizon, part in predictions.groupby('horizon'):
        for year, test in part.groupby(part.date.dt.year):
            known = part[(part.date.dt.year < year) & (part.label_end < test.date.min())]
            weights = ({name:float(name=='coin_flip') for name in names} if known.empty else
                       _weights_from_scores({name:classification_metrics(known.y, known[name]) for name in names}))
            rebuilt = sum(test[name]*weights[name] for name in names)
            error = float((rebuilt-test.existing_ensemble).abs().max())
            assert error < 1e-12
            allocations.append(dict(horizon=int(horizon),year=int(year),known_rows=len(known),
                                    max_reconstruction_error=error,**weights))

    source = pd.Series(np.where(features.CNY_CLOSE.notna(),'CNY',
                                np.where(features.CNH_CLOSE.notna(),'CNH','missing')))
    switches = source.ne(source.shift()) & source.ne('missing') & source.shift().ne('missing') & source.shift().notna()
    metrics.to_csv(args.output_dir/'recomputed_metrics.csv',index=False)
    monthly.to_csv(args.output_dir/'recomputed_monthly.csv',index=False)
    strategies.to_csv(args.output_dir/'recomputed_strategies.csv',index=False)
    pd.DataFrame(allocations).to_csv(args.output_dir/'reconstructed_weights.csv',index=False)
    example = [10.,-9.,10.,-9.]
    summary = dict(input_dir=str(args.input_dir),verified_input_files=len(inputs),model_rows_checked=len(metrics),
                   historical_model_intervals_unchanged=True,historical_strategy_intervals_unchanged=True,
                   months=len(monthly),ensemble_fold_reconstructions=len(allocations),
                   max_ensemble_reconstruction_error=max(row['max_reconstruction_error'] for row in allocations),
                   china_proxy_source_counts=source.value_counts().to_dict(),china_proxy_source_switches=int(switches.sum()),
                   insufficient_example=dict(values=example,mean=float(np.mean(example)),
                                             requested_block=20,interval=block_interval(example,20)),
                   note='Saved exploratory data; no pristine holdout, vintage certification, model fitting, or promotion.')
    (args.output_dir/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
