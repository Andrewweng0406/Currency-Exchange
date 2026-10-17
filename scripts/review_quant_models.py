"""Re-evaluate official-only features and purged calibration without promoting a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtesting.strategy_compare import _strategy_records, _comparisons_from_records
from app.backtesting.dataset import load_feature_frame
from app.config import ROOT, settings
from app.database.session import make_engine
from scripts.analyze_history import (evaluate_models, model_summary, market_summary,
                                    monthly_comparison, calibration_bins, clean, markdown_table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', default='2016-08-17')
    parser.add_argument('--as-of', default='2026-10-16')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/analysis/quant_review_2026-10-16/research')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings()['database']['url'])
    with engine.connect() as connection:
        if engine.dialect.name == 'sqlite':
            connection.exec_driver_sql('PRAGMA query_only=ON')
        with Session(bind=connection) as session:
            f = load_feature_frame(session, 'daily_v3')
            f = f[(f.date >= pd.Timestamp(args.start_date, tz='UTC')) &
                  (f.date < pd.Timestamp(args.as_of, tz='UTC') + pd.Timedelta(days=1))].reset_index(drop=True)
    assert f.source.eq('cbc_taiwan').all(), 'Official-only study cannot mix quote conventions'
    predictions, folds = evaluate_models(f, calibrate=True)
    metrics, years = model_summary(predictions)
    annual, tails = market_summary(f)
    monthly, strategies = monthly_comparison(f, predictions)
    calibration = calibration_bins(predictions)
    # Add the research calibration candidate to reliability diagnostics.
    calibrated_bins = []
    for h, part in predictions.groupby('horizon'):
        column = 'chronological_calibrated_logistic'
        bins = pd.cut(part[column], np.linspace(0, 1, 11), include_lowest=True)
        for band, group in part.groupby(bins, observed=True):
            calibrated_bins.append(dict(horizon=h, model=column, probability_band=str(band), n=len(group),
                                        mean_probability=group[column].mean(), realized_up_fraction=group.y.mean()))
    calibration = pd.concat([calibration, pd.DataFrame(calibrated_bins)], ignore_index=True)
    # Independently check that product and research use the same month budgets.
    p5 = predictions[predictions.horizon == 5].copy()
    price = f[['date','USDTWD_CLOSE']].merge(p5[['date','existing_ensemble']], on='date', how='inner')
    score = np.zeros(len(f))
    from app.config import risk_policy
    for name, weight in risk_policy()['opportunity_score']['weights'].items():
        window = int(name.split('_')[1].removesuffix('d'))
        score += f.USDTWD_CLOSE.rolling(window,min_periods=1).apply(lambda x:1-(x<=x[-1]).mean(),raw=True).to_numpy()*weight*100
    price = price.merge(pd.DataFrame({'date':f.date,'opportunity_score':np.rint(score)}),on='date')
    price = price[price.date.dt.year >= 2023].rename(columns={'existing_ensemble':'prob_up'})
    records = _strategy_records(price,10000)
    product = _comparisons_from_records(records,10000)
    by_name = {item.strategy:item for item in product}
    assert len(monthly) == by_name['fixed_day_once'].payments
    for name, key in [('fixed','fixed_day_once'),('tranche','equal_tranches'),('model','model_timing_once')]:
        expected = strategies[strategies.strategy==name].iloc[0]
        assert abs(expected.average_rate-by_name[key].average_rate)<1e-10
    sensitivity=[]
    for fee in (0,50,100,200):
        for row in strategies.itertuples():
            extra_orders = 3 if row.strategy=='tranche' else 0
            sensitivity.append(dict(strategy=row.strategy, fee_twd_per_order=fee,
                                    mean_savings_after_extra_fees=row.mean_savings_twd-extra_orders*fee))
    tables=dict(predictions=predictions,folds=folds,model_metrics=metrics,yearly_metrics=years,
                annual_returns=annual,tail_distribution=tails,monthly_strategy=monthly,
                strategy_summary=strategies,calibration_bins=calibration,fee_sensitivity=pd.DataFrame(sensitivity))
    for name, table in tables.items():table.to_csv(out/f'{name}.csv',index=False)
    # Recent holdout is already explored historical data, not a pristine final test.
    recent=[]
    for h, part in predictions[predictions.date.dt.year>=2025].groupby('horizon'):
        from app.backtesting.metrics import classification_metrics
        for name in ('coin_flip','historical_rate','time_series','logistic','xgboost','existing_ensemble','chronological_calibrated_logistic'):
            m=classification_metrics(part.y,part[name])
            recent.append(dict(horizon=h,model=name,n=len(part),brier=m.brier_score,auc=m.roc_auc))
    pd.DataFrame(recent).to_csv(out/'recent_2025_onwards.csv',index=False)
    summary=dict(feature_set='daily_v3',model_version='phase4_ensemble_v3',features=len(f),
                 start=str(f.date.min().date()),end=str(f.date.max().date()),
                 model_metrics=metrics.to_dict('records'),strategies=strategies.to_dict('records'),
                 production_monthly_parity=True,calibration_promoted=False)
    (out/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False))
    fig=make_subplots(rows=2,cols=1,subplot_titles=['Probability error (Brier, lower is better)','Cumulative market-close saving vs month start (TWD)'],vertical_spacing=.2)
    for h, part in metrics.groupby('horizon'):
        fig.add_trace(go.Bar(x=part.model,y=part.brier,name=f'{h} observations'),row=1,col=1)
    for name in ('model','tranche'):
        fig.add_trace(go.Scatter(x=monthly.month,y=((monthly.fixed_rate-monthly[f'{name}_rate'])*10000).cumsum(),name=name),row=2,col=1)
    fig.update_layout(height=1000,template='plotly_white',title='Official CBC data — exploratory research, no live promotion')
    fig.write_html(out/'dashboard.html',include_plotlyjs=True)
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False))
    print('Artifacts:',out)


if __name__=='__main__':main()
