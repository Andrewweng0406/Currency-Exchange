"""Read-only, reproducible historical diagnostics; writes reports, never model/DB state."""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import MetaData, Table, func, inspect, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtesting.dataset import add_forward_targets, load_feature_frame, modeling_columns
from app.backtesting.metrics import classification_metrics
from app.backtesting.splits import yearly_expanding_splits
from app.backtesting.uncertainty import InsufficientBlocksError, block_mean_interval
from app.config import ROOT, risk_policy, settings
from app.database.schema import BankRate, Feature, ForeignFlow, FxPrice, MarketData, Prediction
from app.database.session import make_engine
from app.models.training import TimeSeriesDirectionModel, _classification_models, _weights_from_scores
from app.models.calibration import fit_chronological_calibration


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def block_interval(values, block_size=20, repeats=1000):
    """Circular moving-block bootstrap of a mean, preserving local dependence."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [None, None]
    try:
        return list(block_mean_interval(values, block_size, repeats))
    except InsufficientBlocksError:
        return [None, None]


def inventory(session):
    tables = []
    metadata = MetaData()
    for table in inspect(session.bind).get_table_names():
        if table == 'alembic_version':
            continue
        reflected = Table(table, metadata, autoload_with=session.connection())
        time_column = next((name for name in ('observed_at_utc', 'origin_utc', 'target_utc')
                            if name in reflected.c), None)
        statement = select(func.count()).select_from(reflected)
        if time_column:
            column = reflected.c[time_column]
            statement = statement.add_columns(func.min(column), func.max(column))
        row = session.execute(statement).one()
        tables.append(dict(table=table, rows=row[0], time_column=time_column,
                           start=row[1] if time_column else None,
                           end=row[2] if time_column else None))
    sources = []
    for model, dimensions in [
        (FxPrice, ['pair', 'source']), (MarketData, ['symbol', 'source']),
        (ForeignFlow, ['market', 'source']), (BankRate, ['currency', 'bank_name', 'source']),
        (Feature, ['feature_set']), (Prediction, ['model_version', 'horizon']),
    ]:
        cols = [getattr(model, d) for d in dimensions]
        rows = session.execute(select(*cols, func.count(), func.min(model.observed_at_utc),
                                      func.max(model.observed_at_utc)).group_by(*cols)).all()
        for row in rows:
            item = dict(zip(dimensions, row[:len(dimensions)]))
            item.update(table=model.__tablename__, rows=row[-3], start=str(row[-2]), end=str(row[-1]))
            sources.append(item)
    return tables, pd.DataFrame(sources)


def stationary_columns(columns):
    tokens = ('RETURN_', 'CHANGE_', 'DISTANCE_SMA_', 'VOLATILITY_', 'RSI_', 'BOLLINGER_', 'ZSCORE')
    return [c for c in columns if any(t in c for t in tokens)]


def evaluate_models(features, calibrate=False):
    predictions, fold_features = [], []
    for horizon in (1, 5, 20):
        print(f'Walk-forward horizon {horizon}', flush=True)
        frame = add_forward_targets(features, horizons=(horizon,)).frame.replace([np.inf, -np.inf], np.nan)
        target, end = f'TARGET_UP_{horizon}D', f'TARGET_END_{horizon}D'
        frame = frame.dropna(subset=[target, 'USDTWD_CLOSE']).reset_index(drop=True)
        all_cols = modeling_columns(frame)
        prior_parts = []
        for split in yearly_expanding_splits(frame, label_end_col=end):
            train, test = frame.loc[split.train_index], frame.loc[split.test_index]
            assert train[end].max() < test['date'].min()
            y_train = train[target].astype(int)
            stable = [c for c in stationary_columns(all_cols) if train[c].notna().mean() >= .80]
            fx_only = [c for c in stable if c.startswith('USDTWD_')]
            models = _classification_models()
            models['time_series'] = TimeSeriesDirectionModel(horizon)
            probs = {}
            for name, model in models.items():
                model.fit(train[all_cols], y_train)
                probs[name] = model.predict_proba(test[all_cols])[:, 1]
            for name, cols in [('stationary_logistic', stable), ('fx_only_logistic', fx_only)]:
                if not cols:
                    raise ValueError(f'No usable columns for {name}')
                model = _classification_models()['logistic']
                model.fit(train[cols], y_train)
                probs[name] = model.predict_proba(test[cols])[:, 1]
                fold_features.append(dict(horizon=horizon, year=split.test_start.year, model=name,
                                          train_rows=len(train), test_rows=len(test), n_features=len(cols),
                                          columns=';'.join(cols)))
            if calibrate:
                calibrated = fit_chronological_calibration(
                    _classification_models()["logistic"], train[all_cols], y_train,
                    train["date"], train[end],
                )
                assert calibrated.fit_label_end < calibrated.calibration_start
                probs["chronological_calibrated_logistic"] = calibrated.predict_proba(test[all_cols])[:, 1]
                fold_features.append(dict(horizon=horizon, year=split.test_start.year,
                                          model="chronological_calibrated_logistic", train_rows=len(train),
                                          test_rows=len(test), n_features=len(all_cols), columns=";".join(all_cols),
                                          fit_label_end=str(calibrated.fit_label_end),
                                          calibration_start=str(calibrated.calibration_start),
                                          calibration_end=str(calibrated.calibration_end)))
            weights = {n: float(n == "coin_flip") for n in models}
            if prior_parts:
                known = pd.concat(prior_parts, ignore_index=True)
                known = known[known['label_end'] < split.test_start]
                if not known.empty:
                    scores = {n: classification_metrics(known['y'], known[n]) for n in weights}
                    weights = _weights_from_scores(scores)
            probs['existing_ensemble'] = sum(probs[n] * w for n, w in weights.items())
            probs['coin_flip'] = np.full(len(test), .5)
            probs['training_base_rate'] = np.full(len(test), float(y_train.mean()))
            part = pd.DataFrame(dict(date=test['date'].to_numpy(), horizon=horizon,
                                     y=test[target].astype(int).to_numpy(),
                                     target_return=test[f'TARGET_RETURN_{horizon}D'].to_numpy(),
                                     label_end=test[end].to_numpy(), **probs))
            predictions.append(part)
            prior_parts.append(part)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_features)


def model_summary(predictions):
    excluded = {'date', 'horizon', 'y', 'target_return', 'label_end'}
    models = [c for c in predictions if c not in excluded]
    rows, yearly = [], []
    for h, data in predictions.groupby('horizon'):
        for name in models:
            metrics = classification_metrics(data['y'], data[name])
            delta = (data[name] - data['y']) ** 2 - (data['training_base_rate'] - data['y']) ** 2
            low, high = block_interval(delta, max(20, 2 * int(h)))
            coin_low, coin_high = block_interval((data[name] - data["y"]) ** 2 - 0.25, max(20, 2 * int(h)))
            rows.append(dict(brier_delta_vs_coin=metrics.brier_score - .25, coin_ci_low=coin_low, coin_ci_high=coin_high, horizon=int(h), model=name, n=len(data), start=str(data['date'].min().date()),
                             end=str(data['date'].max().date()), accuracy=metrics.accuracy,
                             auc=metrics.roc_auc, brier=metrics.brier_score,
                             calibration_error=metrics.calibration_error, brier_delta_vs_prior=delta.mean(),
                             delta_ci_low=low, delta_ci_high=high))
            for year, group in data.groupby(data['date'].dt.year):
                score = classification_metrics(group['y'], group[name])
                yearly.append(dict(horizon=int(h), model=name, year=int(year), n=len(group),
                                   auc=score.roc_auc, brier=score.brier_score, accuracy=score.accuracy))
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def market_summary(features):
    returns = features['USDTWD_CLOSE'].pct_change(fill_method=None)
    rows = []
    for year, part in features.groupby(features['date'].dt.year):
        sample = returns.loc[part.index].dropna()
        rows.append(dict(year=int(year), observations=len(part), first=float(part.iloc[0]['USDTWD_CLOSE']),
                         last=float(part.iloc[-1]['USDTWD_CLOSE']), minimum=float(part['USDTWD_CLOSE'].min()),
                         maximum=float(part['USDTWD_CLOSE'].max()),
                         return_in_observed_year=float((1 + sample).prod() - 1),
                         realized_daily_std=float(sample.std()), annualized_volatility=float(sample.std() * np.sqrt(252)),
                         fraction_up=float((sample > 0).mean()), worst_usd_up_day=float(sample.max()),
                         worst_usd_down_day=float(sample.min())))
    targets = add_forward_targets(features).frame
    tails = []
    for h in (1, 5, 20):
        r = targets[f'TARGET_RETURN_{h}D'].dropna()
        tails.append(dict(horizon=h, n=len(r), mean=float(r.mean()), std=float(r.std()),
                          up_fraction=float((r > 0).mean()), p10=r.quantile(.10), p90=r.quantile(.90),
                          probability_up_gt_1pct=float((r > .01).mean()),
                          probability_up_gt_2pct=float((r > .02).mean())))
    return pd.DataFrame(rows), pd.DataFrame(tails)


def signal_correlations(features):
    targets = add_forward_targets(features).frame
    columns = stationary_columns(modeling_columns(features))
    rows = []
    for h in (1, 5, 20):
        for col in columns:
            for period, minimum in [('all_history', 2016), ('2021_onwards', 2021)]:
                valid = targets.loc[targets['date'].dt.year >= minimum, ['date', col, f'TARGET_RETURN_{h}D']].dropna()
                if len(valid) < 100 or valid[col].nunique() < 3:
                    continue
                corr = valid[col].corr(valid[f'TARGET_RETURN_{h}D'], method='spearman')
                # Do not present overlapping multi-day rows as independent observations.
                spaced = valid.iloc[::h]
                non_overlap = spaced[col].corr(spaced[f'TARGET_RETURN_{h}D'], method='spearman')
                by_year = [g[col].corr(g[f'TARGET_RETURN_{h}D'], method='spearman')
                           for _, g in valid.groupby(valid['date'].dt.year) if len(g) >= 100 and g[col].nunique() >= 3]
                rows.append(dict(horizon=h, feature=col, period=period, n=len(valid), spearman=corr,
                                 nonoverlap_spearman=non_overlap, nonoverlap_n=len(spaced),
                                 years=len(by_year), same_sign_year_fraction=float(np.mean(np.sign(by_year) == np.sign(corr)))
                                 if by_year else np.nan))
    return pd.DataFrame(rows)


def quality_diagnostics(features):
    availability = []
    for col in modeling_columns(features):
        values = features.loc[features[col].notna(), "date"]
        availability.append(dict(feature=col, missing_fraction=features[col].isna().mean(),
                                 first_available=str(values.min().date()) if len(values) else None,
                                 last_available=str(values.max().date()) if len(values) else None))
    moves = features[["date", "USDTWD_CLOSE", "source"]].copy()
    moves["previous_date"] = moves.date.shift(1)
    moves["previous_close"] = moves.USDTWD_CLOSE.shift(1)
    moves["previous_source"] = moves.source.shift(1)
    moves["gap_days"] = (moves.date - moves.previous_date).dt.days
    moves["return_1d"] = moves.USDTWD_CLOSE.pct_change(fill_method=None)
    extremes = moves.loc[moves.return_1d.abs().nlargest(12).index].sort_values("date")
    return pd.DataFrame(availability), extremes


def calibration_bins(predictions):
    rows = []
    for horizon, part in predictions.groupby("horizon"):
        for name in ("logistic", "xgboost", "time_series", "existing_ensemble", "fx_only_logistic"):
            bands = pd.cut(part[name], np.linspace(0, 1, 11), include_lowest=True)
            for band, group in part.groupby(bands, observed=True):
                rows.append(dict(horizon=int(horizon), model=name, probability_band=str(band), n=len(group),
                                 mean_probability=group[name].mean(), realized_up_fraction=group.y.mean()))
    return pd.DataFrame(rows)


def monthly_comparison(features, predictions, target_usd=10000):
    """Fixed calendar month budgets; no partial final month and no tuned thresholds."""
    price = features[['date', 'USDTWD_CLOSE']].copy()
    policy = risk_policy()
    score = np.zeros(len(price))
    for name, weight in policy['opportunity_score']['weights'].items():
        window = int(name.split('_')[1].removesuffix('d'))
        score += price['USDTWD_CLOSE'].rolling(window, min_periods=1).apply(
            lambda x: 1 - (x <= x[-1]).mean(), raw=True,
        ).to_numpy() * weight * 100
    price['opportunity_score'] = np.rint(score)
    pred5 = predictions[predictions['horizon'] == 5][['date', 'existing_ensemble']]
    price = price.merge(pred5, on='date', how='left')
    cfg = policy['strategy_backtest']['model_timing_once']
    latest_month = price['date'].max().tz_localize(None).to_period('M')
    records = []
    for month, group in price.groupby(price['date'].dt.tz_localize(None).dt.to_period('M')):
        if month.year < 2023 or month >= latest_month or len(group) < 10 or group['existing_ensemble'].isna().any():
            continue
        group = group.reset_index(drop=True)
        fixed_rate = float(group.iloc[0]['USDTWD_CLOSE'])
        tranche_rates = []
        for day in (1, 8, 15, 22):
            available = group[group['date'].dt.day >= day]
            if available.empty:
                break
            tranche_rates.append(float(available.iloc[0]['USDTWD_CLOSE']))
        if len(tranche_rates) != 4:
            continue
        deadline = month.end_time.tz_localize('UTC').normalize()
        funding_cutoff = deadline - pd.Timedelta(days=int(cfg['deadline_buffer_days']))
        funding_cutoff = group.loc[group.date <= funding_cutoff, 'date'].max()
        model_row = group.iloc[-1]
        reason = 'deadline'
        for idx, row in group.iterrows():
            if row['date'] >= funding_cutoff:
                model_row = row
                break
            signal = row['existing_ensemble'] >= cfg['risk_probability_up_min'] or (
                row['opportunity_score'] >= cfg['opportunity_score_min'] and
                row['existing_ensemble'] <= cfg['favorable_probability_up_max'])
            if signal and idx + 1 < len(group):
                model_row = group.iloc[idx + 1]
                reason = 'signal_next_observation'
                break
        records.append(dict(month=str(month), observations=len(group), fixed_rate=fixed_rate,
                            tranche_rate=float(np.mean(tranche_rates)), model_rate=float(model_row['USDTWD_CLOSE']),
                            model_execution_date=str(model_row['date'].date()), trigger=reason))
    monthly = pd.DataFrame(records)
    comparisons = []
    for name in ('fixed', 'tranche', 'model'):
        rates = monthly[f'{name}_rate']
        savings = (monthly['fixed_rate'] - rates) * target_usd
        low, high = block_interval(savings, 3)
        comparisons.append(dict(strategy=name, months=len(monthly), target_usd=target_usd,
                                average_rate=rates.mean(), worst_rate=rates.max(), rate_std=rates.std(ddof=0),
                                mean_savings_twd=savings.mean(), beat_fixed_fraction=float((savings > 0).mean()),
                                savings_ci_low=low, savings_ci_high=high))
    return monthly, pd.DataFrame(comparisons)


def markdown_table(frame, columns, formats=None):
    formats = formats or {}
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        parts = []
        for col in columns:
            value = row[col]
            if col in formats and pd.notna(value):
                value = formats[col].format(value)
            elif pd.isna(value):
                value = 'N/A'
            parts.append(str(value))
        lines.append('| ' + ' | '.join(parts) + ' |')
    return '\n'.join(lines)


def write_report(out, features, tables, coverage, yearly, tails, metrics, yearly_metrics, signals, monthly, strategies, folds):
    feature_set = features.attrs.get("feature_set", "unspecified")
    frames = dict(coverage=coverage, yearly_returns=yearly, return_distribution=tails, model_comparison=metrics,
                  yearly_model_metrics=yearly_metrics, signal_correlations=signals, monthly_strategy=monthly,
                  strategy_summary=strategies, fold_features=folds)
    availability, extremes = quality_diagnostics(features)
    frames["feature_availability"] = availability
    frames["extreme_moves_to_verify"] = extremes
    for name, frame in frames.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    summary = dict(feature_rows=len(features), feature_start=str(features['date'].min().date()),
                   feature_end=str(features['date'].max().date()), tables=tables,
                   features=len(modeling_columns(features)), missing=features.isna().mean().to_dict(),
                   model_comparison=metrics.to_dict('records'), strategies=strategies.to_dict('records'))
    (out / 'summary.json').write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False))
    fmt = {'brier': '{:.4f}', 'auc': '{:.3f}', 'accuracy': '{:.1%}', 'brier_delta_vs_prior': '{:+.4f}',
           'delta_ci_low': '{:+.4f}', 'delta_ci_high': '{:+.4f}'}
    model5 = metrics[metrics.horizon == 5].sort_values('brier')
    top = signals[(signals.horizon == 5) & (signals.period == '2021_onwards')].copy()
    top = top.assign(magnitude=top.spearman.abs()).sort_values('magnitude', ascending=False).head(8)
    best = model5.iloc[0]
    ensemble5 = model5[model5.model == "existing_ensemble"].iloc[0]
    logistic5 = model5[model5.model == "logistic"].iloc[0]
    feature_sources = features["source"].value_counts().to_dict()
    yearly5 = yearly_metrics[(yearly_metrics.horizon == 5) &
                            yearly_metrics.model.isin(["coin_flip", "logistic", "existing_ensemble"])].pivot(
                                index="year", columns="model", values="brier").reset_index()
    nontrivial = metrics[~metrics.model.isin(["coin_flip", "training_base_rate", "historical_rate"])]
    baseline_finding = ("部分研究候選的平均 Brier 低於固定 50%；是否有穩定優勢仍需看跨年及區塊區間，不能只憑平均值啟用。"
                        if nontrivial.brier.lt(.25).any() else
                        "所有研究候選的平均 Brier 都未勝過固定 50%，目前沒有足夠機率預測優勢。")
    bank_rows = next((t["rows"] for t in tables if t["table"] == "bank_rates"), 0)
    report = f'''# 歷史資料分析：截至 {features['date'].max().date()}

## 主要發現

- {baseline_finding} 「勝過含動能的基準」也不是有效預測的充分證據。
- 5 期原始 logistic 的 AUC {logistic5['auc']:.3f} 有些排序資訊，但 Brier {logistic5['brier']:.4f} 很差；現有 ensemble Brier {ensemble5['brier']:.4f}，固定 50% 為 0.25。需同時檢查機率誤差、校準分箱與跨年穩定性。
- 減少成為價格水準的特徵有改善部分機率誤差，但沒有因此建立可用的預測優勢。這次不把歷史測試排名當成正式模型選型。
- 完整月份、固定預算的市場價比較顯示：分批與模型選時相對月初一次換匯的平均成本差異，其區塊不確定區間均跨過零，沒有足夠證據宣稱穩定節省。

## 資料是否保留

有。本報告從目前設定的 database 唯讀取數據，未寫入 database、改動正式模型、部署或發送通知。特徵採用 {feature_set}，共 {len(features):,} 個日期，{features['date'].min().date()} 至 {features['date'].max().date()}。v1 特徵與預測也保留，見下表。

{markdown_table(pd.DataFrame(tables), ['table', 'rows', 'time_column', 'start', 'end'])}

market_data 的 1962 年起始日主要是美債利率，不能解讀為有六十年的 USD/TWD 歷史。銀行報價與線上歷史預測非常少，長期市場資料與真實線上預測證據不可混為一談。資料表按來源與日期更新同一筆資料，不是每次下載都有完整版本保存；本機另有前次稽核備份，但不等於完整 vintage 資料庫。

### 資料來源與缺口

本次匯率特徵來源分布：{feature_sources}。央行收盤有 {feature_sources.get('cbc_taiwan', 0)} 筆；任何市場收盤均不等於銀行對客戶的可成交價。資料中只有約一年外資流量、CNH 僅零星數筆，經濟事件與 macro_data 表是空的。長區間不代表每個變數都有同樣長、同樣可靠的歷史。

最大絕對單筆變動的日期如下，只列資料中觀測值，尚未向第二個獨立來源核實。少數劇烈反轉可能是真實事件、不同報價時段或資料錯誤；不能自行刪掉讓回測變漂亮，也不能先當成真實尾部風險。完整 12 筆及前後來源見 extreme_moves_to_verify.csv，逐欄缺失率見 feature_availability.csv。

{markdown_table(extremes.sort_values('return_1d', key=lambda x:x.abs(), ascending=False).head(6), ['date','previous_close','USDTWD_CLOSE','return_1d','gap_days','source'], {'previous_close':'{:.4f}','USDTWD_CLOSE':'{:.4f}','return_1d':'{:+.2%}'})}

## 歷史匯率風險

年度報酬按當年有效單日報酬連乘，年初與上一個已觀測日銜接；第一年從樣本起始算，最後一年尚未結束。正報酬表示美元變貴／台幣貶值。波動年化採 sqrt(252)，為比較尺度，資料日期不等同經核對的完整交易日曆。

{markdown_table(yearly, ['year', 'observations', 'minimum', 'maximum', 'return_in_observed_year', 'annualized_volatility'], {'return_in_observed_year': '{:+.2%}', 'annualized_volatility': '{:.2%}', 'minimum': '{:.3f}', 'maximum': '{:.3f}'})}

{markdown_table(tails, ['horizon', 'n', 'up_fraction', 'p10', 'p90', 'probability_up_gt_1pct', 'probability_up_gt_2pct'], {c:'{:.2%}' for c in ['up_fraction','p10','p90','probability_up_gt_1pct','probability_up_gt_2pct']})}

horizon 是往後第 1／5／20 筆有效市場觀測，不保證等於日曆天，也未核對每一個台灣交易日曆版本。分布是無條件歷史頻率，不能直接當今天的條件機率。

## 哪種模型真的有預測資訊

以至少五個歷年作訓練、逐年向前測試；訓練標籤到期日必須早於測試第一天。相同日期／樣本比較所有模型，補值及標準化只在訓練期 fit。stationary_logistic 只用報酬、利率變化、均線偏離、波動與標準化指標，並在每次訓練內要求至少 80% 可用；fx_only_logistic 再限制為 USD/TWD 自身指標。這是研究診斷，沒有取代正式模型。

5 筆觀測期，最低 Brier 的候選為 **{best['model']}，Brier {best['brier']:.4f}**。排序本身不代表已驗證的投資優勢，詳見跨年結果與區塊區間。

{markdown_table(model5, ['model', 'n', 'accuracy', 'auc', 'brier', 'brier_delta_vs_prior', 'delta_ci_low', 'delta_ci_high'], fmt)}

coin_flip 永遠預測 50%；training_base_rate 只使用當年訓練期間的實際上漲比例；time_series 是原有含固定動能加成的基準；existing_ensemble 是現有權重規則的真實逐年預測。Brier 衡量機率誤差，越低越好；AUC 衡量排序能力，並不表示報價便宜或能省錢。

brier_delta_vs_prior 為相對 training_base_rate 的損失差，負數較好。95% 區間使用 1,000 次循環移動區塊 bootstrap，區塊長 max(20, 2×horizon)，保留局部相關性；它是近似估計，未調整本次多模型／多訊號探索。不能將多日重疊樣本當獨立證據，也不能因探索排序第一就直接啟用。

5 期跨年 Brier 對照：

{markdown_table(yearly5, ['year','coin_flip','logistic','existing_ensemble'], {c:'{:.4f}' for c in ['coin_flip','logistic','existing_ensemble']})}

校準分箱在 calibration_bins.csv，可對照「預測 80%」的樣本實際有多少比例上漲；這是歷史誤差檢視，沒有拿測試答案重新 fit 機率。

完整 1／5／20 期結果在 model_comparison.csv；跨年穩定性在 yearly_model_metrics.csv；每個訓練期實際使用欄位在 fold_features.csv；逐筆樣本外輸出在 oof_predictions.csv。

## 哪些訊號值得後續研究

以下為 2021 年起與未來五筆報酬絕對 Spearman 相關較大的八個指標。只是探索性排序，可能包含同一類重複訊號與選擇偏誤，不能視為因果或拿來宣稱可賺錢。

{markdown_table(top, ['feature', 'n', 'spearman', 'nonoverlap_spearman', 'years', 'same_sign_year_fraction'], {'spearman':'{:+.3f}', 'nonoverlap_spearman':'{:+.3f}', 'same_sign_year_fraction':'{:.0%}'})}

nonoverlap_spearman 另外每隔五個有效配對抽樣一次，僅供敏感度對照；來源缺失會改變配對序列，因此不保證每段日曆目標完全不重疊。same_sign_year_fraction 為當年樣本至少 100 筆時，年度相關方向與整體一致的比例。原始指標／資料缺失仍可能反映來源或市場時段差異。

## 更明確的月度換匯比較

每個完整月份假設需 USD 10,000、月底付款：fixed 在月內第一筆有效觀測換完；tranche 在日曆 1／8／15／22 日或其後首筆觀測各換 25%；model 使用原有固定門檻，訊號後下一筆觀測成交，在月底前三個日曆日安排保底換匯，休市則按央行報價日曆代理提前。只用已完成的月份，不使用跨月重疊預算窗口、不調參，起點 2023 年。

{markdown_table(strategies, ['strategy', 'months', 'average_rate', 'worst_rate', 'rate_std', 'mean_savings_twd', 'beat_fixed_fraction', 'savings_ci_low', 'savings_ci_high'], {'average_rate':'{:.4f}', 'worst_rate':'{:.4f}', 'rate_std':'{:.4f}', 'mean_savings_twd':'{:+,.0f}', 'beat_fixed_fraction':'{:.0%}', 'savings_ci_low':'{:+,.0f}', 'savings_ci_high':'{:+,.0f}'})}

正 mean_savings_twd 表示比月初換匯省台幣，95% 區間以月份為單位、三個月區塊抽樣。本次月度定義刻意明確化，與前次既有回測的「月底往前 30 天」不同，不能直接比較兩份報告的 NT$ 差額。此比較仍使用市場收盤價，未含華南賣出價、銀行營業時間、價差、費用及真實付款日；不能稱為家庭實際省錢績效。

## 下一步與分析限制

1. 以純 50%／歷史漲幅比例為最低門檻，檢查現有動能基準本身是否製造誤差。逐年比較比單一整體 AUC 更重要。
2. 研究價格水準對跨年模型的不穩定性；可用特徵和試驗規則先固定，未來再累積真正未見資料驗證，避免反覆挑選同一段歷史最好的模型。
3. 先補齊歷史與持續蒐集實際使用銀行的美元即期賣出價、完整交易日曆／資料品質，才有基礎驗證可成交成本。資料庫 {bank_rows} 筆銀行報價不足以支持長期換匯績效。
4. FRED H.10 已按隔週發布與美國假日對齊，其他跨市場來源仍有延後日期代理，未具完整發布時間／歷史版本；故即使已排除訓練標籤交界，仍不能宣稱徹底消除所有前視偏誤。歷史特徵未完整對齊的來源尤其需保守解讀。
5. 本分析不自動提高正式模型權重、解除模型建議守門條件或部署策略。

產物：dashboard.html 可互動查看匯率、波動、模型機率誤差及換匯差額；coverage.csv 提供逐來源覆蓋，其他 CSV 與 summary.json 可重算。執行方式：`.venv/bin/python scripts/analyze_history.py --as-of {features['date'].max().date()}`。
'''
    (out / 'report.md').write_text(report)
    fig = make_subplots(rows=4, cols=1, vertical_spacing=.07,
                        subplot_titles=('USD/TWD history', 'Annual realized volatility', '5-observation probability errors (lower is better)', 'Cumulative cost saving vs first observation of month (TWD)'))
    fig.add_trace(go.Scatter(x=features.date, y=features.USDTWD_CLOSE, name='USD/TWD'), row=1, col=1)
    fig.add_trace(go.Bar(x=yearly.year, y=yearly.annualized_volatility, name='Annual volatility'), row=2, col=1)
    fig.add_trace(go.Bar(x=model5.model, y=model5.brier, name='Brier score'), row=3, col=1)
    for name in ('model', 'tranche'):
        saving = (monthly.fixed_rate - monthly[f'{name}_rate']) * 10000
        fig.add_trace(go.Scatter(x=monthly.month, y=saving.cumsum(), name=name), row=4, col=1)
    fig.update_layout(height=1450, template='plotly_white', title='USD/TWD historical diagnostics — research only', showlegend=True)
    fig.update_yaxes(tickformat='.1%', row=2, col=1)
    figure_html = fig.to_html(full_html=False, include_plotlyjs=True)
    (out / 'dashboard.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>匯率歷史分析</title>'
        '<style>body{font-family:system-ui;margin:32px;max-width:1500px}pre{white-space:pre-wrap;line-height:1.65}</style>'
        '<h1>匯率歷史分析</h1><p>研究回測；未計銀行價差與費用。圖表可以縮放、滑鼠查看數值與切換曲線。</p>'
        + figure_html + '<h2>完整分析與方法</h2><pre>' + html.escape(report) + '</pre></html>')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', default='2026-10-16')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--feature-set', default="daily_v3")
    args = parser.parse_args()
    out = args.output_dir or ROOT / 'data' / 'analysis' / f'{args.as_of}_{args.feature_set}'
    out.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings()['database']['url'])
    with engine.connect() as connection:
        if engine.dialect.name == 'sqlite':
            connection.exec_driver_sql('PRAGMA query_only = ON')
        elif engine.dialect.name == 'postgresql':
            connection.exec_driver_sql('SET TRANSACTION READ ONLY')
        with Session(bind=connection) as session:
            tables, coverage = inventory(session)
            features = load_feature_frame(session, feature_set=args.feature_set)
            cutoff = pd.Timestamp(args.as_of, tz='UTC') + pd.Timedelta(days=1)
            features = features[features.date < cutoff].reset_index(drop=True)
            features.attrs["feature_set"] = args.feature_set
            if features.empty:
                raise ValueError(f'No {args.feature_set} features before cutoff')
            print(f'Read {len(features)} feature rows; database is read-only', flush=True)
            predictions, folds = evaluate_models(features)
    predictions.to_csv(out / 'oof_predictions.csv', index=False)
    calibration_bins(predictions).to_csv(out / "calibration_bins.csv", index=False)
    metrics, yearly_metrics = model_summary(predictions)
    yearly, tails = market_summary(features)
    signals = signal_correlations(features)
    monthly, strategies = monthly_comparison(features, predictions)
    report = write_report(out, features, tables, coverage, yearly, tails, metrics, yearly_metrics, signals, monthly, strategies, folds)
    print(report)
    print(f'Artifacts: {out}')


if __name__ == '__main__':
    main()
