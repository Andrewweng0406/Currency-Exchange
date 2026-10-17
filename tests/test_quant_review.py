import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.schema import Base, BankRate, MarketData, FxPrice, Prediction
from app.features.availability import h10_available_dates
from app.features.builder import _load_market, _load_fx
from app.features.indicators import rsi
from app.models.baselines import ConstantProbabilityModel
from app.models.calibration import fit_chronological_calibration
from app.models.training import _classification_models, _weights_from_scores
from app.ops.bank_quotes import latest_bank_quote
from app.providers.base import ProviderError
from app.providers.cbc import parse_daily_archive
from app.providers.hncb import HuaNanBankProvider
from app.backtesting.exchange_strategy import fixed_day_exchange, equal_tranche_exchange
from app.backtesting.strategy_compare import _model_timing_once, TimingPolicy, _strategy_records


def test_cbc_archive_rejects_conflicting_or_nonpositive_prices():
    with pytest.raises(ProviderError):
        parse_daily_archive([{'日期':'20200101','NTD_USD':'30'},{'日期':'20200101','NTD_USD':'31'}])
    for bad in ('NaN','inf','0','-1'):
        with pytest.raises(ProviderError):
            parse_daily_archive([{'日期':'20200101','NTD_USD':bad}])


def test_cbc_archive_parses_calendar_dates_without_roc_guessing():
    frame = parse_daily_archive([{'日期':'20180226','NTD_USD':'29.235'}])
    assert frame.iloc[0].date == pd.Timestamp('2018-02-26',tz='UTC')
    assert frame.iloc[0].close == 29.235


def test_hncb_captured_empty_type_is_spot_and_uses_quote_time(monkeypatch):
    row={'DATE':'2026-10-16','TIME':'23:56:15','TYPE':'','DESC_ENG':'USD','DESC_CHI':'美金',
         'BUY_AMT_BOARD':'32.275000','SELL_AMT_BOARD':'32.435000'}
    provider=HuaNanBankProvider()
    monkeypatch.setattr(provider,'get',lambda *a,**kw: SimpleNamespace(json=lambda:[row]))
    frame=provider.fetch_current_rates()
    assert frame.iloc[0].spot_selling==32.435
    assert frame.iloc[0].observed_at_utc==pd.Timestamp('2026-10-16 15:56:15',tz='UTC')
    row['SELL_AMT_BOARD']='0'
    with pytest.raises(ProviderError):provider.fetch_current_rates()


def test_h10_waits_for_next_weeks_release_and_federal_holiday():
    dates=pd.Series(pd.to_datetime(['2026-09-28','2026-10-02','2026-10-05','2026-10-09'],utc=True))
    out=h10_available_dates(dates)
    # Sep 28–Oct 2 is released Mon Oct 5; Oct 5–9 waits through Columbus Day.
    assert out.tolist()==[pd.Timestamp('2026-10-06',tz='UTC')]*2+[pd.Timestamp('2026-10-14',tz='UTC')]*2
    assert h10_available_dates(pd.Series(pd.to_datetime(['2008-01-02'],utc=True))).isna().all()


def test_newer_observation_does_not_get_overwritten_by_late_fred():
    engine=create_engine('sqlite://');Base.metadata.create_all(engine)
    with Session(engine) as session:
        for source,date,close in [('yahoo_finance','2020-01-06',100),('yahoo_finance','2020-01-07',101),('fred_csv','2020-01-06',99)]:
            session.add(MarketData(observed_at_utc=pd.Timestamp(date).to_pydatetime(),source=source,symbol='SP500',close=close))
        session.commit()
        frame=_load_market(session)
    assert frame.iloc[-1]['close']==101


def test_official_only_does_not_silently_fill_taiwan_holidays_from_yahoo():
    engine=create_engine('sqlite://');Base.metadata.create_all(engine)
    with Session(engine) as session:
        for source,date in [('cbc_taiwan','2020-01-22'),('yahoo_finance','2020-01-23')]:
            session.add(FxPrice(observed_at_utc=pd.Timestamp(date).to_pydatetime(),source=source,pair='USD/TWD',close=30))
        session.commit()
        assert len(_load_fx(session,source_policy='official_only'))==1


def test_probability_baselines_can_defeat_all_momentum_models():
    scores={name:SimpleNamespace(brier_score=value) for name,value in
            [('coin_flip',.25),('historical_rate',.251),('time_series',.256),('xgboost',.27)]}
    assert _weights_from_scores(scores)=={'coin_flip':1.,'historical_rate':0.,'time_series':0.,'xgboost':0.}
    models=_classification_models()
    x=pd.DataFrame({'x':[1,2,3,4]});y=pd.Series([1,1,1,0])
    assert np.all(models['coin_flip'].fit(x,y).predict_proba(x)[:,1]==.5)
    assert np.all(models['historical_rate'].fit(x,y).predict_proba(x)[:,1]==.75)


def test_rsi_has_meaningful_values_in_one_direction_and_flat_markets():
    assert rsi(pd.Series(range(30),dtype=float),14).iloc[-1]==100
    assert rsi(pd.Series(range(30,0,-1),dtype=float),14).iloc[-1]==0
    assert rsi(pd.Series([10.]*30),14).iloc[-1]==50


def test_calibration_purges_label_overlap_and_never_fits_on_holdout():
    from sklearn.linear_model import LogisticRegression
    dates=pd.Series(pd.bdate_range('2010-01-01',periods=900,tz='UTC'))
    ends=dates.shift(-20)
    # Model fitting history supplied by an outer split already has matured targets.
    ends.iloc[-20:]=pd.date_range(dates.iloc[-1]+pd.Timedelta(days=1),periods=20,tz='UTC')
    x=pd.DataFrame({'a':np.sin(np.arange(900))})
    y=pd.Series(np.arange(900)%2)
    fitted=fit_chronological_calibration(LogisticRegression(),x,y,dates,ends)
    assert fitted.fit_label_end < fitted.calibration_start
    assert fitted.calibration_end==dates.iloc[-1]
    assert fitted.predict_proba(x.tail(3)).shape==(3,2)


def test_family_bank_is_preferred_over_more_recent_fallback():
    engine=create_engine('sqlite://');Base.metadata.create_all(engine)
    with Session(engine) as session:
        for bank,date in [('Hua Nan Commercial Bank','2026-10-16 15:00'),('Land Bank of Taiwan','2026-10-16 23:00')]:
            session.add(BankRate(observed_at_utc=pd.Timestamp(date).to_pydatetime(),source='test',currency='USD',bank_name=bank,spot_selling=32.4))
        session.commit()
        assert latest_bank_quote(session,now=pd.Timestamp('2026-10-17',tz='UTC')).bank_name=='Hua Nan Commercial Bank'


def test_scheduled_exchange_cannot_retroactively_buy_a_previous_price():
    frame=pd.DataFrame({'date':pd.to_datetime(['2026-01-09','2026-01-12'],utc=True),'USDTWD_CLOSE':[30.,31.]})
    result=fixed_day_exchange(frame,1000,pd.Timestamp('2026-01-12',tz='UTC'),days_before=1)
    assert result.average_rate==31
    with pytest.raises(ValueError):
        fixed_day_exchange(frame,1000,pd.Timestamp('2026-01-08',tz='UTC'),days_before=1)
    with pytest.raises(ValueError):
        equal_tranche_exchange(frame,1000,pd.Timestamp('2026-01-08',tz='UTC'))


def test_funding_rolls_before_lunar_holiday_instead_of_inventing_a_trade():
    dates=pd.bdate_range('2025-01-01','2025-01-24',tz='UTC')
    frame=pd.DataFrame({'date':dates,'USDTWD_CLOSE':30+np.arange(len(dates))*.1,'prob_up':.5,'opportunity_score':0})
    policy=TimingPolicy(.65,75,.55,3)
    result=_model_timing_once(frame,1000,policy)
    assert result==pytest.approx(frame.iloc[-1].USDTWD_CLOSE*1000)
    assert not _strategy_records(frame,1000)['model_timing_once'] # January is not known complete yet.


@pytest.mark.parametrize('upper,age_days,expected',[(.01,0,'WEAK'),(-.01,0,'OK'),(-.01,30,'INSUFFICIENT_HISTORY')])
def test_live_skill_needs_all_horizons_baseline_intervals_and_freshness(upper,age_days,expected):
    from app.database.schema import ModelPerformance
    from app.monitoring.model_health import latest_model_health_summary
    engine=create_engine('sqlite://');Base.metadata.create_all(engine)
    now=pd.Timestamp.now(tz='UTC')-pd.Timedelta(days=age_days)
    with Session(engine) as session:
        for h in ('1d','5d','20d'):
            for name,value in [('sample_count',100),('effective_sample_count',12),('skill_upper_vs_coin',upper),('skill_upper_vs_prior',upper)]:
                session.add(ModelPerformance(observed_at_utc=now.to_pydatetime(),source='model_monitoring',model_version='phase4_ensemble_v3',
                                             horizon=h,metric=name,value=value,window='365D'))
        session.commit()
        assert latest_model_health_summary(session).status==expected


def test_positive_average_savings_with_uncertain_interval_does_not_pass():
    from app.backtesting.strategy_compare import StrategyComparison, build_strategy_report
    baseline=StrategyComparison('fixed_day_once',45,31.54,33.196,.907,1000,0,0,0)
    candidate=StrategyComparison('model_timing_once',45,31.513,33.097,.905,900,268,0,.44,
                                  savings_ci_low=-877,savings_ci_high=1401)
    report=build_strategy_report([baseline,candidate])
    assert not report.summary.passed
    assert report.verdict=='MIXED'
    assert not report.should_use_for_timing


def test_blocked_advice_does_not_leak_through_forecast_or_opportunity_alerts():
    from app.alerts.engine import generate_alerts
    from app.ops.data_quality import DataQualitySummary
    risk=SimpleNamespace(twd_risk_score=90,opportunity_score=95)
    quality=DataQualitySummary('UNVERIFIED_MODEL','未驗證','暫停模型建議',True,[])
    alerts=generate_alerts(risk,{'5d':{'prob_up':.95}},data_quality=quality)
    assert {a.alert_type for a in alerts}=={'DATA_QUALITY_WARNING'}
