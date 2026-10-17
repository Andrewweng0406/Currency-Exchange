import importlib.util
import json

import joblib
import pandas as pd
import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.schema import Base, Feature, FxPrice, Prediction, BankRate, ForwardSnapshot, ForwardOutcome
from app.models.baselines import ConstantProbabilityModel
from app.models.training import MODEL_VERSION
from app.monitoring import forward_validation as fv


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    monkeypatch.setattr(fv,'ROOT',tmp_path)
    monkeypatch.setattr(fv,'protocol_spec',lambda:{'version':'test_a','horizons':[1,5,20]})
    artifact_dir=tmp_path/'models/artifacts'
    artifact_dir.mkdir(parents=True)
    origin=pd.Timestamp('2026-10-16',tz='UTC')
    now=origin+pd.Timedelta(hours=12)
    with Session(engine) as session:
        session.add(Feature(feature_set='daily_v3',source='feature_builder',observed_at_utc=origin.to_pydatetime(),
                            values_json=json.dumps({'source':'cbc_taiwan','USDTWD_CLOSE':30.,'DATA_COMPLETENESS':1.})))
        session.add(FxPrice(source='cbc_taiwan',pair='USD/TWD',observed_at_utc=origin.to_pydatetime(),close=30.))
        session.add(BankRate(bank_name='Hua Nan Commercial Bank',source='hua_nan_bank',currency='USD',
                             observed_at_utc=(origin+pd.Timedelta(hours=10)).to_pydatetime(),spot_selling=30.1))
        for h in ('1d','5d','20d'):
            session.add(Prediction(model_version=MODEL_VERSION,horizon=h,source='model_training_latest',
                                   observed_at_utc=origin.to_pydatetime(),prob_up=.6,prob_down=.4))
            artifact={'model_version':MODEL_VERSION,'feature_set':'daily_v3','horizon':h,'feature_columns':['USDTWD_CLOSE','DATA_COMPLETENESS'],
                      'trained_at_utc':(origin+pd.Timedelta(hours=10)).isoformat(),'training_up_rate':.55,
                      'classifiers':{'candidate':ConstantProbabilityModel(.6)},'classification_weights':{'candidate':1.}}
            joblib.dump(artifact,artifact_dir/f'{MODEL_VERSION}_{h}.joblib')
        session.commit()
        yield session,now,tmp_path
    engine.dispose()


def test_capture_is_first_only_and_archives_exact_inputs_and_models(ledger):
    session,now,path=ledger
    first=fv.capture_forward_snapshots(session,now)
    assert first['captured']==3
    rows=session.scalars(select(ForwardSnapshot)).all()
    original=[row.payload_json for row in rows]
    assert len(list((path/'models/artifacts/forward').glob('*.joblib')))==3
    payload=json.loads(rows[0].payload_json)
    assert payload['inputs']['DATA_COMPLETENESS']==1
    assert payload['bank_quote']['source']=='hua_nan_bank'
    assert payload['prob_up']==.6 and payload['coin_flip']==.5
    second=fv.capture_forward_snapshots(session,now+pd.Timedelta(hours=1))
    assert second['duplicates']==3 and second['captured']==0
    assert [row.payload_json for row in session.scalars(select(ForwardSnapshot))]==original


def test_capture_rejects_after_cutoff_even_if_new_data_was_not_downloaded(ledger):
    session,_,_=ledger
    result=fv.capture_forward_snapshots(session,pd.Timestamp('2026-10-19 10:00',tz='UTC'))
    assert result['reason']=='outside_prospective_window'
    assert session.scalar(select(func.count()).select_from(ForwardSnapshot))==0


def test_newer_official_close_rejects_old_feature_row(ledger):
    session,now,_=ledger
    session.add(FxPrice(source='cbc_taiwan',pair='USD/TWD',observed_at_utc=(now.normalize()+pd.Timedelta(hours=1)).to_pydatetime(),close=31.))
    session.commit()
    assert fv.capture_forward_snapshots(session,now)['reason']=='features_behind_official_close'


def test_maturity_and_later_revisions_do_not_rewrite_results(ledger):
    session,now,_=ledger
    fv.capture_forward_snapshots(session,now)
    target=pd.Timestamp('2026-10-19',tz='UTC')
    price=FxPrice(source='cbc_taiwan',pair='USD/TWD',observed_at_utc=target.to_pydatetime(),close=31.)
    session.add(price);session.commit()
    assert fv.evaluate_forward_outcomes(session,target+pd.Timedelta(hours=8))['outcomes_recorded']==0
    assert fv.evaluate_forward_outcomes(session,target+pd.Timedelta(hours=10))['outcomes_recorded']==1
    outcome=session.scalar(select(ForwardOutcome))
    assert outcome.target_up==1 and outcome.close==31.
    evidence=outcome.evidence_json
    price.close=29.;session.commit()
    assert fv.evaluate_forward_outcomes(session,target+pd.Timedelta(hours=11))['outcomes_recorded']==0
    session.refresh(outcome)
    assert outcome.close==31. and outcome.evidence_json==evidence
    summary=fv.forward_validation_summary(session)
    assert summary['cohorts'][0]['matured']==1
    assert summary['cohorts'][0]['pending']==2
    assert summary['automatic_promotion'] is False


def test_code_protocol_changes_start_a_separate_cohort(ledger,monkeypatch):
    session,now,_=ledger
    fv.capture_forward_snapshots(session,now)
    monkeypatch.setattr(fv,'protocol_spec',lambda:{'version':'test_b','horizons':[1,5,20]})
    assert fv.forward_validation_summary(session)['status']=='NOT_STARTED'
    assert fv.capture_forward_snapshots(session,now)['captured']==3
    summary=fv.forward_validation_summary(session)
    assert len(summary['cohorts'])==2
    assert all(group['snapshots']==3 for group in summary['cohorts'])


def test_corrupt_payload_does_not_get_scored(ledger):
    session,now,_=ledger
    fv.capture_forward_snapshots(session,now)
    row=session.scalar(select(ForwardSnapshot))
    row.payload_json='{}';session.commit()
    assert fv.forward_validation_summary(session)['integrity_errors']==1
    with pytest.raises(ValueError,match='fingerprint'):
        fv.evaluate_forward_outcomes(session,now)


def test_database_unique_key_prevents_double_capture(ledger):
    session,now,_=ledger
    fv.capture_forward_snapshots(session,now)
    existing=session.scalar(select(ForwardSnapshot))
    duplicate=ForwardSnapshot(**{col.name:getattr(existing,col.name) for col in ForwardSnapshot.__table__.columns if col.name!='id'})
    with pytest.raises(IntegrityError):
        with session.begin_nested():
            session.add(duplicate);session.flush()


def test_empty_report_does_not_claim_forecasting_skill(ledger):
    session,_,_=ledger
    summary=fv.forward_validation_summary(session)
    assert summary['status']=='NOT_STARTED'
    assert summary['cohorts']==[]


def test_missing_artifact_is_visible_instead_of_claiming_complete_capture(ledger):
    session,now,path=ledger
    (path/f'models/artifacts/{MODEL_VERSION}_20d.joblib').unlink()
    result=fv.capture_forward_snapshots(session,now)
    assert result['captured']==2
    assert result['missing_horizons']==['20d']


def test_additive_migration_is_safe_after_automatic_table_creation():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'migrations/versions/20261016_0002_forward_validation.py'
    spec=importlib.util.spec_from_file_location('forward_migration',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    engine=create_engine('sqlite://')
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()
            module.upgrade()
        assert {'forward_snapshots','forward_outcomes'} <= set(inspect(connection).get_table_names())
def test_cpu_distribution_is_recorded_without_requiring_gpu_package(monkeypatch):
    from app.monitoring.forward_validation import _runtime_packages
    from importlib.metadata import version, PackageNotFoundError
    def installed(name):
        if name == "xgboost-cpu":
            return "3.4.1"
        if name == "xgboost":
            raise PackageNotFoundError(name)
        return version(name)
    monkeypatch.setattr("app.monitoring.forward_validation.importlib.metadata.version", installed)
    assert _runtime_packages()["xgboost_distribution"] == "xgboost-cpu"
    assert _runtime_packages()["xgboost"] == "3.4.1"
