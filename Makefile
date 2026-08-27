.PHONY: install test ingest features train decision ai line-dry-run api dashboard strategy health readiness data-quality migrate profile

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest

ingest:
	. .venv/bin/activate && python scripts/ingest_phase1.py

features:
	. .venv/bin/activate && python scripts/build_features.py

train:
	. .venv/bin/activate && python scripts/train_models.py --horizon all

decision:
	. .venv/bin/activate && python scripts/generate_decision.py

ai:
	. .venv/bin/activate && python scripts/generate_ai_interpretation.py --save

line-dry-run:
	. .venv/bin/activate && python scripts/send_line_report.py

api:
	. .venv/bin/activate && uvicorn app.api.main:app --reload --port 8000

dashboard:
	. .venv/bin/activate && streamlit run dashboard/main.py

strategy:
	. .venv/bin/activate && python scripts/backtest_strategy.py

health:
	. .venv/bin/activate && python scripts/evaluate_model_health.py

readiness:
	. .venv/bin/activate && python scripts/check_readiness.py

data-quality:
	. .venv/bin/activate && python scripts/data_quality_report.py

migrate:
	. .venv/bin/activate && alembic upgrade head

profile:
	. .venv/bin/activate && python scripts/configure_profile.py --help
