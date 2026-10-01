.PHONY: install test api web daily-tw daily-us train rank bench charts tenant docker
install:        ## local dev install (M2 Mac: add [models] for TimesFM/Chronos)
	uv venv --python python3.11 .venv && . .venv/bin/activate && uv pip install -e ".[models,llm,api,dev]"
test:
	python -m pytest -q
api:            ## B2B API on :8000, docs at /docs
	uvicorn fintech_agent.api.main:app --reload --port 8000
web:            ## web app on :8501
	streamlit run fintech_agent/ui/app.py
daily-tw:
	python scripts/daily_job.py --market TW
daily-us:
	python scripts/daily_job.py --market US
train:
	python scripts/train_models.py --universe tw50 --models lgbm --horizons 5 20 --years 13
	python scripts/train_models.py --universe us50 --models lgbm --horizons 5 20 --years 13
rank:
	python scripts/rank_stocks.py --pool twse --pit-top 50 --horizon 5
charts:
	python scripts/export_benchmark_charts.py && python scripts/plot_benchmarks.py
tenant:         ## make tenant NAME="示範投顧" PLAN=pro
	python scripts/manage_tenants.py create --name "$(NAME)" --plan $(PLAN)
docker:
	docker compose up -d --build
