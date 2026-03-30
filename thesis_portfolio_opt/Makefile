.PHONY: install fetch features train backtest figures pipeline test dashboard clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

fetch:  ## Fetch raw data from FRED & YFinance
	python run_pipeline.py --step fetch

features:  ## Build feature matrix
	python run_pipeline.py --step features

train:  ## Train predictive models
	python run_pipeline.py --step train

backtest:  ## Run backtests
	python run_pipeline.py --step backtest

figures:  ## Generate thesis figures
	python run_pipeline.py --step figures

pipeline:  ## Run full pipeline (fetch → features → train → backtest → figures)
	python run_pipeline.py --step all

test:  ## Run test suite
	pytest tests/ -v

test-unit:  ## Run unit tests only (no data required)
	pytest tests/ -v -m unit

dashboard:  ## Launch Streamlit dashboard
	streamlit run app/dashboard.py

clean:  ## Remove generated data files
	rm -f data/raw/*.csv data/processed/*.csv data/results/*.csv
	rm -f data/results/*.pkl data/results/*.pdf data/results/*.png
	rm -f data/results/*.tex data/results/*.xlsx data/results/*.log
