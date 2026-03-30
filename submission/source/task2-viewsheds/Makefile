.PHONY: all deterministic deterministic3d model figures report check smoke

all: deterministic deterministic3d model figures report check

deterministic:
	python3 scripts/run_viewshed.py

deterministic3d:
	python3 scripts/run_viewshed_3d.py

model:
	python3 scripts/train_viewshed_model.py
	python3 scripts/predict_viewshed_model.py

figures:
	python3 scripts/make_figures.py

report:
	python3 scripts/report_quality.py

check:
	python3 scripts/check_task2_completion.py

smoke:
	python3 -m unittest discover -s tests -p 'test_*.py' -v
