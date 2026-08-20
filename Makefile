.PHONY: test validate-data prepare-inputs

test:
	python -m pytest -q

validate-data:
	python dataset_tools/validate_dataset.py

prepare-inputs:
	bash scripts/prepare_baseline_inputs.sh
