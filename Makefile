linter:
	flake8 mavebay
	flake8 tests/

test:
	pytest tests/

test-v:
	pytest -v

test-cov:
	pytest --cov-report html --cov=mavebay --disable-warnings