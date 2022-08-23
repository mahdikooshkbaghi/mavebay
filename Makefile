linter:
	flake8 --per-file-ignores="__init__.py:F401,F403" mavebay
	flake8 --per-file-ignores="__init__.py:F401,F403" tests
	black mavebay
	black tests
	isort --profile black .

test:
	pytest tests/

test-v:
	pytest -v

test-cov:
	pytest --cov-report html --cov=mavebay --disable-warnings