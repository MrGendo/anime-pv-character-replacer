# W1 scope — 常用入口。uv 不在 PATH 时可 `make UV=/path/to/uv ...` 覆盖。
UV ?= uv

.PHONY: setup test segment

setup:
	$(UV) venv --python 3.12
	$(UV) sync

test:
	$(UV) run pytest -q

segment:
	$(UV) run python -m anime_pv_replacer.segment --config config/pipeline.yaml --stage cut
