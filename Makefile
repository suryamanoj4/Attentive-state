.PHONY: sync test lint format smoke smoke-mqar smoke-a5 pilot-h1 pilot-h1-all pilot-a5 aggregate

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check src tests --fix

smoke: smoke-mqar smoke-a5

smoke-mqar:
	uv run readout-recurrence train --config configs/smoke_mqar.yaml

smoke-a5:
	uv run readout-recurrence train --config configs/smoke_a5.yaml

# One-seed, parameter-matched MQAR readout pilot (gated linear vs. attention).
pilot-h1:
	uv run readout-recurrence sweep \
		--config configs/smoke_mqar.yaml \
		--grid model.readout=gated_linear,attention \
		--seeds 0 \
		--output-dir runs/pilot_h1 \
		--match-params \
		--set train.steps=5000 \
		--set train.log_every=1000 \
		--set train.eval_every=1000

# Broader, NOT parameter-matched pilot; use as a capacity/optimization signal.
pilot-h1-all:
	uv run readout-recurrence sweep \
		--config configs/smoke_mqar.yaml \
		--grid model.readout=linear,gated_linear,input_linear,attention,hybrid \
		--seeds 0 \
		--output-dir runs/pilot_h1_all \
		--set train.steps=5000 \
		--set train.log_every=1000 \
		--set train.eval_every=1000

# A5 transition pilot.  Use 50k-100k steps for the real comparison.
pilot-a5:
	uv run readout-recurrence sweep \
		--config configs/a5_base.yaml \
		--grid model.state_type=block_diagonal,cascade_chain \
		--seeds 0 \
		--output-dir runs/pilot_a5 \
		--set train.steps=20000 \
		--set train.log_every=1000 \
		--set train.eval_every=2000

aggregate:
	uv run readout-recurrence aggregate \
		--runs runs/pilot_h1 \
		--output runs/pilot_h1/aggregate \
		--baseline gated_linear \
		--candidate attention
