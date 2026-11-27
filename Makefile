.PHONY: sync test lint format smoke smoke-mqar smoke-a5 pilot-h1 pilot-h1-all pilot-a5 aggregate \
	followup-h2 followup-rank followup-slots followup-cascade followup-hybrid

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

# H2 on the block-dense state type, where the full run showed a real readout gap.
followup-h2:
	uv run readout-recurrence sweep \
		--config configs/followups/h2_block_dense.yaml \
		--grid model.readout=gated_linear,attention \
		--grid task.mqar.n_pairs=8,16,32,64 \
		--grid task.mqar.n_distractors=0,16,64 \
		--seeds 0,1,2 \
		--match-params \
		--output-dir runs/followup_h2

# H3: resolve the rank-8 input_linear confound.  No parameter matching here;
# the rank is the independent variable.
followup-rank:
	uv run readout-recurrence sweep \
		--config configs/followups/rank_sweep.yaml \
		--grid model.readout_rank=8,16,32,64 \
		--seeds 0,1,2 \
		--output-dir runs/followup_rank

# H4: keep the total state width at 128 and vary the slot partition.
followup-slots:
	@for pair in "1 128" "2 64" "4 32" "8 16"; do \
		set -- $$pair; \
		echo "== slots L=$$1, d=$$2 =="; \
		uv run readout-recurrence sweep \
			--config configs/followups/a5_slot_ablation.yaml \
			--grid model.readout=attention,slot_gated,slot_max,gated_linear,linear \
			--seeds 0,1,2 \
			--match-params \
			--output-dir runs/followup_slots_L$$1 \
			--set model.n_slots=$$1 \
			--set model.slot_dim=$$2; \
	done

# Cascade optimization: compare random leakage initialisation with a
# block-diagonal start that must learn to open leakage.
followup-cascade:
	uv run readout-recurrence sweep \
		--config configs/followups/a5_cascade_init.yaml \
		--grid model.cascade_init=random,block_diagonal \
		--grid model.readout=linear,attention \
		--seeds 0,1,2 \
		--match-params \
		--output-dir runs/followup_cascade

# Hybrid optimization: gate bias and the attention branch's output gate.
followup-hybrid:
	uv run readout-recurrence sweep \
		--config configs/followups/mqar_hybrid.yaml \
		--grid model.hybrid_gate_bias_init=-2,0,2 \
		--grid model.attention_output_gate=true,false \
		--seeds 0,1,2 \
		--match-params \
		--output-dir runs/followup_hybrid
