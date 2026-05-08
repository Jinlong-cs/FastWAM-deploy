.PHONY: install download-release check-assets eager-bench trt-bench drift

PYTHON ?= python
PYTHONPATH := src

install:
	$(PYTHON) -m pip install -e .

download-release:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/download_fastwam_assets.py

check-assets:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_fastwam_assets.py

eager-bench:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_fastwam.py \
		--sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
		--text-cache-dir data/text_embeds_cache/robotwin \
		--num-inference-steps 1 \
		--warmup-batches 2 \
		--measure-batches 5 \
		--output outputs/benchmarks/robotwin_uncond_3cam_384_real_sample_text_cache_latency.json

trt-bench:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_trt_partitioned_runtime.py \
		--sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
		--text-cache-dir data/text_embeds_cache/robotwin \
		--num-inference-steps 1 \
		--warmup-batches 2 \
		--measure-batches 5 \
		--output outputs/trt/partitioned_trt_runtime_real_sample_text_cache.json

drift:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/compare_partitioned_trt_drift.py \
		--sample data/real_samples/robotwin_unified_ep000000_frame000000.npz \
		--text-cache-dir data/text_embeds_cache/robotwin \
		--num-inference-steps 1 \
		--output outputs/validation/partitioned_trt_drift_real_text_cache.json
