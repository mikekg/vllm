# Marlin NVFP4-to-FP8 validation

This directory is an index for the converter's canonical tests and external
quality campaign. It deliberately contains no copied tests, logs, models, or
binaries.

## Canonical tests

Run from the repository root with the project virtual environment:

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_marlin_nvfp4_to_fp8.py -v
.venv/bin/python -m pytest tests/quantization/test_modelopt.py -k nvfp4_bycopy -v
.venv/bin/python -m pytest tests/kernels/core/test_fused_silu_mul_block_quant.py -v
.venv/bin/python -m pytest tests/kernels/moe/test_nvfp4_bycopy_moe.py -v
```

The converter suite covers independent Marlin decode oracles, dense and MoE
round trips, K64/K192 padded scratch, zero and maximum tiles, graph replay, and
alias rejection. The other suites cover selection, bias, fused SiLU tail, and
staged MoE workspace contracts. Native converter cases require SM89 or SM90;
the final evidence below was collected on H100.

## H100 evidence

Scheduler job IDs identify external evidence; logs and artifacts are not
vendored here.

| Job | Result |
| --- | --- |
| 6574406 | Stable build passed; ABI SHA-256 `bb42106a78fc56e22f2ae648a566b013474497734974d8dd6591d61b4e6e2efc`. |
| 6574407 | Native K64/K192 payload, zero-tail, scale, and canary checks passed. |
| 6574663 | Fused-tail FP16/BF16 exact byte and scale checks passed. |
| 6574664 | Stable bootstrap and exact overlay-manifest checks passed. |
| 6574754 | Standard production-path gate passed. |
| 6574758 | Compiler FP16/BF16 gate passed. |
| 6574782 | SiLU K192/N192 stream, graph, arena, and zero-tail checks passed; RRMS 0.04678. |
| 6574781 | GELU N64 stream, graph, arena, and zero-tail checks passed; RRMS 0.03897. |
| 6574786-6574789 | Final four-model universal-512 smokes were pending when this index was written. |

Historical repaired-PR controls were job 6573903 (128 persistent conversions
plus graph serving), job 6574026 (FP8 GSM8K: 899/1319), and job 6574027
(paired Marlin GSM8K: 917/1319). They are controls, not substitutes for the
final campaign.

## Quality campaign

The full campaign is 24 end-to-end GSM8K runs over four pinned checkpoints:

- 20 factor runs: `marlin`, `adaptive`, `r1`, `sqrt6`, and `r6` for each model.
- 4 production-normal runs: `adaptive_prod`, one for each model.

Every FP8 variant uses the universal knee of 512 with no shape override;
LM-head and embedding exclusion is type-based. Each run must use a fresh cache
root keyed by source tag, run ID, model, and variant.

The site harness remains external. Configure it through the variables recorded
in `manifest.template.json`, expand that template into the run manifest, and
launch its smoke mode first. Full mode must stay fail-closed until the final
smokes pass and `S39_FULL_CAMPAIGN_GO=1` is explicitly present. Preserve every
per-item prompt hash, label, prediction, correctness flag, invalid flag, token
count, response hash, and response so comparisons can be recounted independently.
