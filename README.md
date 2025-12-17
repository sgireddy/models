# models

MedGemma TFLite conversion utilities and environment setup.

This repo contains a conversion script to export `google/medgemma-4b-it` to a single `.tflite` with:
- 16k KV cache (`kv_cache_max_len=16384`)
- Prefill sequence length 1024 (`prefill_seq_len=1024`)
- Weight-only int8 quantization (`quantize="dynamic_int8"`)

## Files
- `scripts/convert_medgemma.py` — conversion script
- `scripts/test.py` — validation script to verify AI Edge imports on macOS
- `requirements.txt` — pinned package versions

## Recommended environments
- Intel macOS (x86_64) with 32 GB RAM or more
- Linux (Ubuntu) — most reliable for AI Edge wheels

Apple Silicon macOS (ARM64) notes:
- Some `ai_edge_litert` ARM64 wheels miss `libpywrap_litert_common.dylib`, causing import failures.
- If you must use Apple Silicon, consider Rosetta/x86_64 venv as a workaround (GPU acceleration will be unavailable).

## Setup (Intel macOS or Linux)
1) Install Python 3.10
- macOS (Homebrew): `brew install python@3.10`
- Linux: use your package manager or Python installer

2) Create and activate a venv
- macOS/Linux:
  - `python3.10 -m venv .venv310`
  - `source .venv310/bin/activate`

3) Install dependencies
- `pip install --upgrade pip`
- `pip install --no-cache-dir -r requirements.txt`

4) Validate environment
- `python scripts/test.py`
- Expected output includes:
  - `ai_edge_torch.generative.utilities imports: OK`
  - `ai_edge_litert interpreter import: OK`

5) Run conversion (do not run if only validating)
- `python scripts/convert_medgemma.py`
- Output is written to `./medgemma_multimodal_16k/medgemma_16k_int8.tflite`

## Rosetta x86_64 venv (Apple Silicon workaround)
If ARM64 wheels fail to provide `libpywrap_litert_common.dylib`:
- Install Rosetta: `softwareupdate --install-rosetta --agree-to-license`
- Install Intel Homebrew in `/usr/local`
- Create x86_64 venv:
  - `arch -x86_64 /usr/local/bin/python3.10 -m venv .venv310_x86`
  - `source .venv310_x86/bin/activate`
- Install deps:
  - `pip install --upgrade pip`
  - `pip install --no-cache-dir -r requirements.txt`
- Validate: `python scripts/test.py`
- Run conversion: `python scripts/convert_medgemma.py`

## Notes
- Ensure at least 25–40 GB of free disk space for caches and outputs.
- The conversion is mostly CPU-bound; expect tens of minutes on laptops.
- `.gitignore` excludes large artifacts and conversion outputs.
