# fongbe-asr Project Context & Guardrails

## Goal
Fine-tune NVIDIA `nvidia/parakeet-tdt-0.6b-v3` (617M params, FastConformer encoder + TDT decoder/joint, 8x subsampling, TDT durations `[0, 1, 2, 3, 4]`) for Fongbe ASR (`nolimitsxl/osusu`).

## Target Hardware & Hard Constraints (Kaggle 2x T4)
- **GPUs:** 2x NVIDIA T4 (`14.56 GiB` usable VRAM per GPU, DDP via NCCL, `16-mixed` AMP).
  - Static DDP + FP32 AdamW overhead is ~11–12.5 GiB per GPU, leaving only ~2–3.5 GiB for activations.
- **CPU RAM:** ~29 GiB total shared across both DDP ranks and the Jupyter kernel.
  - Keep `num_workers = 0`, `pin_memory = False`, and `shuffle_n = 256` to prevent DataLoader worker RAM multiplication.
  - Keep `HF_HUB_DISABLE_XET=1`, `MALLOC_ARENA_MAX=2`, and `OMP_NUM_THREADS=2` in `src/fongbe/__init__.py`.
- **Disk & Session Limits:**
  - `/kaggle/working` is capped at 19.5 GiB; checkpoints write to `/tmp/...` (`exp_dir`) and sync to Hugging Face Hub (`nolimitsxl/osusu-asr`).
  - Sessions time out at 12 hours, so epochs must finish and upload within a single session.
  - Keep `save_top_k = 1`, `save_last = True`, `always_save_nemo = False`, `save_nemo_on_train_end = True`.

## Dataset & Streaming Architecture
- **HF Dataset:** `nolimitsxl/osusu` (Parquet shards, `streaming=True` via NeMo `HFIterableAudioToBPEDataset`).
  - `zindi_fon` split has **133,627** training utterances, with audio durations up to **60 seconds**.
  - Audio is stored with `Audio(decode=False)` (`{'bytes': ..., 'path': ...}`) because `datasets >= 4.0` returns `TorchCodec` objects when `decode=True`.
  - **Do not remove** `_decode_bytes_aware_nested_get` in `src/fongbe/model.py`; it patches `nemo.collections.asr.data.huggingface.hf_audio_to_text.get_nested_dict_value` to decode `audio["bytes"]` lazily via `soundfile`.
  - Transcripts are already preprocessed (`syllables` column for Stage 2; `text` column for Stage 1).

## Training Stages
- **Stage 1 (Multilingual CPT):** `tokenizers/tokenizer_spe_unigram_v1024`, `text_key="text"`, global batch = 64, encoder LR `1e-4`, head LR `5e-4`.
- **Stage 2 (Fongbe FT):** `tokenizers/tokenizer_spe_unigram_v512`, `text_key="syllables"`, global batch = 32, uniform LR `4e-5`, 10 epochs (`4,176` steps/epoch, `41,760` total steps).

## Critical Invariants (Do Not Break)
1. **Callback order (`src/fongbe/train.py`):** `HFHubUploadCallback` MUST be appended *after* `exp_manager(trainer, config.exp_manager)` so `NeMoModelCheckpoint` writes checkpoints and frees ~7.4 GB RAM before upload runs.
2. **Synchronous, fault-tolerant uploads (`src/fongbe/callbacks/hf_upload.py`):** Upload on `on_train_epoch_end` (every `upload_every_n_epochs`) and `on_train_end`, after `gc.collect()`, wrapped in `try/except`, with `ignore_patterns=["*.tmp", "*-unfinished*"]` and `delete_patterns=["checkpoints/*.ckpt"]`.
3. **NeMo HF streaming requires integer `trainer.limit_train_batches`:** Set it to **1 epoch** of micro-batches (never `max_steps * accumulate_grad_batches`), so `current_epoch` increments and `on_train_epoch_end` fires before Kaggle's 12h timeout.
4. **Joint config timing (`src/fongbe/model.py`):** Copy `fused_batch_size` onto `asr_model.cfg.joint` *before* calling `asr_model.change_vocabulary(...)`.
5. **Encoder activation checkpointing (`src/fongbe/train.py`):** Keep `asr_model.encoder.layers` wrapped with `checkpoint_wrapper(..., checkpoint_impl=CheckpointImpl.NO_REENTRANT)`.
6. **Stream resumability & DDP epoch cycling (`src/fongbe/callbacks/stream_callback.py`, `src/fongbe/model.py`):** Training wraps `_train_dl` with `DynamicDurationBatchDataset(..., cycle=True)` so DDP ranks never hit `StopIteration` before `limit_train_batches`. Any custom wrapper around `_train_dl.dataset` must expose `state_dict()` and `load_state_dict()` delegating to the underlying HF `IterableDataset`. `HFStreamCheckpointCallback` must only stage `_pending_state` in `on_load_checkpoint` (never in `on_save_checkpoint`), so live multi-epoch runs do not reload end-of-epoch stream state on the next epoch.