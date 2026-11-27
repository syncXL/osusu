from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf
from datasets import Audio, concatenate_datasets, interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node


@dataclass(frozen=True)
class CorpusSpec:
    config_name: str
    splits: tuple[str, ...]
    hours: float
    mean_duration_sec: float
    language: str


PRETRAIN_CORPORA: tuple[CorpusSpec, ...] = (
    # Fon corpora
    CorpusSpec(config_name="ffstc_fon", splits=("train",), hours=48.79, mean_duration_sec=5.95, language="fon"),
    CorpusSpec(config_name="fon_hau_fon", splits=("train",), hours=45.49, mean_duration_sec=24.19, language="fon"),
    CorpusSpec(
        config_name="fongbe-speech-zenodo_fon", splits=("train",), hours=5.73, mean_duration_sec=2.51, language="fon"
    ),
    CorpusSpec(config_name="zindi_fon", splits=("train",), hours=338.55, mean_duration_sec=9.12, language="fon"),
    # Ewe corpora
    CorpusSpec(
        config_name="waxalnlp_ewe", splits=("train", "unlabeled"), hours=1052.86, mean_duration_sec=19.05, language="ewe"
    ),
    CorpusSpec(
        config_name="open-bible-speech-african_ewe",
        splits=("train",),
        hours=91.17,
        mean_duration_sec=11.45,
        language="ewe",
    ),
    CorpusSpec(
        config_name="youversion-african-speech_ewe",
        splits=("train",),
        hours=4.24,
        mean_duration_sec=6.22,
        language="ewe",
    ),
)

DEFAULT_DATASET_ID = "nolimitsxl/osusu"
VAL_DATASET_CONFIG = "zindi_fon"
VAL_DATASET_SPLIT = "validation"


def compute_corpus_probabilities(
    corpora: Sequence[CorpusSpec] = PRETRAIN_CORPORA,
    beta_corpus: float = 0.5,
    duration_adjusted: bool = True,
    crop_duration_sec: float = 20.0,
) -> list[float]:
    """Compute OmniASR temperature-scaled corpus sampling probabilities.

    With ``duration_adjusted=True``, each temperature weight ``hours ** beta_corpus``
    is divided by ``min(mean_duration_sec, crop_duration_sec)`` before normalizing so
    the expected audio-seconds per batch match the temperature-scaled target distribution.
    """
    if not corpora:
        raise ValueError("corpora must not be empty.")
    if beta_corpus <= 0:
        raise ValueError(f"beta_corpus must be positive, got {beta_corpus}.")

    raw_weights: list[float] = []
    for corpus in corpora:
        weight = float(corpus.hours) ** float(beta_corpus)
        if duration_adjusted:
            effective_dur = min(float(corpus.mean_duration_sec), float(crop_duration_sec))
            if effective_dur <= 0:
                raise ValueError(f"Invalid effective duration for {corpus.config_name}: {effective_dur}")
            weight /= effective_dur
        raw_weights.append(weight)

    total = sum(raw_weights)
    if total <= 0:
        raise ValueError("Total corpus weight must be positive.")
    return [w / total for w in raw_weights]


def _resample_waveform(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return resample_poly(wav, up=up, down=down).astype(np.float32, copy=False)


def decode_audio_sample(
    sample: dict[str, Any],
    target_sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """Lazily decode a streaming HF audio item to mono float32 at ``target_sample_rate``.

    Only invokes channel downmixing when ``ndim > 1`` and resamples strictly when
    ``sr != target_sample_rate`` so native 16 kHz mono audio incurs zero resampling cost.
    """
    audio = sample.get("audio", sample)
    if not isinstance(audio, dict):
        raise TypeError(f"Expected dict audio payload, got {type(audio).__name__}")

    if "array" in audio and audio["array"] is not None:
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", target_sample_rate))
    else:
        raw_bytes = audio.get("bytes")
        source = io.BytesIO(raw_bytes) if raw_bytes is not None else audio.get("path")
        if source is None:
            raise KeyError("Audio entry must contain 'array', 'bytes', or 'path'.")
        wav, sr = sf.read(source, dtype="float32")
        wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32, copy=False)

    if int(sr) != int(target_sample_rate):
        wav = _resample_waveform(wav, orig_sr=int(sr), target_sr=int(target_sample_rate))
        sr = int(target_sample_rate)

    return wav, int(sr)


def _prepare_single_split(ds: Any) -> Any:
    if hasattr(ds, "column_names") and ds.column_names and "audio" in ds.column_names:
        extra_cols = [c for c in ds.column_names if c != "audio"]
        if extra_cols:
            ds = ds.remove_columns(extra_cols)
    if hasattr(ds, "cast_column"):
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def build_pretrain_streams(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    corpora: Sequence[CorpusSpec] = PRETRAIN_CORPORA,
    beta_corpus: float = 0.5,
    duration_adjusted: bool = True,
    crop_duration_sec: float = 20.0,
    seed: int = 42,
    shuffle_buffer_size: int = 256,
    rank: int = 0,
    world_size: int = 1,
    load_dataset_fn: Callable[..., Any] = load_dataset,
) -> tuple[Any, Any, list[float]]:
    """Build interleaved multi-corpus train stream and strict ``zindi_fon`` validation stream.

    ``zindi_fon`` validation and test splits are never permitted inside ``corpora``.
    """
    for corpus in corpora:
        if corpus.config_name == VAL_DATASET_CONFIG:
            for split in corpus.splits:
                if split in ("validation", "test", "val"):
                    raise ValueError(
                        f"Forbidden split '{split}' for '{VAL_DATASET_CONFIG}' in pretraining corpora."
                    )

    probs = compute_corpus_probabilities(
        corpora=corpora,
        beta_corpus=beta_corpus,
        duration_adjusted=duration_adjusted,
        crop_duration_sec=crop_duration_sec,
    )

    corpus_streams = []
    for corpus in corpora:
        split_datasets = []
        for split in corpus.splits:
            ds = load_dataset_fn(dataset_id, corpus.config_name, split=split, streaming=True)
            split_datasets.append(_prepare_single_split(ds))
        if len(split_datasets) == 1:
            corpus_streams.append(split_datasets[0])
        else:
            corpus_streams.append(concatenate_datasets(split_datasets))

    train_stream = interleave_datasets(
        corpus_streams,
        probabilities=probs,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    if shuffle_buffer_size and shuffle_buffer_size > 0 and hasattr(train_stream, "shuffle"):
        train_stream = train_stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    val_stream = load_dataset_fn(
        dataset_id,
        VAL_DATASET_CONFIG,
        split=VAL_DATASET_SPLIT,
        streaming=True,
    )
    val_stream = _prepare_single_split(val_stream)

    if world_size > 1:
        train_stream = split_dataset_by_node(train_stream, rank=rank, world_size=world_size)
        val_stream = split_dataset_by_node(val_stream, rank=rank, world_size=world_size)

    return train_stream, val_stream, probs
