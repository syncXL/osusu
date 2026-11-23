import io
import torch
import soundfile as sf
import librosa
from collections.abc import Callable
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets, Audio

# 1. Load each language/repo in streaming mode (0 GB disk)
# Using decode=False + soundfile avoids TorchCodec/FFmpeg issues

REPO_ID = "nolimitsxl/fongbe-challenge"
def load_stream(repo, config, split="train", columns: list[str] = ["audio", "text"]):
    ds = load_dataset(repo, config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.select_columns(columns)
    return ds

def load_stream_by_data_file(datafile: dict, split: str="train", columns: list[str] = ["audio", "text"]):
    ds = load_dataset("parquet", data_files=datafile, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.select_columns(columns)
    return ds

# 3. Wrap as a PyTorch IterableDataset for NeMo
class NeMoStreamingASRDataset(IterableDataset):
    def __init__(self, hf_iterable, tokenizer, max_sec=15.0):
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.max_sec = max_sec

    def __iter__(self):
        for ex in self.hf_iterable:
            audio_info = ex["audio"]
            raw_bytes = audio_info["bytes"]
            wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

            duration = len(wav) / 16000.0
            if duration < 0.3 or duration > self.max_sec:
                continue

            tokens = self.tokenizer.text_to_ids(ex["text"])
            if len(tokens) == 0:
                continue

            yield (
                torch.tensor(wav, dtype=torch.float32),
                len(wav),
                torch.tensor(tokens, dtype=torch.long),
                len(tokens),
            )

def nemo_asr_collate(batch):
    wavs, wav_lens, toks, tok_lens = zip(*batch)
    max_wav = max(wav_lens)
    max_tok = max(tok_lens)

    padded_wavs = torch.zeros(len(batch), max_wav, dtype=torch.float32)
    padded_toks = torch.zeros(len(batch), max_tok, dtype=torch.long)
    for i, (w, t) in enumerate(zip(wavs, toks)):
        padded_wavs[i, : len(w)] = w
        padded_toks[i, : len(t)] = t

    return (
        padded_wavs,
        torch.tensor(wav_lens, dtype=torch.long),
        padded_toks,
        torch.tensor(tok_lens, dtype=torch.long),
    )

def get_fongbe_loader(tokenizer, batch_size: int, max_sec: float, num_workers: int=0):
    train_conf = {"train" : f"hf://datasets/{REPO_ID}/data/corpus=zindi/split=train/language=fon/*.parquet"}
    valid_conf = {"validation" : f"hf://datasets/{REPO_ID}/data/corpus=zindi/split=validation/language=fon/*.parquet"}
    train_data = load_stream_by_data_file(datafile=train_conf, split="train")
    val_data = load_stream_by_data_file(valid_conf, split="validation")
    train_ds = NeMoStreamingASRDataset(train_data, tokenizer, max_sec=max_sec)
    val_ds = NeMoStreamingASRDataset(val_data, tokenizer, max_sec=max_sec)

    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=nemo_asr_collate, num_workers=num_workers)
    valid_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=nemo_asr_collate, num_workers=num_workers)
    return train_dl, valid_dl

# def get_pretrain_data(batch_size: int, max_sec: float, num_workers: int=0):

    
