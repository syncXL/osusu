import os

# Allow importing NeMo on CPU-only environments without nvidia cuda-python bindings
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "0")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

import fongbe
from fongbe.callbacks.stream_callback import HFStreamCheckpointCallback
from fongbe.model import DynamicDurationBatchDataset, setup_model_for_training
from fongbe.recipes.fine_tune_fongbe import load_fongbe_cfg


SAMPLE_RATE = 16000


class FakeHFStream(IterableDataset):
    def __init__(self, durations_sec: list[float]):
        super().__init__()
        self.durations_sec = list(durations_sec)
        self._cursor = 0
        self.saved_states: list[dict] = []
        self.loaded_states: list[dict] = []

    def __iter__(self):
        for idx in range(self._cursor, len(self.durations_sec)):
            dur = self.durations_sec[idx]
            n_samples = int(round(dur * SAMPLE_RATE))
            self._cursor = idx + 1
            yield {
                "audio_signal": torch.zeros(n_samples, dtype=torch.float32),
                "a_sig_length": torch.tensor(n_samples, dtype=torch.long),
                "transcripts": torch.tensor([1, 2, 3], dtype=torch.long),
                "transcript_length": torch.tensor(3, dtype=torch.long),
                "utterance_id": idx,
            }

    def state_dict(self):
        state = {"cursor": self._cursor}
        self.saved_states.append(state)
        return state

    def load_state_dict(self, state_dict):
        self.loaded_states.append(dict(state_dict))
        self._cursor = state_dict["cursor"]


class FakeNeMoHFIterableWrapper(IterableDataset):
    """Mimics NeMo's HFIterableAudioToBPEDataset where .dataset holds the HF IterableDataset."""

    def __init__(self, hf_stream: FakeHFStream):
        super().__init__()
        self.dataset = hf_stream
        self.sample_rate = SAMPLE_RATE

    def __iter__(self):
        return iter(self.dataset)

    def collate_fn(self, batch):
        max_len = max(int(item["a_sig_length"]) for item in batch)
        signals = torch.stack(
            [
                torch.nn.functional.pad(item["audio_signal"], (0, max_len - len(item["audio_signal"])))
                for item in batch
            ]
        )
        lengths = torch.stack([item["a_sig_length"] for item in batch])
        tokens = torch.stack([item["transcripts"] for item in batch])
        tok_lengths = torch.stack([item["transcript_length"] for item in batch])
        return signals, lengths, tokens, tok_lengths


def test_dynamic_batching_padded_duration_and_bs1_for_long_clips():
    durations = [
        2.0, 5.0, 10.0, 12.0, 28.0, 29.0, 30.0,
        35.0, 42.5, 50.0, 60.0,
        3.0, 4.0, 25.0, 29.5, 31.0,
    ]
    stream = FakeHFStream(durations)
    wrapper = FakeNeMoHFIterableWrapper(stream)
    ds = DynamicDurationBatchDataset(
        dataset=wrapper,
        max_batch_duration=60.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        sort_buffer_size=32,
        collate_fn=wrapper.collate_fn,
    )
    dl = DataLoader(ds, batch_size=None, num_workers=0)

    batches = list(dl)
    assert len(batches) > 0

    total_items = 0
    long_batch_sizes = []
    for signals, lengths, tokens, tok_lengths in batches:
        bs = signals.shape[0]
        total_items += bs
        max_dur = lengths.max().item() / SAMPLE_RATE
        padded_dur = max_dur * bs

        assert 1 <= bs <= 2
        assert padded_dur <= 60.0 + 1e-6
        assert tokens.shape[0] == bs
        assert tok_lengths.shape[0] == bs

        if (lengths >= 35 * SAMPLE_RATE).any():
            long_batch_sizes.append(bs)

    assert total_items == len(durations)
    assert long_batch_sizes == [1, 1, 1, 1]


def test_dynamic_batching_sorts_within_buffer_and_bounds_padded_duration():
    # Alternating very short and long clips inside one 32-item sort buffer
    durations = [2.0, 55.0] * 16
    stream = FakeHFStream(durations)
    ds = DynamicDurationBatchDataset(
        dataset=stream,
        max_batch_duration=60.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        sort_buffer_size=32,
        collate_fn=None,
    )

    batches = list(ds)
    # 16 short (2.0s) clips pair into 8 batches of size 2; 16 long (55.0s) clips emit as 16 batches of size 1
    bs_two = [b for b in batches if len(b) == 2]
    bs_one = [b for b in batches if len(b) == 1]
    assert len(bs_two) == 8
    assert len(bs_one) == 16
    for batch in batches:
        max_dur = max(int(item["a_sig_length"]) / SAMPLE_RATE for item in batch)
        assert max_dur * len(batch) <= 60.0 + 1e-6


def test_state_dict_delegation_and_stream_callback():
    stream = FakeHFStream([4.0, 6.0, 40.0, 8.0, 10.0])
    wrapper = FakeNeMoHFIterableWrapper(stream)
    ds = DynamicDurationBatchDataset(
        dataset=wrapper,
        max_batch_duration=60.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        sort_buffer_size=2,
        collate_fn=wrapper.collate_fn,
    )
    dl = DataLoader(ds, batch_size=None, num_workers=0)

    it = iter(dl)
    next(it)  # consumes first 2-item buffer -> cursor == 2

    state = ds.state_dict()
    assert state == {"cursor": 2}
    assert stream.saved_states == [{"cursor": 2}]

    ds.load_state_dict({"cursor": 3})
    assert stream.loaded_states[-1] == {"cursor": 3}

    # Verify HFStreamCheckpointCallback interacts cleanly with DataLoader(DynamicDurationBatchDataset)
    cb = HFStreamCheckpointCallback()
    fake_trainer = SimpleNamespace(train_dataloader=dl)
    ckpt = {}
    cb.on_save_checkpoint(fake_trainer, None, ckpt)
    assert ckpt["hf_stream_state"] == {"cursor": 3}

    cb.load_state_dict({"hf_stream_state": {"cursor": 1}})
    cb.on_train_epoch_start(fake_trainer, None)
    assert stream.loaded_states[-1] == {"cursor": 1}


def test_recipe_config_and_env_vars():
    assert os.environ.get("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS") == "0"
    assert hasattr(fongbe, "__file__")

    cfg_s2 = load_fongbe_cfg(stage="stage2", smoke_test=False)
    assert cfg_s2.model.train_ds.batch_size == 2
    assert cfg_s2.model.validation_ds.batch_size == 2
    assert cfg_s2.model.train_ds.max_duration == 60.0
    assert cfg_s2.model.validation_ds.max_duration == 60.0
    assert cfg_s2.model.train_ds.batch_duration == 60.0
    assert cfg_s2.model.joint.fused_batch_size == 1
    assert cfg_s2.model.joint.fuse_loss_wer is True
    assert cfg_s2.trainer.accumulate_grad_batches == 8
    assert cfg_s2.trainer.max_steps == 41760
    assert cfg_s2.trainer.limit_train_batches == 4176 * 8

    cfg_s1 = load_fongbe_cfg(stage="stage1", smoke_test=False)
    assert cfg_s1.model.train_ds.batch_size == 2
    assert cfg_s1.model.validation_ds.batch_size == 2
    assert cfg_s1.trainer.accumulate_grad_batches == 16
    assert cfg_s1.trainer.limit_train_batches == 4176 * 16

    cfg_smoke = load_fongbe_cfg(stage="stage2", smoke_test=True)
    assert cfg_smoke.trainer.max_epochs == 1
    assert cfg_smoke.trainer.max_steps == 20
    assert cfg_smoke.trainer.limit_train_batches == 20 * 8


def test_setup_model_for_training_joint_timing_chunking_and_dl_wrap():
    cfg = load_fongbe_cfg(stage="stage2", smoke_test=True)
    call_order = []

    stream = FakeHFStream([5.0, 45.0])
    wrapper = FakeNeMoHFIterableWrapper(stream)

    mock_model = MagicMock()
    mock_model.cfg = OmegaConf.create({
        "joint": {"fuse_loss_wer": True, "fused_batch_size": 4},
        "decoding": {"fused_batch_size": None},
    })
    mock_model.joint = MagicMock()
    mock_model.joint._cfg = OmegaConf.create({"fuse_loss_wer": True, "fused_batch_size": 4})
    mock_model.tokenizer.vocab_size = 1024

    def fake_change_vocab(**kwargs):
        call_order.append(("change_vocabulary", mock_model.cfg.joint.fused_batch_size, mock_model.joint._cfg.fused_batch_size))

    def fake_setup_train_dl(train_cfg):
        mock_model._train_dl = DataLoader(wrapper, batch_size=train_cfg.batch_size, collate_fn=wrapper.collate_fn)

    mock_model.change_vocabulary.side_effect = fake_change_vocab
    mock_model.setup_training_data.side_effect = fake_setup_train_dl

    with patch("fongbe.model.ASRModel.from_pretrained", return_value=mock_model):
        result = setup_model_for_training(cfg)

    assert result is mock_model
    assert call_order == [("change_vocabulary", 1, 1)]
    mock_model.change_subsampling_conv_chunking_factor.assert_called_once_with(1)
    assert isinstance(mock_model._train_dl, DataLoader)
    assert isinstance(mock_model._train_dl.dataset, DynamicDurationBatchDataset)
    assert mock_model._train_dl.batch_size is None
    assert mock_model._train_dl.num_workers == 0
