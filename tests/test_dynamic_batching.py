import itertools
import logging
import os

# Allow importing NeMo on CPU-only environments without nvidia cuda-python bindings
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "0")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

import fongbe
from fongbe.callbacks.stream_callback import HFStreamCheckpointCallback
from fongbe.model import DynamicDurationBatchDataset, setup_differential_lr, setup_model_for_training
from fongbe.recipes.fine_tune_fongbe import load_fongbe_cfg
from fongbe.train import resolve_head_lr, train_fongbe


SAMPLE_RATE = 16000


class FakeHFStream(IterableDataset):
    def __init__(self, durations_sec: list[float]):
        super().__init__()
        self.durations_sec = list(durations_sec)
        self._cursor = 0
        self._starting_state_dict = None
        self.epoch = 0
        self.epochs_seen: list[int] = []
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

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.epochs_seen.append(epoch)
        self._cursor = 0

    def state_dict(self):
        state = {"cursor": self._cursor, "epoch": self.epoch}
        self.saved_states.append(state)
        return state

    def load_state_dict(self, state_dict):
        self.loaded_states.append(dict(state_dict))
        self._starting_state_dict = dict(state_dict)
        self._cursor = state_dict["cursor"]
        self.epoch = state_dict.get("epoch", 0)


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
    assert state == {"pass": 0, "hf_state": {"cursor": 2, "epoch": 0}}
    assert stream.saved_states == [{"cursor": 2, "epoch": 0}]

    # New state dict format restores _pass and underlying HF stream state
    ds.load_state_dict({"pass": 2, "hf_state": {"cursor": 3, "epoch": 2}})
    assert ds._pass == 2
    assert stream.loaded_states[-1] == {"cursor": 3, "epoch": 2}

    # Backwards-compatible legacy raw HF state dict (no "hf_state" wrapper)
    ds.load_state_dict({"cursor": 1, "epoch": 0})
    assert ds._pass == 0
    assert stream.loaded_states[-1] == {"cursor": 1, "epoch": 0}

    # Live multi-epoch save must NOT stage _pending_state or reload on next epoch start
    cb = HFStreamCheckpointCallback()
    fake_trainer = SimpleNamespace(train_dataloader=dl)
    ckpt = {}
    loaded_before = len(stream.loaded_states)
    cb.on_save_checkpoint(fake_trainer, None, ckpt)
    assert ckpt["hf_stream_state"] == {"pass": 0, "hf_state": {"cursor": 1, "epoch": 0}}
    assert cb._pending_state is None
    assert cb.state_dict() == {}

    cb.on_train_epoch_start(fake_trainer, None)
    assert len(stream.loaded_states) == loaded_before

    # Resuming via on_load_checkpoint stages _pending_state and restores once on epoch start
    cb.on_load_checkpoint(fake_trainer, None, {"hf_stream_state": {"pass": 1, "hf_state": {"cursor": 4, "epoch": 1}}})
    assert cb._pending_state is not None
    cb.on_train_epoch_start(fake_trainer, None)
    assert ds._pass == 1
    assert stream.loaded_states[-1] == {"cursor": 4, "epoch": 1}
    assert cb._pending_state is None

    # Subsequent epoch start must not re-apply old state
    cb.on_train_epoch_start(fake_trainer, None)
    assert len(stream.loaded_states) == loaded_before + 1


def test_dynamic_batching_cycle_and_empty_exit():
    # Stream with 3 short items (2 batches per pass with bs=2, sort_buffer_size=32)
    stream = FakeHFStream([2.0, 3.0, 4.0])
    stream._starting_state_dict = {"cursor": 0, "epoch": 0}
    wrapper = FakeNeMoHFIterableWrapper(stream)
    ds = DynamicDurationBatchDataset(
        dataset=wrapper,
        max_batch_duration=60.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        sort_buffer_size=32,
        collate_fn=wrapper.collate_fn,
        cycle=True,
    )
    dl = DataLoader(ds, batch_size=None, num_workers=0)

    # Pull 5 batches (spans pass 0, pass 1, and into pass 2)
    batches = list(itertools.islice(dl, 5))
    assert len(batches) == 5
    assert ds._pass == 2
    assert stream.epochs_seen == [1, 2]
    assert stream._starting_state_dict is None

    # When all samples are filtered out, cycle=True must terminate cleanly instead of hanging
    filtered_stream = FakeHFStream([0.1, 0.2])
    filtered_ds = DynamicDurationBatchDataset(
        dataset=filtered_stream,
        max_batch_duration=60.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        min_duration=0.5,
        cycle=True,
    )
    assert list(filtered_ds) == []

    # Empty underlying stream with cycle=True also exits cleanly
    empty_ds = DynamicDurationBatchDataset(
        dataset=FakeHFStream([]),
        max_batch_duration=60.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        cycle=True,
    )
    assert list(empty_ds) == []


def test_recipe_config_and_env_vars():
    assert os.environ.get("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS") == "0"
    assert hasattr(fongbe, "__file__")

    cfg_s2 = load_fongbe_cfg(stage="stage2", smoke_test=False)
    assert cfg_s2.model.train_ds.batch_size == 2
    assert cfg_s2.model.validation_ds.batch_size == 2
    assert cfg_s2.model.train_ds.max_duration == 60.0
    assert cfg_s2.model.validation_ds.max_duration == 60.0
    assert cfg_s2.model.train_ds.batch_duration == 48.0
    assert cfg_s2.model.joint.fused_batch_size == 1
    assert cfg_s2.model.joint.fuse_loss_wer is True
    assert cfg_s2.trainer.accumulate_grad_batches == 8
    assert cfg_s2.trainer.max_steps == 41760
    assert cfg_s2.trainer.limit_train_batches == 4176 * 8
    assert cfg_s2.trainer.val_check_interval == 8352
    assert cfg_s2.trainer.limit_train_batches % cfg_s2.trainer.val_check_interval == 0
    assert cfg_s2.trainer.gradient_clip_val == 1.0
    assert cfg_s2.model.optim.sched.warmup_steps == 500
    assert cfg_s2.model.optim.sched.warmup_ratio is None

    cfg_s1 = load_fongbe_cfg(stage="stage1", smoke_test=False)
    assert cfg_s1.model.train_ds.batch_size == 2
    assert cfg_s1.model.validation_ds.batch_size == 2
    assert cfg_s1.trainer.accumulate_grad_batches == 16
    assert cfg_s1.trainer.limit_train_batches == 4176 * 16
    assert cfg_s1.trainer.gradient_clip_val == 1.0
    assert cfg_s1.model.optim.sched.warmup_steps == 500
    assert cfg_s1.model.optim.sched.warmup_ratio is None

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
        "loss": {"loss_name": "tdt", "tdt_kwargs": {"durations": [0, 1, 2, 3, 4]}},
        "rnnt_reduction": "mean_batch",
    })
    mock_model.joint = MagicMock()
    mock_model.joint._cfg = OmegaConf.create({"fuse_loss_wer": True, "fused_batch_size": 4})
    mock_model.joint.num_classes_with_blank = 518
    mock_model.joint.num_extra_outputs = 5
    mock_model.joint.fuse_loss_wer = True
    mock_model.tokenizer.vocab_size = 512
    mock_model.extract_rnnt_loss_cfg.return_value = ("tdt", {"durations": [0, 1, 2, 3, 4]})

    class _FakeTDTInnerLoss:
        pass

    _FakeTDTInnerLoss.__name__ = "TDTLoss"
    fake_loss_instance = MagicMock(_loss=_FakeTDTInnerLoss())

    def fake_change_vocab(**kwargs):
        call_order.append(("change_vocabulary", mock_model.cfg.joint.fused_batch_size, mock_model.joint._cfg.fused_batch_size))

    def fake_setup_train_dl(train_cfg):
        mock_model._train_dl = DataLoader(wrapper, batch_size=train_cfg.batch_size, collate_fn=wrapper.collate_fn)

    mock_model.change_vocabulary.side_effect = fake_change_vocab
    mock_model.setup_training_data.side_effect = fake_setup_train_dl

    with (
        patch("fongbe.model.ASRModel.from_pretrained", return_value=mock_model),
        patch("fongbe.model.RNNTLoss", return_value=fake_loss_instance) as mock_rnnt_loss,
    ):
        result = setup_model_for_training(cfg)

    assert result is mock_model
    assert call_order == [("change_vocabulary", 1, 1)]
    mock_rnnt_loss.assert_called_once_with(
        num_classes=512,
        loss_name="tdt",
        loss_kwargs={"durations": [0, 1, 2, 3, 4]},
        reduction="mean_batch",
    )
    mock_model.change_decoding_strategy.assert_called_once()
    mock_model.joint.set_loss.assert_called_once_with(fake_loss_instance)
    mock_model.joint.set_wer.assert_called_once_with(mock_model.wer)
    mock_model.change_subsampling_conv_chunking_factor.assert_called_once_with(1)
    assert isinstance(mock_model._train_dl, DataLoader)
    assert isinstance(mock_model._train_dl.dataset, DynamicDurationBatchDataset)
    assert mock_model._train_dl.dataset.cycle is True
    assert mock_model._train_dl.batch_size is None
    assert mock_model._train_dl.num_workers == 0


def test_restore_tdt_loss_and_decoding_validations(caplog):
    from fongbe.model import _restore_tdt_loss_and_decoding

    durations = [0, 1, 2, 3, 4]
    loss_kwargs = {"fastemit_lambda": 0.0, "clamp": -1.0, "durations": durations}

    class _FakeTDTLoss:
        pass

    _FakeTDTLoss.__name__ = "TDTLoss"

    def _make_model(vocab_size=512, num_classes_with_blank=518, num_extra=5, fuse_loss_wer=True):
        m = MagicMock()
        m.cfg = OmegaConf.create({
            "loss": {"loss_name": "tdt", "tdt_kwargs": loss_kwargs},
            "decoding": {"strategy": "greedy_batch", "durations": durations},
            "rnnt_reduction": "mean_batch",
        })
        m.tokenizer.vocab_size = vocab_size
        m.joint.num_classes_with_blank = num_classes_with_blank
        m.joint.num_extra_outputs = num_extra
        m.joint.fuse_loss_wer = fuse_loss_wer
        m.extract_rnnt_loss_cfg.return_value = ("tdt", loss_kwargs)
        return m

    # 1. Valid TDT restore with fuse_loss_wer=True
    model = _make_model()
    fake_loss = MagicMock(_loss=_FakeTDTLoss())
    with patch("fongbe.model.RNNTLoss", return_value=fake_loss) as mock_loss_cls:
        _restore_tdt_loss_and_decoding(model)

    mock_loss_cls.assert_called_once_with(
        num_classes=512,
        loss_name="tdt",
        loss_kwargs=loss_kwargs,
        reduction="mean_batch",
    )
    model.change_decoding_strategy.assert_called_once_with({"strategy": "greedy_batch", "durations": durations}, verbose=False)
    model.joint.set_loss.assert_called_once_with(fake_loss)
    model.joint.set_wer.assert_called_once_with(model.wer)

    # 2. fuse_loss_wer=False does not re-wire joint.set_loss / set_wer
    model_no_fuse = _make_model(fuse_loss_wer=False)
    with patch("fongbe.model.RNNTLoss", return_value=fake_loss):
        _restore_tdt_loss_and_decoding(model_no_fuse)
    model_no_fuse.joint.set_loss.assert_not_called()
    model_no_fuse.joint.set_wer.assert_not_called()

    # 3. Real RNNTLoss instantiation builds TDTLoss cleanly without falling back to plain RNN-T
    model_real_loss = _make_model(vocab_size=512, num_classes_with_blank=518, num_extra=5)
    with caplog.at_level(logging.INFO, logger="nemo_logger"):
        _restore_tdt_loss_and_decoding(model_real_loss)
    assert "tdt" in type(model_real_loss.loss._loss).__name__.lower()
    assert "TDT loss restored: num_classes=512, num_extra_outputs=5, durations=[0, 1, 2, 3, 4]" in caplog.text

    # 4. Fails loudly if num_classes != vocab_size (e.g. blank+durations off-by-N)
    model_bad_vocab = _make_model(vocab_size=512, num_classes_with_blank=518, num_extra=0)
    with patch("fongbe.model.RNNTLoss", return_value=fake_loss):
        with pytest.raises(AssertionError, match="TDT num_classes mismatch"):
            _restore_tdt_loss_and_decoding(model_bad_vocab)

    # 5. Fails loudly if plain RNNTLoss slips in instead of TDT
    class _PlainRNNTLoss:
        pass

    _PlainRNNTLoss.__name__ = "RNNTLossNumba"
    model_bad_impl = _make_model()
    with patch("fongbe.model.RNNTLoss", return_value=MagicMock(_loss=_PlainRNNTLoss())):
        with pytest.raises(AssertionError, match="Expected TDT loss implementation"):
            _restore_tdt_loss_and_decoding(model_bad_impl)


def test_setup_differential_lr_persists_across_setup_optimizer_param_groups():
    class FakeASRModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(8, 8)
            self.frozen_enc = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
            self.encoder.register_parameter("frozen", self.frozen_enc)
            self.decoder = torch.nn.Linear(8, 4)
            self.joint = torch.nn.Linear(4, 5)
            self._optimizer_param_groups = None

        def setup_optimizer_param_groups(self):
            # Simulates NeMo ModelPT.setup_optimizer_param_groups wiping custom groups
            self._optimizer_param_groups = [{"params": list(self.parameters())}]

    model = FakeASRModule()
    setup_differential_lr(model, encoder_lr=1e-4, head_lr=5e-4)

    assert len(model._optimizer_param_groups) == 2
    assert model._optimizer_param_groups[0]["lr"] == 1e-4
    assert model._optimizer_param_groups[1]["lr"] == 5e-4
    assert all(p.requires_grad for g in model._optimizer_param_groups for p in g["params"])
    assert len(model._optimizer_param_groups[0]["params"]) == 2  # encoder weight + bias only
    assert len(model._optimizer_param_groups[1]["params"]) == 4  # decoder + joint weights/biases

    # Simulate NeMo setup_optimization() / Lightning configure_optimizers() re-calling setup_optimizer_param_groups()
    model._optimizer_param_groups = None
    model.setup_optimizer_param_groups()
    assert len(model._optimizer_param_groups) == 2
    assert model._optimizer_param_groups[0]["lr"] == 1e-4
    assert model._optimizer_param_groups[1]["lr"] == 5e-4


def test_train_fongbe_lr_and_warmup_overrides(tmp_path):
    assert resolve_head_lr("stage1", encoder_lr=1e-4, head_lr=None) == 5e-4
    assert resolve_head_lr("stage2", encoder_lr=4e-5, head_lr=None) is None
    assert resolve_head_lr("stage1", encoder_lr=1e-4, head_lr=1e-4) is None
    assert resolve_head_lr("stage2", encoder_lr=1e-4, head_lr=3e-4) == 3e-4

    captured = []

    def fake_setup(config, trainer=None, head_lr=None):
        captured.append((config, head_lr))
        m = MagicMock()
        m.encoder.layers = []
        return m

    with (
        patch("fongbe.train.Trainer", return_value=MagicMock(callbacks=[])),
        patch("fongbe.train.exp_manager"),
        patch("fongbe.train.setup_model_for_training", side_effect=fake_setup),
    ):
        train_fongbe(
            model_name="dummy/model",
            exp_dir=str(tmp_path / "exp1"),
            repo_id="dummy/repo",
            stage="stage2",
            encoder_lr=1.5e-4,
            head_lr=4.5e-4,
            warmup_steps=250,
        )
        train_fongbe(
            model_name="dummy/model",
            exp_dir=str(tmp_path / "exp2"),
            repo_id="dummy/repo",
            stage="stage1",
            encoder_lr=2e-4,
            head_lr=2e-4,
            warmup_steps=300,
        )

    cfg1, h1 = captured[0]
    assert cfg1.model.optim.lr == 1.5e-4
    assert cfg1.model.optim.sched.warmup_steps == 250
    assert cfg1.model.optim.sched.warmup_ratio is None
    assert h1 == 4.5e-4

    cfg2, h2 = captured[1]
    assert cfg2.model.optim.lr == 2e-4
    assert cfg2.model.optim.sched.warmup_steps == 300
    assert cfg2.model.optim.sched.warmup_ratio is None
    assert h2 is None

    # Also verify setup_model_for_training skips setup_differential_lr when head_lr == optim.lr
    cfg_eq = load_fongbe_cfg(stage="stage2", smoke_test=True, encoder_lr=1e-4)
    mock_model = MagicMock()
    mock_model.tokenizer.vocab_size = 1024
    with (
        patch("fongbe.model.ASRModel.from_pretrained", return_value=mock_model),
        patch("fongbe.model._restore_tdt_loss_and_decoding"),
        patch("fongbe.model.setup_differential_lr") as mock_diff,
    ):
        setup_model_for_training(cfg_eq, head_lr=1e-4)
        setup_model_for_training(cfg_eq, head_lr=None)
        mock_diff.assert_not_called()
        setup_model_for_training(cfg_eq, head_lr=3e-4)
        mock_diff.assert_called_once_with(mock_model, encoder_lr=1e-4, head_lr=3e-4)


def test_t4_headroom_fused_batch_size_55s_clip_and_ddp_timeout(tmp_path):
    import datetime
    from fongbe.model import _restore_tdt_loss_and_decoding

    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"

    # 1. Read-only property joint mimicking real NeMo RNNTJoint enforces fused_batch_size == 1
    class _ReadOnlyJoint:
        def __init__(self):
            self._fused_batch_size = 8
            self._cfg = OmegaConf.create({"fused_batch_size": 8, "fuse_loss_wer": True})
            self.num_classes_with_blank = 518
            self.num_extra_outputs = 5
            self.fuse_loss_wer = True
            self.set_loss = MagicMock()
            self.set_wer = MagicMock()

        @property
        def fused_batch_size(self):
            return self._fused_batch_size

        def set_fused_batch_size(self, value):
            self._fused_batch_size = value

    class _FakeTDTLoss:
        pass

    _FakeTDTLoss.__name__ = "TDTLoss"

    model = MagicMock()
    model.cfg = OmegaConf.create({
        "joint": {"fused_batch_size": 8, "fuse_loss_wer": True},
        "decoding": {"fused_batch_size": 8},
        "loss": {"loss_name": "tdt", "tdt_kwargs": {"durations": [0, 1, 2, 3, 4]}},
        "rnnt_reduction": "mean_batch",
    })
    model.joint = _ReadOnlyJoint()
    model.tokenizer.vocab_size = 512
    model.extract_rnnt_loss_cfg.return_value = ("tdt", {"durations": [0, 1, 2, 3, 4]})

    with patch("fongbe.model.RNNTLoss", return_value=MagicMock(_loss=_FakeTDTLoss())):
        _restore_tdt_loss_and_decoding(model)

    assert model.joint.fused_batch_size == 1
    assert model.cfg.joint.fused_batch_size == 1
    assert model.cfg.decoding.fused_batch_size == 1

    # 2. max_batch_duration=48.0 with max_duration=60.0 still keeps & yields a single 55s utterance
    stream_55 = FakeHFStream([55.0, 10.0])
    ds = DynamicDurationBatchDataset(
        dataset=stream_55,
        max_batch_duration=48.0,
        max_batch_size=2,
        sample_rate=SAMPLE_RATE,
        sort_buffer_size=32,
        min_duration=0.3,
        max_duration=60.0,
        cycle=False,
    )
    batches = list(ds)
    durs = [[x["a_sig_length"].item() / SAMPLE_RATE for x in b] for b in batches]
    assert durs == [[10.0], [55.0]]

    # 3. train_fongbe configures DDPStrategy timeout=5min and passes upload_on_val to callback
    fake_trainer = MagicMock(callbacks=[])
    fake_model = MagicMock()
    fake_model.encoder.layers = []
    with (
        patch("fongbe.train.hydra.utils.instantiate", return_value="mock_strategy") as mock_inst,
        patch("fongbe.train.Trainer", return_value=fake_trainer),
        patch("fongbe.train.exp_manager"),
        patch("fongbe.train.setup_model_for_training", return_value=fake_model),
    ):
        train_fongbe(
            model_name="dummy/model",
            exp_dir=str(tmp_path / "exp"),
            repo_id="dummy/repo",
            stage="stage2",
            upload_on_val=True,
        )

    assert mock_inst.call_args.kwargs["timeout"] == datetime.timedelta(minutes=5)
    upload_cb = fake_trainer.callbacks[-1]
    assert upload_cb.upload_on_val is True


def test_hf_upload_callback_on_validation_end(tmp_path):
    from fongbe.callbacks.hf_upload import HFHubUploadCallback

    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    cb = HFHubUploadCallback(
        repo_id="dummy/repo",
        exp_dir=str(exp_dir),
        path_in_repo="stage2",
        upload_on_val=True,
    )
    cb.api = MagicMock()

    # Skips sanity checking even on rank 0
    cb.on_validation_end(SimpleNamespace(is_global_zero=True, sanity_checking=True), None)
    cb.api.upload_folder.assert_not_called()

    # Skips non-rank-0
    cb.on_validation_end(SimpleNamespace(is_global_zero=False, sanity_checking=False), None)
    cb.api.upload_folder.assert_not_called()

    # Uploads on rank 0 after real validation
    cb.on_validation_end(SimpleNamespace(is_global_zero=True, sanity_checking=False), None)
    cb.api.upload_folder.assert_called_once_with(
        folder_path=str(exp_dir),
        repo_id="dummy/repo",
        repo_type="model",
        path_in_repo="stage2",
        ignore_patterns=["*.tmp", "*-unfinished*"],
        delete_patterns=["checkpoints/*.ckpt"],
    )

    # Disabled upload_on_val does not upload on validation end
    cb_off = HFHubUploadCallback(
        repo_id="dummy/repo",
        exp_dir=str(exp_dir),
        upload_on_val=False,
    )
    cb_off.api = MagicMock()
    cb_off.on_validation_end(SimpleNamespace(is_global_zero=True, sanity_checking=False), None)
    cb_off.api.upload_folder.assert_not_called()
