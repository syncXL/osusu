from typing import Any

import lightning.pytorch as pl

class HFStreamCheckpointCallback(pl.Callback):
    def __init__(self):
        self._pending_state = None

    def _get_hf_dataset(self, trainer):
        dl = trainer.train_dataloader
        if dl is None:
            return None
        ds = getattr(dl, "dataset", None)
        return getattr(ds, "dataset", ds)

    def state_dict(self) -> dict[str, Any]:
        return {"hf_stream_state" : self._pending_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._pending_state = state_dict.get("hf_stream_state")

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        hf_ds = self._get_hf_dataset(trainer)
        if hf_ds is not None and hasattr(hf_ds, "state_dict"):
            self._pending_state = hf_ds.state_dict()
            checkpoint["hf_stream_state"] = self._pending_state

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._pending_state is not None:
            hf_ds = self._get_hf_dataset(trainer)
            if hf_ds is not None and hasattr(hf_ds, "load_state_dict"):
                hf_ds.load_state_dict(self._pending_state)
                self._pending_state = None