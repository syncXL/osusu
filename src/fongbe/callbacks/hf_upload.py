from pathlib import Path
from lightning.pytorch.callbacks import Callback
from huggingface_hub import HfApi

import gc
from pathlib import Path
from huggingface_hub import HfApi
from lightning.pytorch.callbacks import Callback


class HFHubUploadCallback(Callback):
    def __init__(
        self,
        repo_id: str,
        exp_dir: str,
        path_in_repo: str = "",
        upload_every_n_epochs: int = 1,
    ):
        self.repo_id = repo_id
        self.exp_dir = Path(exp_dir)
        self.path_in_repo = path_in_repo
        self.upload_every_n_epochs = max(1, upload_every_n_epochs)
        self.api = HfApi()

    def _upload(self, trainer):
        if not trainer.is_global_zero or not self.exp_dir.exists():
            return
        gc.collect()
        self.api.upload_folder(
            folder_path=str(self.exp_dir),
            repo_id=self.repo_id,
            repo_type="model",
            path_in_repo=self.path_in_repo or None,
            ignore_patterns=["*.tmp", "*-unfinished*"],
            delete_patterns=["checkpoints/*.ckpt"],
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.upload_every_n_epochs == 0:
            self._upload(trainer)

    def on_train_end(self, trainer, pl_module):
        self._upload(trainer)

# class HFHubUploadCallback(Callback):
#     def __init__(
#         self,
#         repo_id: str,
#         exp_dir: str | None = None,
#         path_in_repo: str = "",
#         token: str | None = None,
#     ):
#         self.repo_id = repo_id
#         self.exp_dir = Path(exp_dir) if exp_dir else None
#         self.path_in_repo = path_in_repo
#         self.api = HfApi(token=token)
#         self._active_future = None

#     def _resolve_log_dir(self, trainer) -> Path | None:
#         if self.exp_dir is not None:
#             return self.exp_dir
#         return Path(trainer.log_dir) if trainer.log_dir else None

#     def _flush_tensorboard(self, trainer):
#         for logger in trainer.loggers:
#             experiment = getattr(logger, "experiment", None)
#             if hasattr(experiment, "flush"):
#                 experiment.flush()

#     def _trigger_upload(self, trainer, blocking: bool = False):
#         if not trainer.is_global_zero:
#             return
#         log_dir = self._resolve_log_dir(trainer)
#         if log_dir is None or not log_dir.exists():
#             return
#         if not blocking and self._active_future is not None and not self._active_future.done():
#             return

#         self._flush_tensorboard(trainer)
#         result = self.api.upload_folder(
#             folder_path=str(log_dir),
#             repo_id=self.repo_id,
#             repo_type="model",
#             path_in_repo=self.path_in_repo or None,
#             ignore_patterns=["*.tmp"],
#             run_as_future=not blocking,
#         )
#         if not blocking:
#             self._active_future = result

#     def on_validation_end(self, trainer, pl_module):
#         self._trigger_upload(trainer, blocking=False)

#     def on_train_end(self, trainer, pl_module):
#         if self._active_future is not None:
#             self._active_future.result()
#         self._trigger_upload(trainer, blocking=True)