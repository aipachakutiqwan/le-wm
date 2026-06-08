import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class _ColumnNorm:
    """Picklable z-score normaliser (closure norm_fn is not picklable by workers)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    normalizer = dt.transforms.WrapTorchTransform(_ColumnNorm(mean, std), source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            self.save_epoch(pl_module.model, trainer.current_epoch + 1)

            # save final epoch even if it falls off the interval
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self.save_epoch(pl_module.model, trainer.current_epoch + 1, force=True)

    def save_epoch(self, model, epoch: int, force: bool = False):
        """Pickle the model object for a 1-indexed epoch, honouring epoch_interval.

        Framework-agnostic — usable from a hand-rolled loop (stage-2) as well as
        the Lightning callback hook (stage-1). Set force=True to ignore the interval.
        """
        if not force and epoch % self.epoch_interval != 0:
            return
        path = self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt"
        self._dump_model(model, path)

    def save_best(self, model):
        """Pickle the model to a stable best-checkpoint path (overwritten on improve)."""
        path = self.dirpath / f"{self.filename}_best_object.ckpt"
        self._dump_model(model, path)
        return path

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")