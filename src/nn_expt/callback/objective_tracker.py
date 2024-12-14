from typing import Any, List, Literal, Mapping

import lightning as L
import torch


class ObjectiveTracker(L.Callback):
    def __init__(
        self,
        objective: str = "loss",
        direction: Literal["minimum", "maximum"] = "minimum",
    ):
        super().__init__()
        self.objective = objective
        self.direction = direction

        self.current_values: List[float] = []
        self.best_value: float | None = None

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        self.current_values = []
        self.best_value = None

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if isinstance(outputs, torch.Tensor):
            value = float(outputs.mean())
        elif isinstance(outputs, Mapping):
            value = float(outputs[self.objective])
        else:
            return

        self.current_values.append(value)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        value = sum(self.current_values) / len(self.current_values)
        if self.best_value is None:
            self.best_value = value
        elif self.direction == "minimum" and value < self.best_value:
            self.best_value = value
        elif self.direction == "maximum" and value > self.best_value:
            self.best_value = value
