"""Masked loss/accuracy helpers and a small running accumulator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


def masked_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.argmax(dim=-1)
    valid = labels != IGNORE_INDEX
    correct = ((predictions == labels) & valid).sum()
    return correct, valid.sum()


@dataclass
class RunningMetrics:
    loss_sum: float = 0.0
    correct: int = 0
    valid: int = 0
    tokens: int = 0

    def update(self, loss_mean: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        correct, valid = masked_accuracy(logits, labels)
        valid_int = int(valid.item())
        self.loss_sum += float(loss_mean.detach().item()) * valid_int
        self.correct += int(correct.item())
        self.valid += valid_int
        self.tokens += labels.numel()

    @property
    def loss(self) -> float:
        return self.loss_sum / max(self.valid, 1)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.valid, 1)

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "valid_tokens": float(self.valid),
            "tokens": float(self.tokens),
        }
