from jaxtyping import Float, Int
from torch import Tensor
import torch
from torch.nn.functional import log_softmax
from einops import rearrange, reduce
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
import math
import numpy.typing as npt
import numpy as np
import torch.nn as nn
from typing import IO, BinaryIO
import os


def cross_entropy(logits: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]):
    log_probs = log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return -reduce(target_log_probs, "... ->", "mean")


class AdamW(Optimizer):
    # self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
    # self.param_groups: List[Dict[str, Any]] = []
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)  # iteration number
                m = state.get("m", 0)  # first moment estimate
                v = state.get("v", 0)  # second moment estimate

                grad = p.grad.data
                p_v = p.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                lr_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p_v -= lr_t * m / (torch.sqrt(v) + eps)
                p_v -= lr * weight_decay * p_v
                p.data = p_v

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
        1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
    )


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    sum = 0
    for g in grads:
        sum += reduce(g**2, "...->", "sum")

    l2_norm = math.sqrt(sum)

    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for g in grads:
            g.data *= scale


def load_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Float[Tensor, "batch_size context_length"], Float[Tensor, "batch_size context_length"]]:
    max_start = len(dataset) - context_length - 1

    starts = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = torch.tensor(np.stack([dataset[start : start + context_length] for start in starts]), device=device)
    labels = torch.tensor(
        np.stack([dataset[start + 1 : start + context_length + 1] for start in starts]), device=device
    )

    return (inputs, labels)


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]
):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


if __name__ == "__main__":
    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = nn.Linear(d_input, d_output)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    it = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
        it += 1

    serialization_path = "checkpoint.pt"
    # Save the model
    save_checkpoint(
        model,
        optimizer,
        iteration=it,
        out=serialization_path,
    )
