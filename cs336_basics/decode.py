import torch
import torch.nn as nn
from jaxtyping import Int, Float
from torch import Tensor
from model import softmax


def decode(
    model: nn.Module,
    input_ids: Int[Tensor, " seq_len"],
    end_tokens: set[int] | None = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Int[Tensor, " seq_len"]:
    model.eval()

    for _ in range(max_tokens):
        logits = model(input_ids)
        logits = logits[-1, :]

        if temperature != 0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > top_p
            mask[0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-8)
            probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)

        if temperature == 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        if end_tokens is not None and int(next_token) in end_tokens:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return input_ids
