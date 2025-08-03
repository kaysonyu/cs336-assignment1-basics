import torch.nn as nn
import torch
from jaxtyping import Float, Int
import math
from torch import Tensor
from einops import einsum, reduce, rearrange


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        std = math.sqrt(2 / (d_out + d_in))
        weight = torch.empty(d_out, d_in, device=device, dtype=dtype)
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std), requires_grad=True
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight: Float[Tensor, "num_embeddings embedding_dim"] = nn.Parameter(
            nn.init.trunc_normal_(weight, a=-3.0, b=3.0), requires_grad=True
        )

    def forward(self, token_ids: Float[Tensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight: Float[Tensor, " d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype), requires_grad=True
        )
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "sum") / x.shape[-1] + self.eps)
        result = x / rms * self.weight
        return result.to(in_type)


def SiLU(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        output1 = self.w1(x)
        output3 = self.w3(x)
        return self.w2(SiLU(output1) * output3)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        dim_theta = theta ** (-2 * torch.arange(d_k // 2, device=device) / d_k)
        final_theta = einsum(torch.arange(max_seq_len, device=device), dim_theta, "seq_len, dim_g -> seq_len dim_g")
        self.register_buffer("sin", torch.sin(final_theta), persistent=False)
        self.register_buffer("cos", torch.cos(final_theta), persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        # [x1 y1 x2 y2] * cos + [-y1 x1 -y2 x2] * sin
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_ = torch.stack((-x2, x1), dim=-1).flatten(-2)
        cos = self.cos[token_positions].repeat_interleave(2, dim=-1)
        sin = self.sin[token_positions].repeat_interleave(2, dim=-1)
        cos_part = x * cos
        sin_part = x_ * sin

        return cos_part + sin_part


def softmax(x: Float[Tensor, "..."], dim: int = -1):
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
):
    sim = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(Q.shape[-1])
    if mask is not None:
        sim = sim.masked_fill(~mask, float("-inf"))
    sim = softmax(sim, dim=-1)
    return einsum(sim, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, use_rope: bool = False, max_seq_len: int = 1024, theta: float = 1000
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len)
        self.use_rope = use_rope

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        batch_dim = x.shape[:-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q, K, V = (
            rearrange(X, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads) for X in (Q, K, V)
        )

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len).unsqueeze(0).expand(*batch_dim, seq_len)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        qkv = scaled_dot_product_attention(Q, K, V, mask=~mask)
        qkv = rearrange(qkv, "... heads seq_len d_k -> ... seq_len (heads d_k)")
        return self.output_proj(qkv)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x_attn = x + self.attn(self.ln1(x))
        return x_attn + self.ffn(self.ln2(x_attn))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, tokens: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
