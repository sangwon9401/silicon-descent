"""01번 실습 정답. 막혔을 때만 보세요!"""
import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    # 마지막 차원의 제곱 평균으로 나눠 크기를 맞추고, 학습된 weight를 곱해요.
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def rope_cos_sin(positions, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]      # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                      # [T, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    # 벡터를 반으로 나눠 [x1, x2] -> [-x2, x1]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: [T, n_heads, head_dim], cos/sin: [T, head_dim]
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def attention(q, k, v):
    """q: [T, nq, hd], k/v: [S, nkv, hd] -> [T, nq, hd]  (causal, GQA)
    T는 이번에 계산하는 토큰 수, S는 지금까지의 전체 토큰 수(S >= T)."""
    T, nq, hd = q.shape
    S, nkv, _ = k.shape
    group = nq // nkv
    k = k.repeat_interleave(group, dim=1)                        # [S, nq, hd]
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("tnd,snd->nts", q, k) / hd ** 0.5       # [nq, T, S]
    # 이번 토큰들의 실제 위치는 S-T ... S-1 이에요. 자기보다 뒤(미래)는 가려요.
    qpos = torch.arange(S - T, S, device=q.device)[:, None]
    kpos = torch.arange(S, device=q.device)[None, :]
    scores = scores.masked_fill(kpos > qpos, float("-inf"))
    probs = scores.softmax(dim=-1)
    return torch.einsum("nts,snd->tnd", probs, v)


def mlp(x, w_gate, w_up, w_down):
    return (F.silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T


def decoder_layer(x, lw, cfg, cos, sin):
    T = x.shape[0]
    h = rms_norm(x, lw["ln1"], cfg["eps"])
    q = (h @ lw["wq"].T).view(T, cfg["nq"], cfg["hd"])
    k = (h @ lw["wk"].T).view(T, cfg["nkv"], cfg["hd"])
    v = (h @ lw["wv"].T).view(T, cfg["nkv"], cfg["hd"])
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    a = attention(q, k, v).reshape(T, -1) @ lw["wo"].T
    x = x + a
    x = x + mlp(rms_norm(x, lw["ln2"], cfg["eps"]), lw["w_gate"], lw["w_up"], lw["w_down"])
    return x


def forward(tokens, W, cfg):
    """tokens: [T] 정수 텐서 -> logits [T, vocab]"""
    x = W["embed"][tokens]
    pos = torch.arange(tokens.shape[0], device=tokens.device)
    cos, sin = rope_cos_sin(pos, cfg["hd"], cfg["theta"], tokens.device)
    for lw in W["layers"]:
        x = decoder_layer(x, lw, cfg, cos, sin)
    x = rms_norm(x, W["norm"], cfg["eps"])
    return x @ W["lm_head"].T


def greedy_generate(tokens, W, cfg, n_new):
    tokens = tokens.clone()
    for _ in range(n_new):
        logits = forward(tokens, W, cfg)
        nxt = logits[-1].argmax()
        tokens = torch.cat([tokens, nxt[None]])
    return tokens
