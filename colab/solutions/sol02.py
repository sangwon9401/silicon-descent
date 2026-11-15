"""02번 실습 정답. 막혔을 때만 보세요!"""
import torch

from solutions.sol01 import apply_rope, attention, mlp, rms_norm, rope_cos_sin


def new_cache(cfg):
    # 레이어마다 K, V를 담을 빈 리스트
    return [{"k": None, "v": None} for _ in range(cfg["L"])]


def forward_with_cache(tokens, W, cfg, cache, start_pos):
    """tokens: 이번에 새로 넣는 토큰들 [T]. start_pos: 이 토큰들의 시작 위치.
    cache를 갱신하고 logits [T, vocab]를 돌려줘요."""
    T = tokens.shape[0]
    x = W["embed"][tokens]
    pos = torch.arange(start_pos, start_pos + T, device=tokens.device)
    cos, sin = rope_cos_sin(pos, cfg["hd"], cfg["theta"], tokens.device)
    for lw, c in zip(W["layers"], cache):
        h = rms_norm(x, lw["ln1"], cfg["eps"])
        q = (h @ lw["wq"].T).view(T, cfg["nq"], cfg["hd"])
        k = (h @ lw["wk"].T).view(T, cfg["nkv"], cfg["hd"])
        v = (h @ lw["wv"].T).view(T, cfg["nkv"], cfg["hd"])
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # 새 K, V를 캐시 뒤에 붙이고, 어텐션은 캐시 전체를 봐요.
        c["k"] = k if c["k"] is None else torch.cat([c["k"], k])
        c["v"] = v if c["v"] is None else torch.cat([c["v"], v])
        a = attention(q, c["k"], c["v"]).reshape(T, -1) @ lw["wo"].T
        x = x + a
        x = x + mlp(rms_norm(x, lw["ln2"], cfg["eps"]), lw["w_gate"], lw["w_up"], lw["w_down"])
    x = rms_norm(x, W["norm"], cfg["eps"])
    return x @ W["lm_head"].T


def generate_with_cache(tokens, W, cfg, n_new):
    cache = new_cache(cfg)
    logits = forward_with_cache(tokens, W, cfg, cache, 0)        # prefill
    out = tokens.clone()
    for _ in range(n_new):
        nxt = logits[-1].argmax()
        out = torch.cat([out, nxt[None]])
        logits = forward_with_cache(nxt[None], W, cfg, cache, out.shape[0] - 1)  # decode
    return out
