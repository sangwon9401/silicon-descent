"""04번 실습 정답. 막혔을 때만 보세요!"""
import torch


def apply_temperature(logits, temperature):
    # logits: [B, V], temperature: [B]
    return logits / temperature[:, None]


def apply_top_k(logits, top_k):
    # 요청마다 k번째로 큰 값보다 작은 건 -inf로
    V = logits.shape[-1]
    k = top_k.clamp(1, V)
    sorted_vals, _ = logits.sort(dim=-1, descending=True)
    kth = sorted_vals.gather(1, (k - 1)[:, None])                # [B, 1]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits, top_p):
    sorted_vals, idx = logits.sort(dim=-1, descending=True)
    probs = sorted_vals.softmax(dim=-1)
    cum = probs.cumsum(dim=-1)
    # 나보다 앞까지의 누적합이 이미 p 이상이면 버려요 (1등은 항상 남아요)
    remove = (cum - probs) >= top_p[:, None]
    sorted_vals = sorted_vals.masked_fill(remove, float("-inf"))
    out = torch.empty_like(logits)
    out.scatter_(1, idx, sorted_vals)
    return out


def sample(logits, temperature, top_k, top_p, generator=None):
    """logits [B, V] -> 토큰 [B]. temperature가 0이면 그 요청은 greedy."""
    greedy = temperature <= 0
    t = torch.where(greedy, torch.ones_like(temperature), temperature)
    x = apply_temperature(logits, t)
    x = apply_top_k(x, top_k)
    x = apply_top_p(x, top_p)
    probs = x.softmax(dim=-1)
    picked = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return torch.where(greedy, logits.argmax(dim=-1), picked)
