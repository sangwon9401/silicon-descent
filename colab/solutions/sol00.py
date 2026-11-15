"""00번 실습 정답. 막혔을 때만 보세요!"""
import torch


def hf_k_weight(model, i):
    return model.model.layers[i].self_attn.k_proj.weight


def hf_next_token(model, ids):
    with torch.no_grad():
        return model(ids[None]).logits[0, -1].argmax()


def get_wk(W, i):
    return W["layers"][i]["wk"]


def kv_dim(cfg):
    return cfg["nkv"] * cfg["hd"]


def make_grid():
    return torch.arange(12).reshape(3, 4)


def last_two_rows(x):
    return x[-2:]


def second_col(x):
    return x[:, 1]


def swap_halves(x):
    half = x.shape[-1] // 2
    return torch.cat([x[..., half:], x[..., :half]], dim=-1)


def row_mean_sq(x):
    return x.pow(2).mean(dim=-1, keepdim=True)


def linear(x, w):
    return x @ w.T


def split_heads(x, n_heads):
    T = x.shape[0]
    return x.view(T, n_heads, -1)


def repeat_heads(k, times):
    return k.repeat_interleave(times, dim=1)


def head_scores(q, k):
    return torch.einsum("tnd,snd->nts", q, k)


def causal_mask(scores):
    T = scores.shape[-1]
    qpos = torch.arange(T)[:, None]
    kpos = torch.arange(T)[None, :]
    return scores.masked_fill(kpos > qpos, float("-inf"))


def next_token(logits):
    return logits[-1].argmax()
