"""실습 노트북들이 같이 쓰는 도우미 함수 모음.

노트북에서 `from common import *` 로 불러와요.
"""
import os
import time

import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def pick_device() -> torch.device:
    """쓸 수 있는 가장 빠른 장치를 골라요: NVIDIA GPU(cuda) > 맥 GPU(mps) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """GPU 작업이 끝날 때까지 기다려요. 시간을 잴 때 꼭 필요해요."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(device=None, dtype=torch.float32):
    """SmolLM2-135M 모델과 토크나이저를 불러와요. 처음 한 번은 약 270MB를 내려받아요."""
    import warnings

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()          # 영어 경고 메시지와 진행 막대를 숨겨요
    hf_logging.disable_progress_bar()
    warnings.filterwarnings("ignore")
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    return model, tok


def timeit(fn, device, warmup=2, repeat=5):
    """fn()을 여러 번 실행해서 한 번당 평균 시간(초)을 돌려줘요."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeat


# ---------- 채점 도우미 ----------

def _ok(msg):
    print(f"✅ 정답이에요! {msg}")


def _ng(msg):
    print(f"❌ 아직이에요. {msg}")


def check_close(name, mine, ref, atol=1e-4, rtol=1e-4):
    """내 결과(mine)와 정답(ref)이 거의 같은지 확인해요."""
    if mine is None or mine is Ellipsis or (isinstance(mine, float) and mine != mine):
        _ng(f"{name}: 아직 결과가 없어요. 함수 안의 `...` 부분을 채워 주세요.")
        return False
    if isinstance(mine, (int, float)):
        mine = torch.tensor(float(mine))
        ref = torch.as_tensor(ref, dtype=torch.float32)
    if not isinstance(mine, torch.Tensor):
        _ng(f"{name}: 결과가 텐서가 아니에요 (지금은 {type(mine).__name__}).")
        return False
    if mine.shape != ref.shape:
        _ng(f"{name}: 모양이 달라요. 내 결과 {tuple(mine.shape)}, 정답 {tuple(ref.shape)}")
        return False
    a, b = mine.float(), ref.float()
    same_inf = torch.isinf(a) & torch.isinf(b) & (a == b)       # -inf끼리는 같은 걸로 쳐요
    diff = torch.where(same_inf, torch.zeros_like(a), (a - b).abs()).max().item()
    if torch.allclose(mine.float(), ref.float(), atol=atol, rtol=rtol):
        _ok(f"{name} (최대 오차 {diff:.2e})")
        return True
    _ng(f"{name}: 값이 달라요. 최대 오차 {diff:.2e}")
    return False


def check_equal(name, mine, ref):
    """두 값이 정확히 같은지 확인해요 (토큰 목록 비교용)."""
    if mine == ref:
        _ok(name)
        return True
    _ng(f"{name}\n   내 결과: {mine}\n   정답:    {ref}")
    return False


def run_check(name, fn):
    """채점 함수를 실행하고, 아직 안 채운 함수 때문에 나는 에러를 친절하게 알려줘요."""
    try:
        ok = fn()
        if ok and os.environ.get("LAB_APP"):
            print(f"[[PASS:{name}]]")          # 실습실 웹사이트가 진도를 기록하는 표시 (화면엔 안 보여요)
        return ok
    except NotImplementedError:
        _ng(f"{name}: 함수를 아직 안 채웠어요.")
    except Exception as e:  # noqa: BLE001 - 학습용이라 모든 에러를 설명해 줘요
        _ng(f"{name}: 실행 중 에러가 났어요 → {type(e).__name__}: {e}")
    return False


# ---------- 모델 가중치 꺼내기 ----------

def model_cfg(model) -> dict:
    """우리가 짤 코드에 필요한 설정값만 골라서 dict로 만들어요."""
    c = model.config
    theta = getattr(c, "rope_theta", None) or c.rope_parameters["rope_theta"]
    return dict(
        L=c.num_hidden_layers,            # 레이어 수 (30)
        nq=c.num_attention_heads,         # Q 헤드 수 (9)
        nkv=c.num_key_value_heads,        # K, V 헤드 수 (3)
        hd=c.head_dim,                    # 헤드 하나의 크기 (64)
        eps=c.rms_norm_eps,               # RMSNorm에서 0으로 나누는 걸 막는 작은 수
        theta=theta,                      # RoPE 회전 속도 기준값
    )


def extract_weights(model) -> dict:
    """HuggingFace 모델에서 가중치 텐서만 꺼내 이름을 짧게 붙여요."""
    m = model.model
    layers = []
    for l in m.layers:
        layers.append(dict(
            ln1=l.input_layernorm.weight.data,
            ln2=l.post_attention_layernorm.weight.data,
            wq=l.self_attn.q_proj.weight.data,
            wk=l.self_attn.k_proj.weight.data,
            wv=l.self_attn.v_proj.weight.data,
            wo=l.self_attn.o_proj.weight.data,
            w_gate=l.mlp.gate_proj.weight.data,
            w_up=l.mlp.up_proj.weight.data,
            w_down=l.mlp.down_proj.weight.data,
        ))
    return dict(embed=m.embed_tokens.weight.data, norm=m.norm.weight.data,
                lm_head=model.lm_head.weight.data, layers=layers)


def setup_korean_font():
    """그래프에 한글이 깨지지 않게 글꼴을 골라요 (맥: AppleGothic)."""
    import matplotlib
    from matplotlib import font_manager
    nanum = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"   # Colab 준비 칸이 설치해요
    if os.path.exists(nanum):
        font_manager.fontManager.addfont(nanum)
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
        if cand in names:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
