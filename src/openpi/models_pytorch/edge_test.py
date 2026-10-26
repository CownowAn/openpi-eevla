"""Tests for the edge-view token placements of the PyTorch pi05 model (dummy Gemma sizes)."""

# ruff: noqa: SLF001  (the tests check the model's embedding and masking internals directly)

import dataclasses
import functools

import jax
import pytest
import torch

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch

PLACEMENTS = ("A1", "A2", "A3", "B1", "B2", "B3")
EDGE_KEYS = ("left_wrist_inner", "right_wrist_inner")
BATCH = 2
TEXT_LEN = 16
TEXT_VALID = (5, 11)  # valid prompt tokens per sample
IMAGE_MASK = {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
HORIZON = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Token types along the sequence, per placement: (prefix, suffix).
LAYOUT = {
    "none": (("img", "txt"), ("act",)),
    "A1": (("img", "txt", "edge"), ("act",)),
    "A2": (("img", "txt", "edge"), ("act",)),
    "A3": (("img", "edge", "txt"), ("act",)),
    "B1": (("img", "txt"), ("edge", "act")),
    "B2": (("img", "txt"), ("act", "edge")),
    "B3": (("img", "txt"), ("edge", "act")),
}


def expected_attends(placement: str, query: str, key: str) -> bool:
    """Whether a `query` token may attend to a `key` token (the scenario tables)."""
    vlm = ("img", "txt")
    if key == "act":
        return query == "act" or (query == "edge" and placement in ("B2", "B3"))
    if query == "act" or (key in vlm and query in (*vlm, "edge")):
        return True
    if key == "edge":
        if query == "edge":
            return True
        return query in vlm and placement in ("A2", "A3")
    raise AssertionError((placement, query, key))


def make_config(placement: str, **kwargs) -> pi0_config.Pi0Config:
    edge = {} if placement == "none" else {"edge_placement": placement, "edge_image_keys": EDGE_KEYS}
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_horizon=HORIZON,
        max_token_len=TEXT_LEN,
        pytorch_compile_mode=None,
        **edge,
        **kwargs,
    )


@functools.cache
def make_model(placement: str, **kwargs) -> pi0_pytorch.PI0Pytorch:
    torch.manual_seed(0)
    model = pi0_pytorch.PI0Pytorch(make_config(placement, **kwargs)).to(DEVICE).eval()
    if model.edge_vision is not None:
        # Randomly initialized DINOv3 has LayerScale 1e-5, so its CLS token barely depends on the image
        # (the pretrained weights loaded for training do not have this problem).
        with torch.no_grad():
            for name, param in model.edge_vision.encoders.named_parameters():
                if "gamma" in name:
                    param.fill_(1.0)
    # The model enables TF32 matmuls; the cached and full forward passes are compared in full float32.
    torch.set_float32_matmul_precision("highest")
    # Deterministic kernels, so that "does not depend on the edge views" can be checked exactly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model


def make_observation(seed: int = 0, edge_seed: int = 1, *, edge_mask: bool = True) -> _model.Observation:
    gen = torch.Generator().manual_seed(seed)
    edge_gen = torch.Generator().manual_seed(edge_seed)
    prompt_mask = torch.arange(TEXT_LEN)[None, :] < torch.tensor(TEXT_VALID)[:, None]
    observation = _model.Observation(
        # Channels-first, as `Observation.from_dict` gives the PyTorch model.
        images={key: torch.rand(BATCH, 3, 224, 224, generator=gen) * 2 - 1 for key in IMAGE_MASK},
        image_masks={key: torch.full((BATCH,), value) for key, value in IMAGE_MASK.items()},
        state=torch.rand(BATCH, 32, generator=gen) * 2 - 1,
        tokenized_prompt=torch.randint(0, 1000, (BATCH, TEXT_LEN), generator=gen) * prompt_mask,
        tokenized_prompt_mask=prompt_mask,
        edge_images={
            key: torch.randint(0, 256, (BATCH, 256, 256, 3), generator=edge_gen, dtype=torch.uint8) for key in EDGE_KEYS
        },
        edge_image_masks={key: torch.full((BATCH,), edge_mask) for key in EDGE_KEYS},
    )
    return jax.tree.map(lambda x: x.to(DEVICE), observation)


def embed_all(model, observation, noise, time):
    """Prefix + suffix embeddings, pad/attention/position masks as in the training path."""
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    edge_tokens, edge_keep = model._embed_edge(observation)
    prefix = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, edge_tokens, edge_keep)
    suffix = model.embed_suffix(state, noise, time, edge_tokens, edge_keep)
    pad = torch.cat([prefix[1], suffix[1]], dim=1)
    att = torch.cat([prefix[2], suffix[2]], dim=1)
    pos = torch.cat([prefix[3], suffix[4]], dim=1)
    return prefix, suffix, pi0_pytorch.make_att_2d_masks(pad, att), pad, torch.cumsum(pos, dim=1) - 1


def token_types(placement: str) -> list[str]:
    sizes = {"img": 256 * len(IMAGE_MASK), "txt": TEXT_LEN, "edge": len(EDGE_KEYS), "act": HORIZON}
    prefix, suffix = LAYOUT[placement]
    return [kind for kind in (*prefix, *suffix) for _ in range(sizes[kind])]


def velocity(model, observation, x_t, time):
    with torch.no_grad():
        preprocessed = model._preprocess_observation(observation, train=False)
        return model._predict_velocity(preprocessed, observation, x_t, time)


def noise_like(seed: int = 2):
    return torch.randn(BATCH, HORIZON, 32, generator=torch.Generator().manual_seed(seed)).to(DEVICE)


@pytest.mark.parametrize("placement", ["none", *PLACEMENTS])
def test_attention_blocks_and_positions(placement):
    model = make_model(placement)
    with torch.no_grad():
        _, _, att_2d, pad, positions = embed_all(
            model, make_observation(), noise_like(), torch.ones(BATCH, device=DEVICE)
        )
    types = token_types(placement)
    assert att_2d.shape == (BATCH, len(types), len(types))

    kinds = sorted(set(types))
    index = {kind: torch.tensor([i for i, t in enumerate(types) if t == kind], device=DEVICE) for kind in kinds}
    for b in range(BATCH):
        valid = {kind: idx[pad[b, idx]] for kind, idx in index.items()}
        for query in kinds:
            for key in kinds:
                block = att_2d[b][valid[query]][:, valid[key]]
                assert bool(block.all() if expected_attends(placement, query, key) else ~block.any()), (
                    placement,
                    query,
                    key,
                )

        # Valid tokens take consecutive positions in sequence order; padding takes none.
        assert positions[b][pad[b]].tolist() == list(range(int(pad[b].sum())))
        # The actions follow every non-action token unless they come first in the suffix (B2).
        n_vlm = 256 * sum(IMAGE_MASK.values()) + TEXT_VALID[b]
        n_before_act = n_vlm + (0 if placement in ("none", "B2") else len(EDGE_KEYS))
        assert positions[b][index["act"]].tolist() == list(range(n_before_act, n_before_act + HORIZON))


@pytest.mark.parametrize("placement", ["none", *PLACEMENTS])
def test_cached_sampling_matches_training_path(placement):
    """Two Euler steps through the KV-cache path equal the same steps through full forward passes."""
    model = make_model(placement)
    observation, noise = make_observation(), noise_like()
    with torch.no_grad():
        sampled = model.sample_actions(DEVICE, observation, noise=noise, num_steps=2)
    x_t = noise
    for t in (1.0, 0.5):
        x_t = x_t - 0.5 * velocity(model, observation, x_t, torch.full((BATCH,), t, device=DEVICE))
    torch.testing.assert_close(sampled, x_t, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_edge_tokens_are_read_and_masked(placement):
    model = make_model(placement)
    noise, time = noise_like(), torch.full((BATCH,), 0.7, device=DEVICE)
    # Exact comparisons: with random weights the edge views move the actions only by ~1e-6.
    base = velocity(model, make_observation(edge_seed=1), noise, time)
    assert torch.equal(base, velocity(model, make_observation(edge_seed=1), noise, time))
    assert not torch.equal(base, velocity(model, make_observation(edge_seed=3), noise, time))

    # Masked edge views have no effect.
    masked = velocity(model, make_observation(edge_seed=1, edge_mask=False), noise, time)
    masked_other = velocity(model, make_observation(edge_seed=3, edge_mask=False), noise, time)
    assert torch.equal(masked, masked_other)


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_vlm_tokens_see_edges_only_in_a2_a3(placement, attn_implementation):
    """Image/text outputs of the PaliGemma pass depend on the edge views only for A2 and A3.

    Only valid tokens are compared: a padding query has every key masked, so its (unused) output averages
    over all tokens, edges included.
    """
    model = make_model(placement)
    config = model.paligemma_with_expert.paligemma.language_model.config
    saved, config._attn_implementation = config._attn_implementation, attn_implementation

    def vlm_outputs(edge_seed):
        observation = make_observation(edge_seed=edge_seed)
        with torch.no_grad():
            (prefix_embs, prefix_pad, prefix_att, prefix_pos), *_ = embed_all(
                model, observation, noise_like(), torch.ones(BATCH, device=DEVICE)
            )
            mask = model._prepare_attention_masks_4d(pi0_pytorch.make_att_2d_masks(prefix_pad, prefix_att))
            (out, _), _ = model.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=torch.cumsum(prefix_pos, dim=1) - 1,
                inputs_embeds=[prefix_embs, None],
            )
        types = token_types(placement)[: out.shape[1]]
        vlm = torch.tensor([t in ("img", "txt") for t in types], device=DEVICE)
        return out[vlm[None, :] & prefix_pad]

    try:
        assert torch.equal(vlm_outputs(1), vlm_outputs(3)) == (placement not in ("A2", "A3"))
    finally:
        config._attn_implementation = saved


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_edge_dropout_keeps_positions(placement):
    model = make_model(placement, edge_dropout_p=1.0)
    observation, noise = make_observation(), noise_like()
    with torch.no_grad():
        _, _, _, pad_eval, pos_eval = embed_all(model, observation, noise, torch.ones(BATCH, device=DEVICE))
        model.train()
        try:
            _, _, _, pad_train, pos_train = embed_all(model, observation, noise, torch.ones(BATCH, device=DEVICE))
        finally:
            model.eval()
    edge = torch.tensor([t == "edge" for t in token_types(placement)], device=DEVICE)
    assert pad_eval[:, edge].all()
    assert not pad_train[:, edge].any()
    torch.testing.assert_close(pad_train[:, ~edge], pad_eval[:, ~edge])
    torch.testing.assert_close(pos_train, pos_eval)


@pytest.mark.parametrize("placement", ["A1", "B1"])
def test_frozen_vlm_trains_edges_and_expert(placement):
    model = make_model(placement, vlm_train_mode="frozen")
    model.train()
    try:
        observation, noise = make_observation(), noise_like()
        actions = torch.randn(BATCH, HORIZON, 32, device=DEVICE)
        model(observation, actions, noise=noise, time=torch.full((BATCH,), 0.5, device=DEVICE)).mean().backward()
    finally:
        model.eval()
    params = dict(model.named_parameters())
    vlm = {name for name in params if name.startswith("paligemma_with_expert.paligemma.")}
    assert vlm
    assert all(not params[name].requires_grad for name in vlm)
    assert all(params[name].grad is None for name in vlm)
    for prefix in ("edge_vision.encoders.", "edge_vision.proj.", "paligemma_with_expert.gemma_expert."):
        grads = [p.grad for name, p in params.items() if name.startswith(prefix) and p.requires_grad]
        assert grads, prefix
        assert any(g is not None and g.abs().sum() > 0 for g in grads), prefix
    model.zero_grad(set_to_none=True)


def test_config_validation():
    with pytest.raises(ValueError, match="set together"):
        pi0_config.Pi0Config(pi05=True, edge_placement="B1")
    with pytest.raises(ValueError, match="only supported for pi05"):
        pi0_config.Pi0Config(edge_placement="B1", edge_image_keys=EDGE_KEYS)
    config = dataclasses.replace(make_config("B1"), action_horizon=8)
    assert config.use_edge
    assert config.action_horizon == 8
