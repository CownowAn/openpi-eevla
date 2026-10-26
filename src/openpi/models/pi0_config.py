import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

# Where the edge-view tokens (EE-VLA: one DINOv3 CLS token per extra camera) enter the sequence.
# Prefix placements are processed by the PaliGemma weights, suffix placements by the action expert.
#   A1: [img][text][edge] prefix, edge in its own block (image/text tokens do not attend to it)
#   A2: [img][text][edge] prefix, one bidirectional block
#   A3: [img][edge][text] prefix, one bidirectional block
#   B1: [edge][act] suffix, separate blocks (edge does not attend to the actions)
#   B2: [act][edge] suffix, one block (actions keep their original positions)
#   B3: [edge][act] suffix, one block
EdgePlacement = Literal["none", "A1", "A2", "A3", "B1", "B2", "B3"]
PREFIX_EDGE_PLACEMENTS = ("A1", "A2", "A3")
SUFFIX_EDGE_PLACEMENTS = ("B1", "B2", "B3")
EDGE_IMAGE_RESOLUTION = (256, 256)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Edge-view tokens (PyTorch model only). `edge_image_keys` names the observation's `edge_images`
    # entries, one trainable encoder per key.
    edge_placement: EdgePlacement = "none"
    edge_image_keys: tuple[str, ...] = ()
    edge_encoder_model: str = "vit_base_patch16_dinov3.lvd1689m"
    # Training-time edge dropout: with this probability a sample has all of its edge tokens masked out.
    edge_dropout_p: float = 0.2
    # "frozen" freezes the PaliGemma backbone (SigLIP + Gemma 2B); the action expert always trains.
    vlm_train_mode: Literal["full", "frozen"] = "full"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if (self.edge_placement == "none") != (not self.edge_image_keys):
            raise ValueError("edge_placement and edge_image_keys must be set together.")
        if self.edge_placement != "none" and not self.pi05:
            raise ValueError("Edge tokens are only supported for pi05.")

    @property
    def use_edge(self) -> bool:
        return self.edge_placement != "none"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                edge_images={
                    key: jax.ShapeDtypeStruct([batch_size, *EDGE_IMAGE_RESOLUTION, 3], jnp.uint8)
                    for key in self.edge_image_keys
                }
                or None,
                edge_image_masks=dict.fromkeys(self.edge_image_keys, image_mask_spec) or None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
