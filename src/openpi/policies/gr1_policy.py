"""Transforms for the GR1 multi-view tabletop datasets (GR00T LeRobot format, `GR1ArmsAndWaistFourierHands`).

The dataset's 44-D state and action hold legs and neck as well; like the GR00T GR1 modality config, the
policy uses arms, hands and waist (29-D) in the order left arm, right arm, left hand, right hand, waist.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# (start, end) of each part in the dataset's 44-D state/action vector (meta/modality.json).
GR1_PARTS = {
    "left_arm": (0, 7),
    "right_arm": (22, 29),
    "left_hand": (7, 13),
    "right_hand": (29, 35),
    "waist": (41, 44),
}
GR1_DIM = sum(end - start for start, end in GR1_PARTS.values())  # 29
# Arms and hands are trained as deltas from the current state, the waist stays absolute.
GR1_DELTA_MASK = transforms.make_bool_mask(GR1_DIM - 3, -3)

VLM_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def gr1_select(x: np.ndarray) -> np.ndarray:
    """Pick the policy's 29 dims (in GR1_PARTS order) out of the dataset's 44-D last axis."""
    x = np.asarray(x)
    return np.concatenate([x[..., start:end] for start, end in GR1_PARTS.values()], axis=-1)


def make_gr1_example(views: tuple[str, ...]) -> dict:
    """Creates a random input example for the GR1 policy."""
    example = {
        "observation/state": np.random.rand(44),
        "prompt": "press the blue button on the side of the box",
    }
    for view in views:
        example[f"observation/images/{view}"] = np.random.randint(256, size=(256, 256, 3), dtype=np.uint8)
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GR1Inputs(transforms.DataTransformFn):
    """Maps GR1 views to the model inputs.

    `vlm_views` fill the SigLIP image slots in order (base, left wrist, right wrist); unused slots are masked
    out. `edge_views` go to the edge encoders (`Observation.edge_images`), keyed by view name.
    """

    model_type: _model.ModelType
    vlm_views: tuple[str, ...] = ("ego_view",)
    edge_views: tuple[str, ...] = ()

    def __post_init__(self):
        if not 1 <= len(self.vlm_views) <= len(VLM_IMAGE_SLOTS):
            raise ValueError(f"Need 1 to {len(VLM_IMAGE_SLOTS)} VLM views, got {self.vlm_views}.")

    def __call__(self, data: dict) -> dict:
        vlm_images = [_parse_image(data[f"observation/images/{view}"]) for view in self.vlm_views]
        images, image_masks = {}, {}
        for i, slot in enumerate(VLM_IMAGE_SLOTS):
            if i < len(vlm_images):
                images[slot], image_masks[slot] = vlm_images[i], np.True_
            else:
                images[slot], image_masks[slot] = np.zeros_like(vlm_images[0]), np.False_

        inputs = {
            "state": gr1_select(data["observation/state"]),
            "image": images,
            "image_mask": image_masks,
        }
        if self.edge_views:
            inputs["edge_image"] = {view: _parse_image(data[f"observation/images/{view}"]) for view in self.edge_views}
            inputs["edge_image_mask"] = dict.fromkeys(self.edge_views, np.True_)

        if "actions" in data:
            inputs["actions"] = gr1_select(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GR1Outputs(transforms.DataTransformFn):
    """Returns the 29-D GR1 actions (left arm, right arm, left hand, right hand, waist)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :GR1_DIM])}
