import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import gr1_policy

VIEWS = ("ego_view", "left_wrist_inner", "right_wrist_inner")


def test_gr1_select_order():
    x = np.arange(44, dtype=np.float32)
    expected = [*range(7), *range(22, 29), *range(7, 13), *range(29, 35), *range(41, 44)]
    np.testing.assert_array_equal(gr1_policy.gr1_select(x), expected)
    assert gr1_policy.GR1_DIM == 29
    assert gr1_policy.GR1_DELTA_MASK == (True,) * 26 + (False,) * 3


def test_inputs_vlm_and_edge_views():
    example = gr1_policy.make_gr1_example(VIEWS)
    example["actions"] = np.random.rand(10, 44)
    out = gr1_policy.GR1Inputs(_model.ModelType.PI05, vlm_views=("ego_view",), edge_views=VIEWS[1:])(example)

    np.testing.assert_array_equal(out["image"]["base_0_rgb"], example["observation/images/ego_view"])
    assert out["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": False}
    assert list(out["edge_image"]) == list(VIEWS[1:])
    for view in VIEWS[1:]:
        np.testing.assert_array_equal(out["edge_image"][view], example[f"observation/images/{view}"])
    assert out["state"].shape == (29,)
    assert out["actions"].shape == (10, 29)


def test_inputs_all_views_in_vlm():
    example = gr1_policy.make_gr1_example(VIEWS)
    out = gr1_policy.GR1Inputs(_model.ModelType.PI05, vlm_views=VIEWS)(example)
    assert all(out["image_mask"].values())
    np.testing.assert_array_equal(out["image"]["right_wrist_0_rgb"], example["observation/images/right_wrist_inner"])
    assert "edge_image" not in out


def test_chw_float_images_are_converted():
    example = gr1_policy.make_gr1_example(VIEWS[:1])
    example["observation/images/ego_view"] = np.random.rand(3, 256, 256).astype(np.float32)
    out = gr1_policy.GR1Inputs(_model.ModelType.PI05)(example)
    assert out["image"]["base_0_rgb"].shape == (256, 256, 3)
    assert out["image"]["base_0_rgb"].dtype == np.uint8


def test_delta_round_trip():
    state, actions = np.random.rand(29), np.random.rand(10, 29)
    delta = transforms.DeltaActions(gr1_policy.GR1_DELTA_MASK)({"state": state, "actions": actions.copy()})
    np.testing.assert_allclose(delta["actions"][:, :26], actions[:, :26] - state[:26])
    np.testing.assert_array_equal(delta["actions"][:, 26:], actions[:, 26:])
    restored = transforms.AbsoluteActions(gr1_policy.GR1_DELTA_MASK)(delta)
    np.testing.assert_allclose(restored["actions"], actions)


def test_outputs_keep_29_dims():
    out = gr1_policy.GR1Outputs()({"actions": np.random.rand(10, 32)})
    assert out["actions"].shape == (10, 29)
