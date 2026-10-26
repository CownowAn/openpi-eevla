# π0.5 + Edge-view tokens (EE-VLA on openpi)

Experimental code that ports the idea of GR00T N1.7 EE-VLA (Edge-Expert VLA) to openpi's π0.5 (PyTorch implementation).
The main view goes through SigLIP → PaliGemma as before, and each extra camera (edge view) is summarized by **the CLS token of its own DINOv3 ViT-B/16** and inserted into the sequence. Six scenarios (A1–A3, B1–B3) are supported, which differ in where the tokens go in the sequence and which tokens can attend to them.

- Base: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) (`215abfb`)
- Scope: **PyTorch model only**. The JAX `Pi0` raises `NotImplementedError` when given an edge config.
- First target data: GR1 ColorButtonBoxV3 L3, `cam3b` (ego + left/right wrist-inner)

---

## 1. Edge tokens

```
edge view i (256×256 uint8) ─► DINOv3_i (per view, trained) ─► CLS (768)
                              ─► Linear(768→D) ─► GELU ─► Linear(D→D) ─► LayerNorm ─► one edge token
D = 2048 (A placements: PaliGemma width) / 1024 (B placements: action expert width)
```

- **Sample-level edge dropout** during training: with probability `edge_dropout_p` (default 0.2), all edge tokens of a sample are removed from attention.
- Edge tokens that are dropped or whose camera mask is off **still take their RoPE positions**, so the positions of later tokens (text, actions) do not change whether edges are present or not. The attention mask (`pad_masks`) and the position mask (`pos_masks`) are kept separate for this.
- DINOv3 weights: the model is built without weights. When training starts and the base checkpoint has no edge keys, `train_pytorch.py` loads timm's pretrained DINOv3 (`vit_base_patch16_dinov3.lvd1689m`).

## 2. Scenarios

`ar_mask` rule (openpi `make_attn_mask`): a token can attend only to tokens whose cumulative sum is less than or equal to its own. A `1` starts a new block, and earlier blocks cannot attend to it.

| | Sequence (`\|` = prefix/suffix boundary) | edge `ar_mask` | act `ar_mask` | Weights that process the edges |
|---|---|---|---|---|
| **A1** | `[img][text][edge]` \| `[act]` | `1 0…` | `1 0…` | PaliGemma (2048d) |
| **A2** | `[img][text][edge]` \| `[act]` | `0 0…` | `1 0…` | PaliGemma |
| **A3** | `[img][edge][text]` \| `[act]` | `0 0…` | `1 0…` | PaliGemma |
| **B1** | `[img][text]` \| `[edge][act]` | `1 0…` | `1 0…` | Action expert (1024d) |
| **B2** | `[img][text]` \| `[act][edge]` | `0 0…` | `1 0…` | Action expert |
| **B3** | `[img][text]` \| `[edge][act]` | `1 0…` | `0 0…` | Action expert |

Who can attend to whom (✓ = can attend):

| query → key | img/text → edge | edge → img/text | edge → act | act → edge |
|---|---|---|---|---|
| A1 | ✗ | ✓ | ✗ | ✓ |
| A2, A3 | ✓ | ✓ | ✗ | ✓ |
| B1 | ✗ | ✓ | ✗ | ✓ |
| B2, B3 | ✗ | ✓ | ✓ | ✓ |

Properties of each scenario:

| | VLM sees edges | Edges depend on t (adaRMS) | Extra inference cost per step | Action RoPE positions | Notes |
|---|---|---|---|---|---|
| A1 | ✗ | ✗ | 0 (prefix KV cache) | +V | VLM representations unchanged; edges pass through the 2B model |
| A2 | ✓ | ✗ | 0 | +V | The VLM mixes in edge information |
| A3 | ✓ | ✗ | 0 | unchanged | Same as A2, but keeps the `"Action:"` ↔ action adjacency |
| B1 | ✗ | ✓ | V tokens | +V | Same scheme as the π0 state token; closest to EE-VLA |
| B2 | ✗ | ✓ (+ x_t) | V tokens | unchanged | Output is the **first** 50 suffix tokens (`[:, :H]`) |
| B3 | ✗ | ✓ (+ x_t) | V tokens | +V | Same mask as B2, same positions as B1 |

- A placements: the edges are in the prefix, so they go into the KV cache and are not affected by adaRMS(t).
- B placements: the action expert's adaRMSNorm is applied per expert, so the edge tokens are also conditioned on t. They are therefore recomputed at every Euler step. DINOv3 runs only once, outside the loop.
- Comparison pairs:
  - B1 ↔ B3: same positions; they differ only in whether the edges see x_t.
  - B3 ↔ B2: same mask; they differ only in RoPE positions.
  - A1 ↔ B1: they differ only in which weights process the edges (2B vs 300M + t).
  - A1 ↔ A3: they differ in whether the VLM sees the edges.

## 3. Config (`Pi0Config`)

| Field | Default | Description |
|---|---|---|
| `edge_placement` | `"none"` | `none`, `A1`–`A3`, `B1`–`B3` |
| `edge_image_keys` | `()` | Edge view names, one DINOv3 per view. Must be set together with `edge_placement` |
| `edge_encoder_model` | `vit_base_patch16_dinov3.lvd1689m` | timm model name |
| `edge_dropout_p` | `0.2` | Sample-level edge dropout |
| `vlm_train_mode` | `"full"` | `"frozen"` freezes SigLIP + Gemma 2B. The action expert and the edge encoders always train |
| `action_horizon` | `50` | π0.5 default. Can be changed from the CLI |

With `edge_placement="none"`, the behavior is **bit-for-bit identical** to the original openpi (see the verification below).

## 4. Training configs

From `_gr1_colorbutton_configs()` in `src/openpi/training/config.py`:

| Config | VLM (SigLIP) views | Edge views |
|---|---|---|
| `pi05_gr1_colorbutton_cam3b` | ego, left_wrist_inner, right_wrist_inner | – (baseline) |
| `pi05_gr1_colorbutton_cam3b_edge_{A1,A2,A3,B1,B2,B3}` | ego | left_wrist_inner, right_wrist_inner |

Data processing (`policies/gr1_policy.py`, `LeRobotGR1DataConfig`):
- State and actions use **29** of the 44 dims (left arm 7, right arm 7, left hand 6, right hand 6, waist 3), in the same order as the GR00T GR1 modality config.
- Arms and hands are deltas from the current state; the waist stays absolute (`DeltaActions` / `AbsoluteActions`, mask `True×26 + False×3`).
- The state is normalized and discretized into 256 bins in the prompt, the π0.5 way.
- VLM views fill `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` in order; unused slots are masked out.
- The prompt is the dataset's task string.
- The dataset's action column is `action` (`action_sequence_keys=("action",)`).

## 5. How to run

### 5.1 Environment

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

`timm==1.0.20` was added to the dependencies (for DINOv3).

### 5.2 Dataset path

openpi reads the dataset from `$HF_LEROBOT_HOME/<repo_id>`. Link to it with a symlink so the original stays untouched.

```bash
mkdir -p ~/tmp/openpi_lerobot
ln -sfn /nfs/data/sohyun/data/multiview_gr1/ColorButtonBoxV3/lerobot/gr1_mv.ColorButtonBoxV3_L3 \
        ~/tmp/openpi_lerobot/gr1_mv.ColorButtonBoxV3_L3
export HF_LEROBOT_HOME=~/tmp/openpi_lerobot
```

The dataset is in LeRobot v2.0 format; openpi's lerobot (v2.1) reads it through backward compatibility.

### 5.3 Normalization statistics

All variants share one file, `assets/pi05_gr1_colorbutton_cam3b/gr1_mv.ColorButtonBoxV3_L3/norm_stats.json` (`assets/` is gitignored).

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name pi05_gr1_colorbutton_cam3b [--max-frames N]
```

### 5.4 Convert the π0.5 base checkpoint to PyTorch

Convert with the edge-free baseline config. The edge encoders are filled with DINOv3 when training starts.

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_gr1_colorbutton_cam3b \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base/params \
    --output_path /nfs/data/sohyun/checkpoints/openpi/pi05_base_pytorch
```

The configs' default `pytorch_weight_path` is the output path above.

### 5.5 Training

```bash
export HF_LEROBOT_HOME=~/tmp/openpi_lerobot
# Single GPU
CUDA_VISIBLE_DEVICES=<gpu> uv run scripts/train_pytorch.py pi05_gr1_colorbutton_cam3b_edge_B1 --exp-name b1_run
# Multiple GPUs
CUDA_VISIBLE_DEVICES=<gpus> uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> \
    scripts/train_pytorch.py pi05_gr1_colorbutton_cam3b_edge_B1 --exp-name b1_run
```

Commonly changed options (tyro CLI):

```bash
--model.action-horizon 16          # default 50
--model.vlm-train-mode frozen      # default full
--model.edge-dropout-p 0.0         # default 0.2
--batch-size 32 --num-train-steps 30000 --save-interval 1000
--no-wandb-enabled
```

`pytorch_weight_path` loading rule: if the only keys missing from the checkpoint are `edge_vision.*`, loading proceeds and those encoders are initialized from pretrained DINOv3. Any other missing or unexpected key is an error.

## 6. Tests and verification

```bash
# Model (dummy Gemma sizes, on GPU if available): 41 tests
CUDA_VISIBLE_DEVICES=<gpu> uv run pytest src/openpi/models_pytorch/edge_test.py
# GR1 transforms: 6 tests
uv run pytest src/openpi/policies/gr1_policy_test.py
```

What `edge_test.py` checks:

| Test | Checks |
|---|---|
| `test_attention_blocks_and_positions` | For each of the 7 modes (none + 6), the attention blocks match the tables above; position ids are consecutive and the actions start at the right position |
| `test_cached_sampling_matches_training_path` | Inference with the KV cache (`sample_actions`, 2 Euler steps) matches the same steps computed in one pass through the training path |
| `test_edge_tokens_are_read_and_masked` | Edge pixels change the output, and masking them makes their effect exactly zero |
| `test_vlm_tokens_see_edges_only_in_a2_a3` | The PaliGemma outputs of valid image/text tokens depend on the edges only in A2 and A3 (with both sdpa and eager attention) |
| `test_edge_dropout_keeps_positions` | Dropping the edges leaves the position ids unchanged |
| `test_frozen_vlm_trains_edges_and_expert` | In frozen mode the VLM gets no gradients, while the edge encoders and the action expert do |

Results checked on 2026-10-26:
- All 47 tests above pass.
- `edge_placement="none"` and the original `pi0_pytorch.py` (git HEAD) give bit-for-bit identical training losses and 10-step inference results.
- `train_pytorch.py` ran for 3 steps at dummy size (A1 full, B2 frozen): real data loading → base checkpoint loading and DINOv3 initialization → training → saving, plus a strict reload of the saved checkpoint. In frozen mode the VLM tensors did not change (0/477).
- At full size (Gemma 2B + 300M, bf16, random init), all 6 scenarios ran forward/backward and 10-step inference on a real GR1 batch. Outputs were finite, and peak training memory was about 14.7 GB at batch size 2.

## 7. Known limitations and TODO

- **Video decoding is slow.** LeRobot decodes all 11 videos in the dataset for every sample (only 3 are used). Computing the normalization statistics runs at about 5 frames/s, so the full 93,731 frames would take hours. Training data loading has the same problem, so it should be fixed to decode only the needed videos before real training.
- **The normalization statistics are currently computed from a 3,200-frame sample.** Recompute them on the full dataset before real training.
- The π0.5 base checkpoint has not been converted yet (5.4).
- Edge views get no image augmentation (SigLIP views get openpi's default augmentation).
- LoRA: openpi's PyTorch implementation does not support it, so `vlm_train_mode` only offers `full`/`frozen`.
- Evaluation: there is no client yet that connects to the GR00T sim rollouts (openpi policy server ↔ sim).
- A randomly initialized DINOv3 has LayerScale 1e-5, so its CLS token barely depends on the image. The tests set gamma to 1. This does not apply to real training, which uses the pretrained weights.

## 8. Changed files

| File | Changes |
|---|---|
| `src/openpi/models_pytorch/edge_vision.py` | New. Per-view DINOv3 + MLP + LN, sample-level dropout |
| `src/openpi/models_pytorch/pi0_pytorch.py` | Edge embedding, per-scenario blocks and positions, `_predict_velocity` split out, B2 output slicing |
| `src/openpi/models_pytorch/gemma_pytorch.py` | SigLIP projection size follows the VLM width (2048 for `gemma_2b`, as before) |
| `src/openpi/models/pi0_config.py` | Edge config fields and validation |
| `src/openpi/models/model.py` | `Observation.edge_images` / `edge_image_masks` |
| `src/openpi/models/pi0.py` | The JAX model rejects edge configs |
| `src/openpi/policies/gr1_policy.py` | New. GR1 29-D selection, camera mapping |
| `src/openpi/training/config.py` | `LeRobotGR1DataConfig`, 7 ColorButton configs |
| `scripts/train_pytorch.py` | Loads base checkpoints without edge keys and initializes DINOv3; passes only trainable parameters to the optimizer |
| `src/openpi/models_pytorch/edge_test.py`, `src/openpi/policies/gr1_policy_test.py` | New tests |
