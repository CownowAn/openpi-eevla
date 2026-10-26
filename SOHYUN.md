# π0.5 + Edge-view tokens (EE-VLA on openpi)

GR00T N1.7 EE-VLA(Edge-Expert VLA)의 아이디어를 openpi의 π0.5(PyTorch 구현)에 옮긴 실험 코드입니다.
메인 뷰는 기존대로 SigLIP → PaliGemma로 들어가고, 추가 카메라(edge view)는 **뷰마다 따로 둔 DINOv3 ViT-B/16의 CLS 토큰 1개**로 요약해 시퀀스에 끼워 넣습니다. 이 토큰을 시퀀스의 어디에 넣고 누가 볼 수 있게 할지에 따라 6가지 시나리오(A1–A3, B1–B3)를 지원합니다.

- 베이스: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) (`215abfb`)
- 구현 범위: **PyTorch 모델만** 지원합니다. JAX `Pi0`에 edge 설정을 주면 `NotImplementedError`가 납니다.
- 첫 대상 데이터: GR1 ColorButtonBoxV3 L3, `cam3b` (ego + left/right wrist-inner)

---

## 1. Edge 토큰 만들기

```
edge view i (256×256 uint8) ─► DINOv3_i (뷰별, 학습됨) ─► CLS (768)
                              ─► Linear(768→D) ─► GELU ─► Linear(D→D) ─► LayerNorm ─► edge 토큰 1개
D = 2048 (A 계열: PaliGemma width) / 1024 (B 계열: 액션 전문가 width)
```

- 학습 중 **샘플 단위 edge dropout**: 확률 `edge_dropout_p`(기본 0.2)로 그 샘플의 edge 토큰을 모두 attention에서 뺍니다.
- dropout으로 빠지거나 카메라 마스크가 꺼진 edge 토큰도 **RoPE 위치는 계속 차지**합니다. 그래서 edge가 있든 없든 뒤쪽 토큰(텍스트, 액션)의 위치가 바뀌지 않습니다. attention 마스크(`pad_masks`)와 위치 마스크(`pos_masks`)를 분리해서 구현했습니다.
- DINOv3 가중치: 모델은 가중치 없이 만들고, 학습을 시작할 때 base 체크포인트에 edge 키가 없으면 `train_pytorch.py`가 timm의 사전학습 DINOv3(`vit_base_patch16_dinov3.lvd1689m`)를 불러옵니다.

## 2. 시나리오

`ar_mask` 규칙(openpi `make_attn_mask`): 토큰은 누적합이 자기보다 작거나 같은 토큰만 볼 수 있습니다. `1`은 새 블록의 시작이고, 앞쪽 블록은 이 블록을 볼 수 없습니다.

| | 시퀀스 (`\|` = prefix/suffix 경계) | edge `ar_mask` | act `ar_mask` | edge를 처리하는 가중치 |
|---|---|---|---|---|
| **A1** | `[img][text][edge]` \| `[act]` | `1 0…` | `1 0…` | PaliGemma (2048d) |
| **A2** | `[img][text][edge]` \| `[act]` | `0 0…` | `1 0…` | PaliGemma |
| **A3** | `[img][edge][text]` \| `[act]` | `0 0…` | `1 0…` | PaliGemma |
| **B1** | `[img][text]` \| `[edge][act]` | `1 0…` | `1 0…` | 액션 전문가 (1024d) |
| **B2** | `[img][text]` \| `[act][edge]` | `0 0…` | `1 0…` | 액션 전문가 |
| **B3** | `[img][text]` \| `[edge][act]` | `1 0…` | `0 0…` | 액션 전문가 |

누가 누구를 보는지 (✓ = attend 가능):

| query → key | img/text → edge | edge → img/text | edge → act | act → edge |
|---|---|---|---|---|
| A1 | ✗ | ✓ | ✗ | ✓ |
| A2, A3 | ✓ | ✓ | ✗ | ✓ |
| B1 | ✗ | ✓ | ✗ | ✓ |
| B2, B3 | ✗ | ✓ | ✓ | ✓ |

시나리오별 특징:

| | VLM이 edge를 봄 | edge가 t(adaRMS)에 의존 | 추론 시 step당 추가 비용 | 액션 RoPE 위치 | 비고 |
|---|---|---|---|---|---|
| A1 | ✗ | ✗ | 0 (prefix KV 캐시) | +V | VLM 표현은 그대로, edge는 2B를 통과 |
| A2 | ✓ | ✗ | 0 | +V | VLM이 edge 정보를 섞음 |
| A3 | ✓ | ✗ | 0 | 유지 | A2와 같고, `"Action:"`↔액션 인접 관계를 유지 |
| B1 | ✗ | ✓ | V 토큰 | +V | π0 상태 토큰과 같은 방식. EE-VLA에 가장 가까움 |
| B2 | ✗ | ✓ (+ x_t) | V 토큰 | 유지 | 출력은 suffix의 **앞쪽** 50개(`[:, :H]`) |
| B3 | ✗ | ✓ (+ x_t) | V 토큰 | +V | B2와 마스크가 같고, 위치는 B1과 같음 |

- A 계열: edge가 prefix에 있으므로 KV 캐시에 함께 들어가고, adaRMS(t)의 영향을 받지 않습니다.
- B 계열: 액션 전문가의 adaRMSNorm은 전문가 단위로 걸리므로 edge 토큰도 t 조건을 받습니다. 그래서 Euler step마다 다시 계산합니다. DINOv3는 루프 밖에서 1번만 돌립니다.
- 비교 쌍:
  - B1 ↔ B3: 위치는 같고, edge가 x_t를 보는지만 다릅니다.
  - B3 ↔ B2: 마스크는 같고, RoPE 위치만 다릅니다.
  - A1 ↔ B1: edge를 처리하는 가중치만 다릅니다(2B vs 300M+t).
  - A1 ↔ A3: VLM이 edge를 보는지가 다릅니다.

## 3. 설정 (`Pi0Config`)

| 필드 | 기본값 | 설명 |
|---|---|---|
| `edge_placement` | `"none"` | `none`, `A1`–`A3`, `B1`–`B3` |
| `edge_image_keys` | `()` | edge 뷰 이름. 뷰마다 DINOv3 하나. `edge_placement`와 함께 지정해야 함 |
| `edge_encoder_model` | `vit_base_patch16_dinov3.lvd1689m` | timm 모델 이름 |
| `edge_dropout_p` | `0.2` | 샘플 단위 edge dropout |
| `vlm_train_mode` | `"full"` | `"frozen"`이면 SigLIP + Gemma 2B를 고정. 액션 전문가와 edge 인코더는 항상 학습 |
| `action_horizon` | `50` | π0.5 기본값. CLI로 변경 가능 |

`edge_placement="none"`이면 동작이 원래 openpi와 **비트 단위로 같습니다**(아래 검증 참고).

## 4. 학습 config

`src/openpi/training/config.py`의 `_gr1_colorbutton_configs()`:

| config | VLM(SigLIP) 뷰 | edge 뷰 |
|---|---|---|
| `pi05_gr1_colorbutton_cam3b` | ego, left_wrist_inner, right_wrist_inner | – (baseline) |
| `pi05_gr1_colorbutton_cam3b_edge_{A1,A2,A3,B1,B2,B3}` | ego | left_wrist_inner, right_wrist_inner |

데이터 처리 (`policies/gr1_policy.py`, `LeRobotGR1DataConfig`):
- 상태와 액션은 44-D 중 **29-D**(left arm 7, right arm 7, left hand 6, right hand 6, waist 3)만 씁니다. GR00T GR1 modality config와 같은 순서입니다.
- 팔과 손은 현재 상태 대비 delta, 허리는 절대값입니다(`DeltaActions` / `AbsoluteActions`, 마스크 `True×26 + False×3`).
- 상태는 π0.5 방식대로 정규화한 뒤 256구간으로 나눠 프롬프트에 넣습니다.
- VLM 뷰는 `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` 순서로 채우고, 쓰지 않는 슬롯은 마스킹합니다.
- 프롬프트는 데이터셋의 task 문자열입니다.
- 데이터셋 액션 컬럼은 `action`입니다(`action_sequence_keys=("action",)`).

## 5. 실행 방법

### 5.1 환경

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

`timm==1.0.20`이 의존성에 추가되어 있습니다(DINOv3).

### 5.2 데이터셋 경로

openpi는 `$HF_LEROBOT_HOME/<repo_id>`에서 데이터셋을 읽습니다. 원본은 건드리지 않고 심볼릭 링크로 연결합니다.

```bash
mkdir -p ~/tmp/openpi_lerobot
ln -sfn /nfs/data/sohyun/data/multiview_gr1/ColorButtonBoxV3/lerobot/gr1_mv.ColorButtonBoxV3_L3 \
        ~/tmp/openpi_lerobot/gr1_mv.ColorButtonBoxV3_L3
export HF_LEROBOT_HOME=~/tmp/openpi_lerobot
```

데이터셋은 LeRobot v2.0 포맷이고, openpi의 lerobot(v2.1)이 하위 호환으로 읽습니다.

### 5.3 정규화 통계

모든 변형이 `assets/pi05_gr1_colorbutton_cam3b/gr1_mv.ColorButtonBoxV3_L3/norm_stats.json` 하나를 같이 씁니다(`assets/`는 gitignore 대상).

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name pi05_gr1_colorbutton_cam3b [--max-frames N]
```

### 5.4 π0.5 base 체크포인트를 PyTorch로 변환

edge 없는 baseline config로 변환합니다. edge 인코더는 학습을 시작할 때 DINOv3로 채워집니다.

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_gr1_colorbutton_cam3b \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base/params \
    --output_path /nfs/data/sohyun/checkpoints/openpi/pi05_base_pytorch
```

config의 `pytorch_weight_path` 기본값이 위 출력 경로입니다.

### 5.5 학습

```bash
export HF_LEROBOT_HOME=~/tmp/openpi_lerobot
# 단일 GPU
CUDA_VISIBLE_DEVICES=<gpu> uv run scripts/train_pytorch.py pi05_gr1_colorbutton_cam3b_edge_B1 --exp-name b1_run
# 멀티 GPU
CUDA_VISIBLE_DEVICES=<gpus> uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> \
    scripts/train_pytorch.py pi05_gr1_colorbutton_cam3b_edge_B1 --exp-name b1_run
```

자주 바꾸는 옵션(tyro CLI):

```bash
--model.action-horizon 16          # 기본 50
--model.vlm-train-mode frozen      # 기본 full
--model.edge-dropout-p 0.0         # 기본 0.2
--batch-size 32 --num-train-steps 30000 --save-interval 1000
--no-wandb-enabled
```

`pytorch_weight_path` 로드 규칙: 체크포인트에 없는 키가 `edge_vision.*`뿐이면 허용하고 DINOv3 사전학습 가중치로 초기화합니다. 그 밖의 키가 없거나 남으면 에러를 냅니다.

## 6. 테스트와 검증

```bash
# 모델 (dummy Gemma 크기, GPU 있으면 GPU): 41개
CUDA_VISIBLE_DEVICES=<gpu> uv run pytest src/openpi/models_pytorch/edge_test.py
# GR1 transform: 6개
uv run pytest src/openpi/policies/gr1_policy_test.py
```

`edge_test.py`가 확인하는 것:

| 테스트 | 내용 |
|---|---|
| `test_attention_blocks_and_positions` | 7가지 모드(none + 6개) 각각에서 위 표와 같은 attention 블록이 나오는지, 그리고 위치 id가 연속이고 액션 시작 위치가 맞는지 |
| `test_cached_sampling_matches_training_path` | KV 캐시를 쓰는 추론(`sample_actions`, Euler 2 step)과, 학습 경로로 한 번에 계산한 결과가 일치하는지 |
| `test_edge_tokens_are_read_and_masked` | edge 픽셀이 출력을 바꾸는지, 그리고 마스킹하면 영향이 정확히 0인지 |
| `test_vlm_tokens_see_edges_only_in_a2_a3` | 유효한 이미지·텍스트 토큰의 PaliGemma 출력이 A2·A3에서만 edge에 의존하는지 (sdpa와 eager 모두) |
| `test_edge_dropout_keeps_positions` | dropout으로 edge를 빼도 위치 id가 같은지 |
| `test_frozen_vlm_trains_edges_and_expert` | frozen 모드에서 VLM에는 grad가 없고, edge 인코더와 액션 전문가에는 grad가 있는지 |

2026-10-26에 확인한 결과:
- 위 테스트 47개가 모두 통과했습니다.
- `edge_placement="none"`과 원래 `pi0_pytorch.py`(git HEAD)의 학습 loss와 10 step 추론 결과가 비트 단위로 같습니다.
- dummy 크기로 `train_pytorch.py`를 3 step 돌렸습니다(A1 full, B2 frozen). 실제 데이터 로드 → base 체크포인트 로드와 DINOv3 초기화 → 학습 → 저장, 그리고 저장한 체크포인트의 strict 재로드까지 확인했습니다. frozen에서는 VLM 텐서가 바뀌지 않았습니다(0/477).
- 실제 크기(Gemma 2B + 300M, bf16, 무작위 초기화)로 6개 시나리오 모두 실제 GR1 배치에서 forward/backward와 10 step 추론을 돌렸습니다. 결과는 유한값이었고, 학습 피크 메모리는 batch 2에서 약 14.7GB였습니다.

## 7. 알려진 한계와 TODO

- **영상 디코딩이 느립니다.** LeRobot이 샘플마다 데이터셋의 영상 11개를 모두 디코딩합니다(쓰는 건 3개). 정규화 통계 계산이 약 5 프레임/초라, 전체 93,731 프레임이면 몇 시간이 걸립니다. 학습 데이터 로딩도 같은 영향을 받으므로, 본 학습 전에 필요한 영상만 디코딩하도록 고쳐야 합니다.
- **정규화 통계는 현재 3,200 프레임 샘플로 계산한 것입니다.** 본 학습 전에 전체 데이터로 다시 계산해야 합니다.
- π0.5 base 체크포인트는 아직 변환하지 않았습니다(5.4).
- edge 뷰에는 이미지 augmentation이 없습니다(SigLIP 뷰에는 openpi 기본 augmentation이 적용됨).
- LoRA: openpi PyTorch 구현이 지원하지 않습니다. `vlm_train_mode`는 `full`/`frozen`만 있습니다.
- 평가: GR00T sim 롤아웃과 연결하는 클라이언트(openpi policy server ↔ sim)는 아직 없습니다.
- 무작위 초기화한 DINOv3는 LayerScale이 1e-5라 CLS가 이미지에 거의 무관합니다. 테스트에서는 gamma를 1로 바꿔 씁니다. 실제 학습은 사전학습 가중치를 쓰므로 해당되지 않습니다.

## 8. 바뀐 파일

| 파일 | 내용 |
|---|---|
| `src/openpi/models_pytorch/edge_vision.py` | 신규. 뷰별 DINOv3 + MLP + LN, 샘플 단위 dropout |
| `src/openpi/models_pytorch/pi0_pytorch.py` | edge 임베딩, 시나리오별 블록/위치 처리, `_predict_velocity` 분리, B2 출력 슬라이싱 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | SigLIP projection 크기를 VLM width에 맞춤 (`gemma_2b`에서는 원래와 같은 2048) |
| `src/openpi/models/pi0_config.py` | edge 설정 필드와 검증 |
| `src/openpi/models/model.py` | `Observation.edge_images` / `edge_image_masks` |
| `src/openpi/models/pi0.py` | JAX 모델에서 edge 설정을 거부 |
| `src/openpi/policies/gr1_policy.py` | 신규. GR1 29-D 선택, 카메라 매핑 |
| `src/openpi/training/config.py` | `LeRobotGR1DataConfig`, ColorButton config 7개 |
| `scripts/train_pytorch.py` | edge 키가 없는 base 체크포인트 로드와 DINOv3 초기화, 학습 대상 파라미터만 optimizer에 전달 |
| `src/openpi/models_pytorch/edge_test.py`, `src/openpi/policies/gr1_policy_test.py` | 신규 테스트 |
