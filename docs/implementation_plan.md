# CertVLA Implementation Plan

> Phase 0 output. This document is the result of a code-audit-only planning stage.
> No code has been written or modified.

---

## 1. Project Understanding Summary

### Core Insight

Existing VLAs (Vision-Language-Action models) are fragile on long-horizon tasks not because they lack a planner, verifier, or memory bank, but because their internal state is never *defined* to be responsible for the structured local consequences of the next action chunk.

CertVLA redefines the VLA internal state as **the minimal state sufficient to certify the structured consequence of the next action chunk**, and builds a closed loop of state -> certificate -> action -> gap -> repair around this single redefined object.

### Key Mathematical Objects

| Symbol | Name | Role |
|--------|------|------|
| `z_t` | Recursive task state | Single persistent token, gated-updated each chunk |
| `s_t` | State readout | `R_phi(z_t)` -- slot-structured readout, **must** read only from `z_t` |
| `c_t` | Local consequence certificate | `Q_psi(z_t)` -- per-slot (role, goal_value) pairs |
| `A_t` | Action chunk | `pi(z_t, c_t, o_t)` -- conditioned on state AND certificate |
| `Gamma_t` | Certificate gap | Difference between predicted and observed post-execution state |

### Hard No-Go Lanes

- No planner + verifier + memory system assembly
- No full symbolic / PDDL planning
- No natural-language certificates
- No multi-state-token memory bank as protagonist
- No multi-view/proprio/wrist "all-in-one" v1
- No from-scratch pretraining

---

## 2. Current Codebase Audit Results

### 2.1 Repository Layout

```
openvla-oft/
  prismatic/                    # Core library
    extern/hf/
      modeling_prismatic.py     # THE model: PrismaticForConditionalGeneration, OpenVLAForActionPrediction
      configuration_prismatic.py
      processing_prismatic.py
    models/
      action_heads.py           # L1RegressionActionHead, DiffusionActionHead, MLPResNet
      projectors.py             # ProprioProjector, NoisyActionProjector
      film_vit_wrapper.py       # FiLM language-visual conditioning
      vlms/prismatic.py         # Internal VLM (not used in HF path)
      vlas/openvla.py           # Internal VLA (not used in HF path)
      backbones/                # Vision (SigLIP+DINOv2) and LLM (Llama2) backbones
    vla/
      constants.py              # Platform-specific constants (auto-detected: LIBERO/ALOHA/BRIDGE)
      action_tokenizer.py       # 256-bin discretization
      datasets/
        datasets.py             # RLDSBatchTransform, RLDSDataset, EpisodicRLDSDataset
        rlds/                   # TF RLDS pipeline, OXE mixtures
    training/
      train_utils.py            # Loss helpers, mask computation
    util/
      data_utils.py             # PaddedCollatorForActionPrediction
  vla-scripts/
    finetune.py                 # MAIN training entry (draccus config, LoRA, DDP)
    deploy.py                   # FastAPI inference server
    merge_lora_weights_and_save.py
  experiments/
    robot/
      libero/
        run_libero_eval.py      # LIBERO benchmark evaluation
        libero_utils.py         # Env wrappers, video saving
      aloha/                    # Real robot eval
      openvla_utils.py          # Shared: get_model, get_action_head, etc.
      robot_utils.py            # Shared: get_action, set_seed, etc.
  LIBERO/                       # LIBERO benchmark subproject (separate setup.py)
    libero/libero/envs/
      predicates/               # In, On, Open, Close, Stack, InContact, TurnOn/Off
      object_states/            # ObjectState: pos, quat, contact, contain, joint, ontop
  docs/
  scripts/                      # HF weight conversion
```

### 2.2 Training Entry Point

**`vla-scripts/finetune.py`** -- single file, ~1140 lines.

- Config: `FinetuneConfig` dataclass (draccus)
- Model loading: `AutoModelForVision2Seq.from_pretrained()` -> `OpenVLAForActionPrediction`
- LoRA: `peft.get_peft_model()` on all linear layers
- Auxiliary modules (separately DDP-wrapped): `L1RegressionActionHead` or `DiffusionActionHead`, `ProprioProjector`, `NoisyActionProjector`
- Training loop: iterates RLDS DataLoader, calls `run_forward_pass()`, AdamW + MultiStepLR
- Key function: `run_forward_pass()` (line 269) -- VLA forward, extract `actions_hidden_states`, compute loss

### 2.3 Data Loading

- `RLDSDataset` (IterableDataset) wraps TF RLDS pipeline
- `RLDSBatchTransform` converts RLDS batches to model format
- Each sample: `(pixel_values, input_ids, labels, actions)` where actions = chunk of `NUM_ACTIONS_CHUNK` steps
- RLDS config: `window_size=1`, `future_action_window_size=NUM_ACTIONS_CHUNK-1`
- No episode-level metadata exposed (no sim state, no object positions)

### 2.3.1 CRITICAL: Dataset is RLDS on HuggingFace, NOT HDF5

The training dataset is **`openvla/modified_libero_rlds`** on HuggingFace, downloaded via:
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

The RLDS dataset contains **ONLY**:
| Field | Details |
|-------|---------|
| `image` (agentview RGB) | 256x256 RGB, rotated 180 degrees |
| `wrist_image` (eye-in-hand RGB) | 256x256 RGB, rotated 180 degrees |
| `action` | 7-dim: EEF delta (6) + gripper (1) |
| `state` (proprioceptive) | 8-dim: EEF pose (6) + gripper qpos (2) |
| `language_instruction` | Task description string |

**NOT included**: full simulator state, object positions, contact info, joint states beyond EEF.

The intermediate HDF5 files (created by `experiments/robot/libero/regenerate_libero_dataset.py`)
DO contain full sim state (`env.sim.get_state().flatten()`), `robot_states`, `joint_states`, etc.
But the HDF5-to-RLDS conversion (not included in this repo) stripped this down to only
image + action + compact proprio + language.

**Implication for Phase 1**: Oracle slot labels CANNOT be extracted from the RLDS dataset directly.
The labeling pipeline must work by one of two strategies:

1. **Replay strategy**: Use the original HDF5 files (or re-generate them via `regenerate_libero_dataset.py`)
   to replay episodes in LIBERO simulator, extract sim state at each step, compute slot labels,
   and save labels as a sidecar file alongside the RLDS dataset.

2. **Re-generation strategy**: Modify `regenerate_libero_dataset.py` to also extract and save slot
   labels during replay, then include them in a custom RLDS conversion.

Strategy 1 (sidecar files) is recommended for Phase 1 because it requires zero changes to the
existing RLDS pipeline. The sidecar labels are loaded separately and joined to RLDS batches
by episode ID + timestep index during training.

### 2.4 Forward Path (Training)

1. Image -> dual ViT (SigLIP+DINOv2) -> projector -> `projected_patch_embeddings`
2. Optionally append proprio token and diffusion timestep token
3. Build sequence: `[BOS][vision_patches][prompt_tokens + action_tokens][STOP]`
4. Action token embeddings zeroed (L1) or replaced with noisy actions (diffusion)
5. Full LLM forward (Llama 2 7B with bidirectional attention on action region)
6. Extract `actions_hidden_states` from action-token positions
7. Pass through action head -> predicted actions -> L1 loss

### 2.5 Forward Path (Inference)

`OpenVLAForActionPrediction.predict_action()` (line 944):
- Appends placeholder action tokens + stop token
- Creates fake labels for mask computation
- Single forward pass -> extract action hidden states -> action head -> unnormalize

### 2.6 Configuration System

- Pure Python dataclass + `draccus` (CLI override) -- **no YAML** for main configs
- Platform constants auto-detected from `sys.argv` in `constants.py`

### 2.7 Test Infrastructure

**None.** No test files, no tests/ directory, no pytest config. Validation is done only via simulation rollouts in `run_libero_eval.py`.

### 2.8 LIBERO Environment Access (via Sim Replay, NOT via RLDS)

LIBERO provides rich oracle state via `env.sim.data` **when running the simulator**:
- Object positions: `env.sim.data.body_xpos[obj_body_id]`
- Object quaternions: `env.sim.data.body_xquat[obj_body_id]`
- Joint states: `env.sim.data.qpos[joint_addr]`
- Contact checking: `env.check_contact(obj1, obj2)`
- Containment checking: `obj.in_box(pos1, pos2)`
- On-top checking: `obj.check_ontop(other)`
- Predicate functions: `In`, `On`, `Open`, `Close`, `Stack`, `InContact`, `TurnOn`, `TurnOff`
- Robot EE state: from `obs["robot0_eef_pos"]`, `obs["robot0_eef_quat"]`, `obs["robot0_gripper_qpos"]`

**However, this state is NOT in the RLDS training dataset.** The `openvla/modified_libero_rlds`
HuggingFace dataset only contains images + actions + 8-dim proprio + language. To obtain oracle
slot labels, we must **replay episodes in the LIBERO simulator** using either:
- The original raw HDF5 files + `regenerate_libero_dataset.py`, OR
- The regenerated HDF5 files that are produced as an intermediate step

The replay script (`regenerate_libero_dataset.py`) does save `env.sim.get_state().flatten()` (full
MuJoCo state) and `robot_states`, `joint_states` per step in the intermediate HDF5. From there,
slot labels can be computed by running LIBERO env predicates on the replayed states.

This makes the labeling pipeline a **two-stage offline process**:
1. Replay demos in simulator -> extract full state at each timestep
2. Compute slot labels from full state -> save as sidecar files (e.g. `.npz` per episode)

---

## 3. Recommended Directory Structure

Minimal-invasion principle: add a `certvla/` top-level package, a `configs/certvla/` directory, and a `tests/` directory. Touch the base code as little as possible.

### New Files to Create

```
certvla/                          # NEW top-level package
  __init__.py
  slots/
    __init__.py
    schema.py                     # SlotVocab enum, SlotMeta dataclass, SLOT_REGISTRY
    metrics.py                    # Per-slot distance fns d_j, value coercion
    role_sets.py                  # J_E, J_R, J_C family definitions
    preserve_rules.py             # Latch-preserve + support-preserve logic
  data/
    __init__.py
    chunk_sample.py               # ChunkSample / CertChunkSample dataclasses
    state_labels.py               # Oracle label extractor interface + LIBERO impl (requires sim)
    certificate_mining.py         # advance / preserve / ignore mining from episodes
    goal_signature.py             # Terminal goal signature aggregation
    label_episodes.py             # Offline script: replay HDF5 -> compute labels -> save sidecar .npz
    # counterfactuals.py          # Phase 3+ (not Phase 1)
    # parser_interfaces.py        # Phase 3+ (not Phase 1)

configs/
  certvla/                        # NEW config sub-directory (YAML or dataclass)
    slots_v1.py                   # Frozen v1 slot vocabulary config

tests/                            # NEW top-level test directory
  __init__.py
  conftest.py                     # Shared fixtures
  test_slot_schema.py
  test_slot_metrics.py
  test_certificate_mining.py
  test_preserve_rules.py
  test_goal_signature.py
```

### Existing Files That Must Be Modified (Later Phases)

| File | When | What |
|------|------|------|
| `prismatic/extern/hf/modeling_prismatic.py` | Phase 2 | Add state token to sequence, modify `forward()` |
| `prismatic/models/action_heads.py` | Phase 2 | Add `CertActionHead` (coarse+fine, cert-conditioned) |
| `prismatic/vla/datasets/datasets.py` | Phase 2-3 | Extend `RLDSBatchTransform` to include slot labels |
| `prismatic/util/data_utils.py` | Phase 2-3 | Extend collator for cert fields |
| `vla-scripts/finetune.py` | Phase 3 | Integrate cert losses, curriculum, new heads |
| `prismatic/vla/constants.py` | Phase 2 | Add CertVLA constants (NUM_SLOTS, etc.) |
| `experiments/robot/libero/run_libero_eval.py` | Phase 4 | Add gap computation, repair loop |

### Existing Files NOT To Touch

| File | Reason |
|------|--------|
| `prismatic/models/backbones/` | Vision and LLM backbones unchanged |
| `prismatic/extern/hf/configuration_prismatic.py` | Config schema stable until Phase 5 |
| `prismatic/vla/datasets/rlds/` | RLDS pipeline unchanged; cert data layered on top |
| `prismatic/models/film_vit_wrapper.py` | FiLM is orthogonal |
| `prismatic/models/vlms/`, `prismatic/models/vlas/` | Internal model path not used |
| `LIBERO/libero/` | Upstream benchmark code, read-only |

---

## 4. Phased Implementation Blueprint

### Phase 0: Planning & Repo Audit (CURRENT -- COMPLETE)

**Goal**: Understand codebase, produce this plan.
**Output**: `docs/implementation_plan.md` (this document).
**Status**: Done.

---

### Phase 1: Data & Labeling Layer

**Goal**: Implement slot schema, oracle labeling (via sim replay), certificate mining, and preserve rules. All pure-data, no model changes.

**Input Dependencies**:
- LIBERO environment installed (`pip install -e LIBERO`)
- Original LIBERO raw HDF5 demo files (downloaded per LIBERO benchmark instructions) OR
  the regenerated HDF5 files from `regenerate_libero_dataset.py`
- The RLDS dataset `openvla/modified_libero_rlds` is NOT directly needed for Phase 1
  (Phase 1 only builds the labeling pipeline; RLDS integration happens in Phase 3)
- No model weights needed

**Output Products**:
1. `certvla/slots/schema.py` -- frozen v1 slot vocabulary
2. `certvla/slots/metrics.py` -- per-slot distance functions
3. `certvla/slots/role_sets.py` -- J_E / J_R / J_C families
4. `certvla/slots/preserve_rules.py` -- latch-preserve + support-preserve
5. `certvla/data/chunk_sample.py` -- data structures for labeled chunks
6. `certvla/data/state_labels.py` -- LIBERO oracle label extractor (requires sim replay)
7. `certvla/data/certificate_mining.py` -- advance/preserve/ignore labeler
8. `certvla/data/goal_signature.py` -- episode-level goal signature
9. `certvla/data/label_episodes.py` -- offline script: replay HDF5 in sim, compute labels, save sidecar files
10. Unit tests for all of the above

**Files to Create/Modify**:
- Create all files listed above
- No modification to any existing file

**Minimal Tests**:
- `test_slot_schema.py`: schema loads, enums correct, value domains enforced
- `test_slot_metrics.py`: distance functions return correct types, handle edge cases
- `test_certificate_mining.py`: given synthetic episode with known state transitions, verify advance/preserve/ignore labels match expectations
- `test_preserve_rules.py`: given known advance set + latch states, verify preserve set is correct
- `test_goal_signature.py`: aggregation over terminal steps produces correct signature

**Risks**:
- Replaying HDF5 demos in LIBERO requires MuJoCo + LIBERO env to be properly installed
  (this is a GPU/Linux-only dependency; cannot be done on a machine without a display or MuJoCo license)
- The raw LIBERO HDF5 files must be separately downloaded; they are NOT the same as the RLDS dataset
- Oracle label extractor needs a running LIBERO env to query `sim.data` -> labels must be pre-computed offline via `label_episodes.py`, stored as sidecar files, NOT computed on-the-fly during training
- Thresholds (tau_delta, tau_rho, tau_upsilon) are hyperparameters that may need tuning; start with conservative values
- The sidecar label files must later (Phase 3) be aligned with the RLDS dataset by episode ID + timestep index; this alignment mechanism needs careful design

---

### Phase 2: Model Layer

**Goal**: Add persistent state token, state readout head, certificate head, and cert-conditioned action head to the model.

**Input Dependencies**:
- Phase 1 complete (slot schema, data structures)
- Pre-trained OpenVLA-OFT weights

**Output Products**:
1. `certvla/model/state_token.py` -- state token embedding + gated update
2. `certvla/model/state_readout.py` -- R_phi: z_t -> s_t (per-slot readout)
3. `certvla/model/certificate_head.py` -- Q_psi: z_t -> c_t (role classification + goal prediction)
4. `certvla/model/action_head.py` -- coarse + fine action head, conditioned on c_t
5. `certvla/model/certvla_wrapper.py` -- wrapper that composes all heads around base VLA
6. Shape/smoke tests

**Files to Create/Modify**:
- Create all files listed above
- Modify `prismatic/extern/hf/modeling_prismatic.py`: inject state token into sequence, return hidden state at state-token position
- Modify `prismatic/vla/constants.py`: add CERTVLA constants
- Modify `prismatic/util/data_utils.py`: extend collator for cert fields

**Minimal Tests**:
- `test_model_shapes.py`: given dummy input, verify all head outputs have correct shapes
- Verify state token survives through forward pass
- Verify readout/cert heads produce outputs only from z_t position hidden state (no vision bypass)

**Risks**:
- Inserting a token into the LLM sequence changes all positional encodings downstream -- must choose insertion point carefully (after vision patches, before text? or after text, before actions?)
- The bidirectional attention region for action tokens is hard-coded; must ensure state token is not in the bidirectional region
- Gated update requires `z_{t-1}` from previous chunk -- at training time each sample is independent (no temporal context); need to decide: (a) train with `z_0 = learned_init` for every sample, (b) build mini-episode batches. v1 should start with (a).

---

### Phase 3: Training Layer

**Goal**: Implement all losses, training curriculum, scheduled sampling, and integrate with `finetune.py`.

**Input Dependencies**:
- Phase 1 + Phase 2 complete
- Pre-computed sidecar label files (from Phase 1 offline labeling script)
- The RLDS dataset `openvla/modified_libero_rlds`

**Output Products**:
1. `certvla/training/losses.py` -- L_state, L_role, L_goal, L_act, L_cons, L_dep, L_cf
2. `certvla/training/curriculum.py` -- staged loss weight scheduling
3. `certvla/training/sched_sampling.py` -- scheduled sampling for state token
4. `certvla/training/batch_types.py` -- typed batch dict for CertVLA
5. `certvla/data/rlds_cert_dataset.py` -- custom dataset that joins RLDS batches with sidecar labels
6. Config dataclass extensions for cert training
7. Modified `finetune.py` with cert training path

**Files to Create/Modify**:
- Create all files listed above
- Modify `vla-scripts/finetune.py`: add cert training branch (controlled by config flag)
- Modify `prismatic/vla/datasets/datasets.py`: extend batch transform to include cert labels

**Minimal Tests**:
- `test_losses.py`: each loss returns scalar, correct gradient flow, handles edge cases (no advance slots, all ignore, etc.)
- Smoke training: 10 steps, verify loss decreases
- Verify L_dep negative pair actually produces higher action error

**Risks**:
- 7 loss terms with weight scheduling is fragile; must implement careful weight defaults and validation
- Certificate dependence loss requires two forward passes (positive and negative cert); may double memory
- Counterfactual loss requires augmented image pairs; defer to Phase 3b or Phase 4

---

### Phase 4: Inference Loop

**Goal**: Implement certificate gap computation, threshold-based drift detection, and short-horizon local repair.

**Input Dependencies**:
- Phase 1-3 complete
- Trained CertVLA checkpoint

**Output Products**:
1. `certvla/inference/gap.py` -- probability-weighted gap computation
2. `certvla/inference/thresholding.py` -- adaptive threshold logic
3. `certvla/inference/repair.py` -- short-horizon repair action generation
4. `certvla/inference/rollout.py` -- full eval loop with gap + repair
5. Modified `run_libero_eval.py` with cert gap and repair

**Files to Create/Modify**:
- Create all files listed above
- Modify `experiments/robot/libero/run_libero_eval.py`: add cert inference mode

**Minimal Tests**:
- `test_gap.py`: given known predicted and observed states, verify gap computation
- `test_thresholding.py`: verify threshold triggers repair correctly
- End-to-end rollout smoke test in LIBERO

**Risks**:
- Repair re-prediction doubles inference cost per chunk; must keep repair horizon short
- Gap threshold tau(c_t) is task-dependent; start with a single global threshold

---

### Phase 5: Integration & Cleanup

**Goal**: End-to-end training + evaluation pipeline, documentation, configs.

**Input Dependencies**:
- All previous phases complete

**Output Products**:
1. Unified run scripts
2. YAML/dataclass configs for all training stages
3. `docs/progress.md` updated
4. Minimal demo script
5. Experiment tracking setup (WandB integration for cert metrics)

**Files to Create/Modify**:
- Create run scripts
- Final config consolidation
- Documentation updates

**Risks**:
- Integration bugs between phases; mitigated by per-phase tests

---

## 5. High-Risk Design Points & Mitigations

### Risk 1: State readout bypasses z_t via vision tokens

**Description**: If the readout head R_phi can access the full LLM hidden state (which fuses vision+text+action), then z_t becomes decorative. The readout must operate *only* on the hidden state at the z_t token position.

**Mitigation**:
- Architecturally enforce: readout/cert heads take `h[z_pos]` (a single vector), not the full sequence hidden states.
- Add a unit test that verifies readout gradients flow only through `z_t` position.
- Never pass full `last_hidden_states` to readout -- only the scalar slice `last_hidden_states[:, z_pos, :]`.

### Risk 2: Preserve rules become pure data mining

**Description**: If `preserve` labels are mined purely from statistics (like "didn't change much"), they become noisy and miss the structural requirement. The context doc is explicit: **advance is mined from data, preserve is defined by structural priors**.

**Mitigation**:
- Implement preserve as explicit rule functions in `preserve_rules.py`, not as statistical thresholds.
- Latch-preserve: any completed result slot not in current advance set.
- Support-preserve: explicit rule table (e.g., "during transport, hand_occupancy must be preserved").
- Rules are declared, not learned. Test them against known manipulation scenarios.

### Risk 3: Action head ignores certificate

**Description**: If the action head can predict good actions from `z_t + o_t` alone, it will learn to ignore `c_t`. This is why L_dep (certificate dependence loss) is mandatory.

**Mitigation**:
- Implement L_dep from Phase 3 day one -- not as an afterthought.
- Architecture: coarse action branch takes `z_t + c_t` WITHOUT `o_t`; only the fine residual branch sees `o_t`.
- Monitor L_dep margin during training; if margin collapses, the action head is ignoring cert.

### Risk 4: Chunk / label definition instability

**Description**: "What counts as a chunk" and "how state labels align to chunk boundaries" can silently break training if not carefully defined.

**Mitigation**:
- Use the existing `NUM_ACTIONS_CHUNK = 8` (LIBERO) as the chunk boundary.
- State labels are at chunk boundaries: `s_t` at chunk start, `s_{t+H}` at chunk end.
- Offline label generation pre-computes labels for all chunk boundaries in an episode.
- Unit test: for a known episode, verify label timestamps align with action chunk boundaries.

### Risk 5: Over-generalization in data layer

**Description**: Building a "universal slot parser" that handles LIBERO + real robot + arbitrary tasks from day one makes Phase 1 unshippable.

**Mitigation**:
- Phase 1 targets LIBERO-only with oracle sim state via replay.
- Interface is abstract (`StateLabeler` base class), but only `LiberoOracleLabeler` is implemented.
- Real-robot pseudo-label parser is explicitly deferred to Phase 3+.

### Risk 5b: RLDS dataset does not contain sim state -- sidecar alignment is fragile

**Description**: The HuggingFace RLDS dataset (`openvla/modified_libero_rlds`) contains only images, actions, 8-dim proprio, and language strings. Slot labels must be computed offline by replaying HDF5 demos in the LIBERO simulator, then stored as sidecar files. At training time, these sidecar labels must be correctly joined to RLDS batches by episode ID and timestep index. This alignment is fragile because:
- The RLDS pipeline applies no-op filtering and shuffling
- Timestep indices in RLDS may not directly correspond to HDF5 demo indices
- The `regenerate_libero_dataset.py` script already filters no-op actions, so the regenerated HDF5 should align with the RLDS data, but this must be verified

**Mitigation**:
- Phase 1 sidecar labels are built from the **regenerated** HDF5 files (which already have no-ops removed), not from the raw LIBERO demos. This ensures timestep alignment.
- Add a verification step in `label_episodes.py` that checks the number of steps per episode matches between the HDF5 and the sidecar file.
- At training time (Phase 3), build a lookup table mapping `(dataset_name, episode_idx, timestep_idx)` to sidecar label files at dataset initialization, not per-batch.
- If alignment proves unreliable, fallback strategy: produce a complete custom RLDS dataset that includes slot labels as additional observation keys (this would be Phase 3 scope).

### Risk 6: First version becomes a multi-modal kitchen sink

**Description**: Temptation to support wrist cameras, proprio, multi-image, multi-state-token, etc. in v1.

**Mitigation**:
- v1: single RGB image, no proprio in cert heads, single state token, single persistent state.
- All multi-modal extensions are explicitly Phase 5+ or v2.

### Risk 7: State token initialization at training time

**Description**: Each training sample is an independent (o_t, l, A_t) chunk, not part of an episode sequence. The recursive `z_t = Phi(z_{t-1}, o_t, l)` needs `z_{t-1}`, but training samples are shuffled.

**Mitigation**:
- v1: use a learnable `z_0` as initial state for every training sample. The gated update `z_t = g_t * f(o_t, l, z_0) + (1 - g_t) * z_0` still provides a meaningful residual structure.
- Episode-sequential training (feeding multi-chunk sequences) is a v2 enhancement.
- This is a known limitation of v1 and should be documented.

---

## 6. Phase 1 Coding Task List

Phase 1 scope: **Data & labeling layer only.** No model code, no training changes, no inference changes.

### Task 1.1: Create package skeleton

Create directory structure and `__init__.py` files:
```
certvla/__init__.py
certvla/slots/__init__.py
certvla/data/__init__.py
tests/__init__.py
tests/conftest.py
```

### Task 1.2: Implement slot schema (`certvla/slots/schema.py`)

Define the v1 slot vocabulary:

```python
# Contents:
# - SlotName enum with 10 slots (ee_target_proximity, hand_occupancy, ...)
# - SlotDomain enum: BINARY, CATEGORICAL, CONTINUOUS, CONFIDENCE
# - SlotFamily enum: ENABLING, RESULT, CONFIDENCE
# - SlotMeta dataclass: name, domain, family, categories (if categorical), valid_range
# - SLOT_REGISTRY: Dict[SlotName, SlotMeta] -- the frozen v1 vocabulary
# - SLOT_VOCAB_SIZE: int = 10
# - function get_slot_meta(name: SlotName) -> SlotMeta
```

Specific slots to define (per context doc section 5.3):

| Slot | Domain | Family |
|------|--------|--------|
| ee_target_proximity | continuous [0,1] | ENABLING |
| hand_occupancy | categorical {empty, target, other} | ENABLING |
| target_contact | binary | ENABLING |
| target_goal_proximity | continuous [0,1] | RESULT |
| support_relation | categorical {none, on_goal, on_other} | RESULT |
| containment_relation | categorical {none, in_goal, in_other} | RESULT |
| articulation_progress | continuous [0,1] | ENABLING |
| orientation_alignment | continuous [0,1] | ENABLING |
| completion_latch | binary | RESULT |
| task_visible_confidence | continuous [0,1] | CONFIDENCE |

### Task 1.3: Implement slot metrics (`certvla/slots/metrics.py`)

Per-slot distance function `d_j(a, b)`:
- Binary slots: `|a - b|` (0 or 1)
- Categorical slots: `1 - (a == b)` (0-1 Hamming)
- Continuous slots: `|a - b|` (L1 in [0,1])

Also implement:
- `slot_value_to_tensor(slot_meta, value)` -- convert raw value to tensor representation
- `tensor_to_slot_value(slot_meta, tensor)` -- inverse

### Task 1.4: Implement role sets (`certvla/slots/role_sets.py`)

Define:
- `J_E`: set of enabling slots = {ee_target_proximity, hand_occupancy, target_contact, articulation_progress, orientation_alignment}
- `J_R`: set of result slots = {target_goal_proximity, support_relation, containment_relation, completion_latch}
- `J_C`: set of confidence slots = {task_visible_confidence}

Also define:
- `J_cert = J_E | J_R` (the slots that participate in certificates)
- `get_family(slot_name) -> SlotFamily`

### Task 1.5: Implement preserve rules (`certvla/slots/preserve_rules.py`)

Implement the two structural preserve mechanisms:

**Latch-preserve**:
```python
def latch_preserve(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Slots where completion_latch=1 and not in advance set."""
    result = set()
    for j in J_R:
        if state.get("completion_latch", 0) == 1 and j not in advance_set:
            result.add(j)
    return result
```

**Support-preserve** (rule table):
```python
# Example rules:
# - If advance includes target_goal_proximity (transport phase), preserve hand_occupancy == target
# - If advance includes containment_relation, preserve articulation_progress (keep container open)
# - If advance includes support_relation, preserve target_contact

def support_preserve(state: SlotStateDict, advance_set: Set[SlotName]) -> Set[SlotName]:
    """Structural rules for which enabling conditions must hold during current advance."""
    ...
```

**Combined**:
```python
def compute_preserve_set(state, advance_set) -> Set[SlotName]:
    return latch_preserve(state, advance_set) | support_preserve(state, advance_set)
```

### Task 1.6: Implement chunk sample dataclasses (`certvla/data/chunk_sample.py`)

```python
@dataclass
class SlotState:
    """State values for all slots at one timestep."""
    values: Dict[SlotName, Union[float, int, str]]
    validity_mask: Dict[SlotName, bool]
    confidence: Dict[SlotName, float]

@dataclass
class CertificateLabel:
    """Certificate label for one chunk."""
    roles: Dict[SlotName, str]         # "advance" | "preserve" | "ignore"
    goal_values: Dict[SlotName, Any]   # only for advance slots: s_{t+H}^j

@dataclass
class CertChunkSample:
    """A fully labeled chunk sample for CertVLA training."""
    observation: np.ndarray            # RGB image at t
    instruction: str                   # language instruction
    actions: np.ndarray                # (H, action_dim) action chunk
    state_t: SlotState                 # state at chunk start
    state_t_H: SlotState              # state at chunk end
    certificate: CertificateLabel      # mined certificate
    goal_signature: Optional[SlotState]  # episode-level goal signature
```

### Task 1.7: Implement LIBERO oracle state labeler (`certvla/data/state_labels.py`)

**IMPORTANT**: The training dataset `openvla/modified_libero_rlds` on HuggingFace does NOT contain
simulator state. It only has images + 7-dim actions + 8-dim proprio (EEF pose + gripper qpos) +
language_instruction. Oracle slot labels must be obtained by **replaying episodes in the LIBERO
simulator** using the original HDF5 demo files.

The labeler has two layers:

**Layer 1: Abstract interface**
```python
class StateLabeler(ABC):
    """Abstract interface for extracting slot state labels."""
    @abstractmethod
    def extract_state(self, env, obs, instruction: str) -> SlotState:
        ...
```

**Layer 2: LIBERO oracle implementation (requires running simulator)**
```python
class LiberoOracleLabeler(StateLabeler):
    """Extracts slot labels from LIBERO sim state during episode replay."""
    def extract_state(self, env, obs, instruction: str) -> SlotState:
        # Uses env.sim.data to compute:
        # - ee_target_proximity: distance(ee_pos, target_pos) normalized
        # - hand_occupancy: check gripper contact + identify object
        # - target_contact: env.check_contact(gripper, target)
        # - target_goal_proximity: distance(target_pos, goal_pos) normalized
        # - support_relation: On predicate checks
        # - containment_relation: In predicate checks
        # - articulation_progress: joint_state normalized
        # - orientation_alignment: quaternion alignment
        # - completion_latch: derived from predicate persistence
        # - task_visible_confidence: 1.0 (always visible in sim)
        ...
```

**Important design note**: The labeler requires env.sim access, which means it can only run
during offline replay (not during training). The `instruction` parameter is needed to resolve
role bindings (which object is the "target", which is the "goal receptacle", etc.).

### Task 1.7b: Implement offline labeling script (`certvla/data/label_episodes.py`)

This script is the bridge between the HDF5 demo files and the sidecar label files:

```python
def label_all_episodes(
    libero_task_suite: str,       # e.g. "libero_spatial"
    raw_hdf5_dir: str,            # path to regenerated HDF5 files
    output_label_dir: str,        # path to save sidecar .npz files
    chunk_size: int = 8,          # NUM_ACTIONS_CHUNK
):
    """
    For each episode in the HDF5 dataset:
    1. Load initial state + actions from HDF5
    2. Replay in LIBERO simulator
    3. At each timestep, call LiberoOracleLabeler.extract_state()
    4. After full replay, compute goal_signature from terminal states
    5. At each chunk boundary, call mine_certificate()
    6. Save per-episode labels as .npz: {
         "slot_states": array of shape (T, num_slots),  # per-timestep slot values
         "validity_masks": ...,
         "chunk_certificates": ...,  # per-chunk advance/preserve/ignore
         "goal_signature": ...,
       }
    """
```

**Sidecar file naming convention**: `{task_name}_demo_{i}_labels.npz` matching
the HDF5 naming `{task_name}_demo.hdf5 / demo_{i}`.

This script will need:
- The regenerated HDF5 files (from `regenerate_libero_dataset.py`)
- A working LIBERO + MuJoCo installation
- The slot schema + labeler + certificate miner from earlier tasks

The output sidecar files are independent of the RLDS dataset. In Phase 3, a custom dataset
wrapper will load sidecar labels and join them to RLDS batches by episode/timestep alignment.

### Task 1.8: Implement certificate mining (`certvla/data/certificate_mining.py`)

Implement the advance/preserve/ignore classification per context doc sections 8.2-8.6:

```python
def mine_certificate(
    state_t: SlotState,
    state_t_H: SlotState,
    goal_signature: SlotState,
    future_states: List[SlotState],  # s_{t+H+1}, ..., s_{t+H+L}
    slot_registry: Dict,
    thresholds: MiningThresholds,
) -> CertificateLabel:
    """Mine advance/preserve/ignore labels for one chunk."""
    advance_set = set()

    # 1. For each result slot j in J_R:
    #    compute delta, rho, upsilon
    #    if advance criteria met -> advance

    # 2. For each enabling slot j in J_E:
    #    compute delta, eta (future support)
    #    if advance criteria met -> advance

    # 3. Compute preserve set via structural rules
    preserve_set = compute_preserve_set(state_t, advance_set)

    # 4. Everything else -> ignore

    # 5. For advance slots, goal_value = s_{t+H}^j

    return CertificateLabel(roles=..., goal_values=...)
```

Also implement:
```python
@dataclass
class MiningThresholds:
    tau_delta: float = 0.1
    tau_rho: float = 0.6
    tau_upsilon: float = 0.05
    tau_R: float = 0.1
    L_future: int = 5   # lookahead window for persistence/eta
    epsilon_j: float = 0.05
```

### Task 1.9: Implement goal signature (`certvla/data/goal_signature.py`)

```python
def compute_goal_signature(
    episode_states: List[SlotState],
    K: int = 5,
) -> SlotState:
    """Aggregate terminal K steps into a goal signature s*."""
    terminal_states = episode_states[-K:]
    # For continuous slots: average
    # For binary slots: majority vote
    # For categorical slots: mode
    ...
```

### Task 1.10: Write unit tests

Create under `tests/`:

1. **`test_slot_schema.py`**:
   - All 10 slots exist in registry
   - Domains are correct
   - Families are correct
   - Value validation works (rejects out-of-range values)

2. **`test_slot_metrics.py`**:
   - Binary distance: d(0,0)=0, d(0,1)=1
   - Categorical distance: d(same,same)=0, d(a,b)=1
   - Continuous distance: d(0.2, 0.8)=0.6
   - Round-trip tensor conversion

3. **`test_certificate_mining.py`**:
   - Synthetic episode: grasp-and-place scenario
   - Chunk 1 (approach): ee_target_proximity advances, hand_occupancy ignore
   - Chunk 2 (grasp): target_contact advances, hand_occupancy advances
   - Chunk 3 (transport): target_goal_proximity advances, hand_occupancy preserved
   - Chunk 4 (place): support_relation advances, completion_latch advances

4. **`test_preserve_rules.py`**:
   - Latch-preserve: completed slot not in advance -> preserve
   - Support-preserve: transport phase -> hand_occupancy preserved
   - Advance slot is never simultaneously preserved

5. **`test_goal_signature.py`**:
   - Successful episode -> goal signature reflects terminal state
   - Handles mixed slot types correctly

### Task 1.11: Verify LIBERO demo data access (prerequisite investigation)

Before implementing `LiberoOracleLabeler` body, write a minimal investigation script that:
1. Loads one LIBERO regenerated HDF5 file (the intermediate output of `regenerate_libero_dataset.py`)
2. Inspects available keys: `states`, `robot_states`, `joint_states`, `obs/ee_states`, etc.
3. Replays actions in LIBERO env using `env.set_init_state(states[0])` + `env.step(action)`
4. At each step, extracts `env.sim.data.body_xpos`, joint states, contacts
5. Confirms all 10 slots are computable from the available sim state
6. Documents which LIBERO env API calls correspond to each slot

This is a **prerequisite investigation** for Task 1.7. If some slots are not directly computable
from sim state, the implementation must be adjusted.

**NOTE**: This task requires access to:
- The regenerated HDF5 files (from running `regenerate_libero_dataset.py` on raw LIBERO data)
- A working LIBERO + MuJoCo + robosuite installation
- A machine with display/rendering capabilities (or headless rendering via EGL/OSMesa)

If this investigation cannot be done immediately (e.g. no LIBERO env available), Tasks 1.1-1.6
and 1.8-1.10 can still proceed using synthetic test data. Task 1.7 implementation body would
be deferred until the investigation completes.

---

## 7. What NOT To Do Now

1. **DO NOT** modify any file in `prismatic/` or `vla-scripts/`
2. **DO NOT** implement model heads (state_token, readout, certificate_head, action_head)
3. **DO NOT** implement any loss functions
4. **DO NOT** implement inference/gap/repair
5. **DO NOT** implement counterfactual data augmentation (Phase 3+)
6. **DO NOT** try to build an end-to-end training pipeline
7. **DO NOT** add proprio, wrist camera, or multi-image support
8. **DO NOT** build a "universal parser" that handles non-LIBERO environments
9. **DO NOT** attempt episode-sequential training (multi-chunk batches)
10. **DO NOT** implement WandB integration for cert metrics
11. **DO NOT** add YAML config files yet (use Python dataclasses until Phase 3)
12. **DO NOT** change the base OpenVLA/OFT training logic in any way
