# CertVLA Progress

## Phase 1: Data & Labeling Layer — COMPLETE

### Completed

| File | Description |
|------|-------------|
| `certvla/__init__.py` | Package root |
| `certvla/slots/__init__.py` | Slots subpackage, re-exports core API |
| `certvla/slots/schema.py` | Frozen v1 slot vocabulary: 10 slots, `SlotName` enum, `SlotDomain`, `SlotFamily`, `SlotMeta` dataclass, `SLOT_REGISTRY`, value validation |
| `certvla/slots/metrics.py` | Per-slot distance `d_j(a, b)`, `slot_value_to_tensor`, `tensor_to_slot_value`, `slot_state_to_flat_tensor`, `flat_tensor_dim` |
| `certvla/slots/role_sets.py` | `J_E` (5 enabling), `J_R` (4 result), `J_C` (1 confidence), `J_CERT = J_E | J_R` with compile-time assertions |
| `certvla/slots/preserve_rules.py` | `latch_preserve`, `support_preserve` (rule table with 5 structural rules), `compute_preserve_set` |
| `certvla/data/__init__.py` | Data subpackage, re-exports core API |
| `certvla/data/chunk_sample.py` | `SlotState`, `CertificateLabel`, `CertChunkSample` dataclasses |
| `certvla/data/state_labels.py` | `StateLabeler` (ABC), `PseudoLabelInterface` (ABC), `LiberoOracleLabeler` (interface defined, body deferred to Task 1.11) |
| `certvla/data/certificate_mining.py` | `mine_certificate()` with full advance/preserve/ignore logic, `MiningThresholds` dataclass |
| `certvla/data/goal_signature.py` | `compute_goal_signature()` with per-domain aggregation (mean/vote/mode) |
| `certvla/data/label_episodes.py` | Offline labeling script structure, `save_episode_labels`, `load_episode_labels`, CLI entrypoint |
| `certvla/data/counterfactuals.py` | `CounterfactualPair`, `CounterfactualBuilder` (ABC), `IdentityCounterfactualBuilder` (placeholder) |
| `configs/certvla/slots_v1.py` | Default mining thresholds, goal K, chunk size |
| `tests/conftest.py` | Shared fixtures (`slot_registry`, `make_slot_state`) |
| `tests/test_slot_schema.py` | 20 tests: registry, domains, families, categories, validation |
| `tests/test_slot_metrics.py` | 18 tests: distance fns, tensor conversion, round-trips, flat tensor |
| `tests/test_certificate_mining.py` | 9 tests: full 4-chunk grasp-and-place scenario + edge cases |
| `tests/test_preserve_rules.py` | 11 tests: latch-preserve, support-preserve, combined, disjointness |
| `tests/test_goal_signature.py` | 9 tests: terminal state, averaging, voting, mode, edge cases |

**Test results: 74/74 passed** (0.13s)

### Not Completed (Deferred by design)

| Item | Reason | When |
|------|--------|------|
| `LiberoOracleLabeler.extract_state()` body | Requires LIBERO + MuJoCo env running on Linux/GPU | Task 1.11 (prereq investigation) |
| `label_all_episodes()` body | Depends on `LiberoOracleLabeler` implementation | After Task 1.11 |
| Actual counterfactual augmentation | Image augmentation pipeline is Phase 3+ scope | Phase 3 |
| `PseudoLabelInterface` implementation | Real-robot pseudo-labeling is Phase 3+ scope | Phase 3+ |
| RLDS-sidecar alignment verification | Requires both RLDS dataset and sidecar labels | Phase 3 |

### Known Risks

1. **LIBERO env dependency**: `LiberoOracleLabeler` and `label_all_episodes()` require a working LIBERO + MuJoCo + robosuite installation on a Linux machine with GPU/display. Task 1.11 (environment investigation) must be completed before these can be fully implemented.

2. **Mining threshold sensitivity**: The default `MiningThresholds` values (tau_delta=0.1, tau_rho=0.6, etc.) are conservative starting points. They may need tuning once real LIBERO episode data is available. The 4-chunk synthetic test scenario validates correctness of the logic, not the threshold values.

3. **RLDS-sidecar alignment**: Sidecar `.npz` files are keyed by task name + demo index. The alignment with RLDS dataset timestep indices has not been verified. The regenerated HDF5 files (no-ops already removed) should align, but this must be confirmed in Phase 3.

4. **Support-preserve rule coverage**: The 5 structural rules in `preserve_rules.py` cover the most common LIBERO manipulation patterns (transport, placement, containment). Additional rules may be needed for less common tasks (e.g., articulation-only tasks, stacking). The rule table is designed to be easily extensible.

5. **Categorical slot encoding**: Categorical slots use one-hot encoding in the flat tensor representation. This works for 3-category slots but may need rethinking if slot vocabularies grow in future versions.

### Design Decisions Made

1. **Advance is data-mined, preserve is structural** — per context doc section 8.5. Preserve rules are explicit functions in `preserve_rules.py`, not statistical thresholds.

2. **SlotMeta is frozen** — `@dataclass(frozen=True)` prevents accidental mutation of the vocabulary.

3. **Validity mask + confidence are per-slot, per-timestep** — allows fine-grained control over which slots participate in training for which tasks.

4. **Goal signature uses domain-specific aggregation** — continuous: mean, binary: majority vote, categorical: mode. This avoids nonsensical averaging of categorical values.

5. **No existing files modified** — Phase 1 is fully additive. Zero changes to `prismatic/`, `vla-scripts/`, or any other existing code.

### Next Phase Recommendations

**Phase 2: Model Layer** — see below.

---

## Phase 2: Model Layer — COMPLETE

### Completed

| File | Description |
|------|-------------|
| `certvla/model/__init__.py` | Model subpackage |
| `certvla/model/outputs.py` | `CertVLAOutput` dataclass: z_t, state_readout, role_logits, goal_preds, actions (coarse + fine + combined), gate_value |
| `certvla/model/state_token.py` | `StateTokenModule`: learnable z_0, `get_state_token_embedding()`, `gated_update()` with sigmoid gate |
| `certvla/model/state_readout.py` | `StateReadoutHead`: z_t -> per-slot predictions. Shared trunk + per-slot output heads. Binary/continuous use sigmoid; categorical outputs raw logits |
| `certvla/model/certificate_head.py` | `CertificateHead`: z_t -> per-slot (role_logits, goal_preds) for J_CERT (9 slots). 3-way role classification [advance/preserve/ignore] + domain-appropriate goal predictions |
| `certvla/model/action_head.py` | `CertActionHead`: coarse + fine residual architecture. `CoarseActionBranch` (z_t + cert, no observation), `FineActionBranch` (actions_hidden_states + z_t + cert), `CertificateEmbedding`, learnable `lambda_res` |
| `certvla/model/certvla_wrapper.py` | `CertVLAWrapper`: composes all heads. Takes LLM `last_hidden_states` + position indices, returns `CertVLAOutput`. Does NOT subclass or modify base VLA |
| `tests/test_model_shapes.py` | 21 tests: shape verification, gradient isolation, certificate influence, parameter counts, full forward smoke test |

**Test results: 95/95 passed (74 Phase 1 + 21 Phase 2)** (2.08s)

### Architecture Summary

```
LLM last_hidden_states (B, seq_len, 4096)
  |
  |-- [state_token_pos] --> tilde_z_t (B, 4096)
  |                           |
  |                      gated_update(tilde_z_t, z_prev)
  |                           |
  |                         z_t (B, 4096)
  |                        /    \
  |          StateReadoutHead  CertificateHead
  |              |                |
  |          s_t (10 slots)   role_logits (9 slots x 3)
  |                           goal_preds (9 slots x dim)
  |                                |
  |                        CertificateEmbedding
  |                                |
  |                           cert_embed (B, 256)
  |                           /          \
  |-- [action positions] --> FineActionBranch  CoarseActionBranch
  |                           |                    |
  |                    actions_fine          actions_coarse
  |                           \                  /
  |                      actions = coarse + lambda_res * fine
  |                              (B, 8, 7)
```

### Design Constraints Satisfied

1. **State readout only reads from z_t**: `StateReadoutHead.forward(z_t)` takes a single vector. Test `test_readout_isolated_from_action_tokens` verifies gradients at action positions are zero when only readout loss is backpropagated.
2. **Certificate only reads from z_t**: Same architecture pattern as readout. Test `test_gradient_only_through_z_t` verifies.
3. **Action head explicitly receives certificate**: `CertActionHead.forward()` requires `role_logits` and `goal_preds` arguments. Test `test_cert_influences_actions` verifies changing certificate changes output.
4. **Fine branch accesses observation (via actions_hidden_states)**: `FineActionBranch` receives LLM hidden states which encode visual+text tokens.
5. **Single persistent state token**: `StateTokenModule` provides one `z_0` embedding of shape `(1, llm_dim)`.
6. **No external memory bank, no NL certificates, no symbolic planner**: Only structured slot outputs.

### Modifications to Existing Code: NONE

Phase 2 is fully additive. **Zero files in `prismatic/` or `vla-scripts/` were modified.** The wrapper operates on pre-extracted hidden states. Integration with the base VLA forward pass (injecting state token embedding into `_build_multimodal_attention`) is deferred to Phase 3.

The one design decision that avoids modifying `modeling_prismatic.py` now is: `CertVLAWrapper` takes `last_hidden_states` and position indices as arguments, rather than hooking into the LLM forward. This means Phase 3 will need to:
1. Append the state token embedding to `projected_patch_embeddings` before the LLM call
2. Pass `output_hidden_states=True` to the LLM
3. Call `CertVLAWrapper.forward()` with the extracted hidden states

### Design Decisions Made

1. **Wrapper pattern, not subclass**: `CertVLAWrapper` is a standalone `nn.Module`, not a subclass of `OpenVLAForActionPrediction`. This maximizes decoupling from the base model.

2. **No `prismatic` imports in `certvla/model/`**: To avoid triggering the heavy `prismatic` import chain (which requires `draccus`, `tensorflow`, etc.), all model modules define default constants locally (`_DEFAULT_ACTION_DIM = 7`, `_DEFAULT_NUM_ACTIONS_CHUNK = 8`) and accept dimensions as constructor parameters. At runtime (Phase 3), the actual constants from `prismatic.vla.constants` are passed in.

3. **Gate initialization**: `gate_proj` weights and biases are zero-initialized so the initial gate value is ~0.5 (balanced mix of new and old state). This follows standard practice for gated residual connections.

4. **Coarse branch has NO observation access**: `CoarseActionBranch` only sees `z_t + cert_embed`. This is a hard architectural constraint that ensures the coarse action semantics depend on the certificate. The `FineActionBranch` adds geometric correction using `actions_hidden_states` (which encode visual observations).

5. **Learnable `lambda_res`**: Initialized to 0.1 so the fine residual starts small and grows as training progresses. This prevents the fine branch from dominating early in training when the certificate is still noisy.

6. **Shared trunk per head**: State readout and certificate head each use a 2-layer shared trunk before per-slot output projections. This reduces parameter count while allowing per-slot specialization.

### Known Risks

1. **State token position in sequence**: The current design assumes the state token is inserted at a known position (after vision patches). In Phase 3, the exact position must be computed correctly accounting for BOS token, number of vision patches, and optional proprio/diffusion tokens.

2. **Fine branch input dimension**: `FineActionBranch` concatenates `(action_dim * llm_dim + llm_dim + cert_embed_dim)` per chunk step. With real dimensions (7 * 4096 + 4096 + 256 = 32,928), this is large. If memory is a concern, Phase 3 may need to add a projection layer to reduce the action hidden state dimension before concatenation.

3. **Parameter count**: With real `llm_dim=4096`, the CertVLA wrapper has ~70-100M parameters (dominated by the fine branch's large input projection). This is ~1-1.5% of the 7B base model, which is acceptable but should be monitored.

4. **v1 training: z_prev = z_0 always**: At training time, every sample uses the learnable z_0 as the "previous state." This means the gated update learns a residual structure but never experiences true recurrence. Episode-sequential training is a v2 enhancement.

### Next Phase Recommendations

**Phase 3: Training Layer** — see below.

---

## Phase 3: Training Layer — COMPLETE

### Completed

| File | Description |
|------|-------------|
| `certvla/training/__init__.py` | Training subpackage, re-exports all loss functions, curriculum, sampler |
| `certvla/training/losses.py` | All 7 loss functions: `cert_state_loss`, `cert_role_loss` (focal CE), `cert_goal_loss`, `cert_action_loss` (L1), `cert_consistency_loss` (advance + preserve), `cert_dependence_loss` (margin/triplet), `cert_counterfactual_loss` (invariance + breaking, minimal v1), `cert_total_loss` (weighted combiner). Helpers: `focal_cross_entropy`, `_per_slot_loss`, `_slot_pred_distance` |
| `certvla/training/curriculum.py` | `TrainingStage` enum (4 stages), `StageConfig` dataclass (loss weights + freeze flags + hyper-params), `DEFAULT_STAGES` dict, `CurriculumScheduler` (step-based stage lookup, `get_loss_weights()`, `should_compute_dep()`, `should_compute_cf()`) |
| `certvla/training/sched_sampling.py` | `ScheduledSampler` with constant/linear/cosine schedules, warmup support, `get_teacher_force_prob()`, `should_use_teacher()` |
| `tests/test_losses.py` | 30 tests: focal CE (3), L_state (3), L_role (2), L_goal (2), L_act (1), L_cons (2), L_dep (2), L_cf (3), total loss (2), curriculum (6), scheduled sampling (4), full smoke (1) |

**Test results: 144/144 passed (74 Phase 1 + 21 Phase 2 + 30 Phase 3 + 19 Phase 4)** (1.56s)

### Loss Functions Summary

| Loss | Symbol | File Location | Key Design |
|------|--------|--------------|------------|
| State readout | `L_state` | `losses.py:117` | Per-slot BCE/CE/L1 with mask × confidence weighting |
| Certificate role | `L_role` | `losses.py:148` | Focal CE (gamma=2.0) over J_CERT, handles class imbalance |
| Advance goal | `L_goal` | `losses.py:179` | Per-slot loss only where role==advance |
| Action chunk | `L_act` | `losses.py:215` | L1 regression, mean over (H × action_dim) |
| Consistency | `L_cons` | `losses.py:232` | Advance: goal vs GT@t+H. Preserve: readout vs GT@t+H. lambda_pre weight |
| Dependence | `L_dep` | `losses.py:292` | Margin triplet: max(0, m + e_pos - e_neg). Requires 2 action-head forwards |
| Counterfactual | `L_cf` | `losses.py:319` | L_inv (MSE invariance) + L_brk (margin divergence). No-op if no augmented pairs |
| **Total** | `L` | `losses.py:377` | Weighted sum via lambda_s..lambda_cf from StageConfig |

### Stage 1–4 Configuration

| Stage | Name | Active Losses | Frozen | Trainable |
|-------|------|---------------|--------|-----------|
| 1 | `stage1_state` | L_state | backbone, cert head, action head | state token + readout |
| 2 | `stage2_certificate` | L_state + L_role + L_goal | backbone, action head | state + cert head |
| 3 | `stage3_policy` | All supervised (+ L_cons, L_dep) | backbone | state + cert + action |
| 4 | `stage4_counterfactual` | All + L_cf | backbone | state + cert + action |

Default step boundaries: 0–5K / 5K–15K / 15K–40K / 40K–60K (configurable).

### Design Decisions Made

1. **Per-slot loss uses domain-appropriate functions**: BCE for binary (after sigmoid), CE for categorical (raw logits), L1 for continuous/confidence (after sigmoid). All losses reduce to scalar via mask × confidence × mean.

2. **Focal CE for role classification**: gamma=2.0 down-weights easy "ignore" examples. gamma=0 recovers standard CE (verified by test).

3. **Consistency loss uses GT at t+H in v1**: Since v1 does single-chunk training without recurrence, the model does not produce state predictions at t+H. Advance consistency reduces to reinforcing L_goal; preserve consistency is unique (forces readout stability for preserved slots).

4. **L_dep requires two forward passes through action head**: One with correct certificate, one with corrupted certificate. `CurriculumScheduler.should_compute_dep()` allows skipping in early stages.

5. **L_cf is a no-op without augmented pairs**: Returns zero gracefully. Full augmented-image pipeline is deferred. The interface supports nuisance-preserving (L_inv) and consequence-breaking (L_brk) terms when z_pos/z_neg embeddings are provided.

6. **Scheduled sampling v1 default**: constant schedule at prob=1.0 (always use z_0). Linear/cosine decay available for v2 episode-sequential training.

### Known Risks

1. **7 loss terms with weight scheduling is fragile**: Default weights are conservative (1.0 for primary, 0.5 for auxiliary). Will need grid search or automated tuning.

2. **L_dep doubles memory for action head**: Two forward passes through the action head per training step. `should_compute_dep()` gate prevents this in stages 1–2.

3. **Consistency loss advance term is redundant with L_goal in v1**: Both measure goal_pred vs GT@t+H. The preserve component provides unique signal.

---

## Phase 4: Inference & Repair — COMPLETE

### Completed

| File | Description |
|------|-------------|
| `certvla/inference/__init__.py` | Inference subpackage, re-exports gap, repair, logger |
| `certvla/inference/gap.py` | `slot_gap()`: per-slot role-probability-weighted gap. `aggregate_certificate_gap()`: confidence-weighted aggregation with slot importance. `GapResult` dataclass |
| `certvla/inference/repair.py` | `RepairController`: short-horizon local repair loop. `RepairConfig`: threshold, max attempts. Accepts `model_fn` callable (decoupled from model class) |
| `certvla/inference/logging.py` | `InferenceLogger`: episode-level trace collection. `StepRecord`, `EpisodeTrace` dataclasses. Summary statistics, warning recording, episode pruning |
| `tests/test_inference.py` | 19 tests: slot gap (4), aggregated gap (5), repair controller (3), logger (5), end-to-end fake rollout (1 + CertVLAWrapper) |

**Test results: 144/144 passed** (1.56s)

### Gap Definition & Implementation

**Per-slot gap** (`certvla/inference/gap.py:slot_gap`):
```
gamma_t^j = p_adv^j * d_j(goal^j, s_{t+H}^j)
          + p_pre^j * d_j(s_t^j,   s_{t+H}^j)
```
- `p_adv`, `p_pre` = softmax(role_logits) role probabilities
- `d_j` = differentiable per-slot distance (L1 for binary/continuous, TV for categorical)
- Only computed for J_CERT (9 cert slots)

**Aggregated gap** (`certvla/inference/gap.py:aggregate_certificate_gap`):
```
Gamma_t = [ sum_j omega_j * kappa_j * gamma_j ]
        / [ sum_j omega_j * kappa_j + epsilon ]
```
- `omega_j` = static slot importance weight (default 1.0)
- `kappa_j` = dynamic confidence weight
- Returns `GapResult` with per_slot, aggregated, and role_probs

### Repair Loop

`RepairController` (`certvla/inference/repair.py`):
1. Run initial forward via `model_fn(hidden, stp, asp, z_prev) -> CertVLAOutput`
2. Compute gap via `slot_gap()` + `aggregate_certificate_gap()`
3. If `gap.aggregated.mean() > threshold` → re-forward up to `max_repair_steps` times
4. Accept the attempt with lowest gap (best-of-N strategy)
5. Log all attempts and warnings via `InferenceLogger`

The `model_fn` callable decouples the repair loop from the concrete model class. At integration time it wraps `CertVLAWrapper.forward()`.

**v1 gap proxy**: Since the model only processes observation at time t, `state_readout_tH` is not available from a single forward pass. The v1 fallback uses `goal_preds` as the expected t+H state, yielding a self-consistency gap.

### Modifications to Existing Code: NONE

Phase 4 is fully additive. Zero files in `prismatic/` or `vla-scripts/` were modified.

### Known Risks

1. **v1 gap proxy is weak**: Using goal_preds as state_tH proxy means advance gap is always d(goal, goal) = 0. Real gap requires either a second forward pass with o_{t+H} or episode-sequential inference. The preserve gap (d(s_t, goal)) still provides useful signal.

2. **Repair loop is stateless**: Each re-forward uses the same hidden states and z_prev. Without model stochasticity (dropout, temperature), re-forwards may produce identical results. At inference time, the model should use either dropout or temperature > 0 for repair to be effective.

3. **No full replanner**: The repair loop is local retry only. It does not re-plan the task, modify the instruction, or use external symbolic reasoning. Full replanning is future scope.

### Next Phase Recommendations

**Phase 5: Integration** should implement:
1. State token injection into the VLA forward pass (modify `finetune.py`)
2. Custom dataset joining RLDS batches with sidecar labels
3. Config dataclass extensions for cert training
4. End-to-end training script with curriculum scheduler
5. Evaluation harness for LIBERO tasks with gap monitoring

