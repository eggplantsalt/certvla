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

**Phase 3: Training Layer** should implement:
1. Loss functions: L_state, L_role, L_goal, L_act, L_cons, L_dep, L_cf
2. Training curriculum (staged loss weight scheduling)
3. Scheduled sampling for state token
4. Custom dataset that joins RLDS batches with sidecar labels
5. Integration with `finetune.py` (state token injection into VLA forward pass)
6. Config dataclass extensions for cert training

