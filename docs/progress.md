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

**Phase 2: Model Layer** should implement:
1. Persistent state token (`certvla/model/state_token.py`)
2. State readout head R_phi (`certvla/model/state_readout.py`)
3. Certificate head Q_psi (`certvla/model/certificate_head.py`)
4. Cert-conditioned action head (`certvla/model/action_head.py`)
5. CertVLA wrapper (`certvla/model/certvla_wrapper.py`)
6. Modifications to `modeling_prismatic.py` to inject state token
7. Shape/smoke tests

**Before Phase 2**: Complete Task 1.11 (LIBERO env investigation) if a LIBERO-capable machine is available. This unblocks `LiberoOracleLabeler` implementation and real sidecar label generation.
