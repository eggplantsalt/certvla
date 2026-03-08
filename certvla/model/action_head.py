"""
Certificate-conditioned action head (coarse + fine residual).

Per context doc section 4.4:
    A_t^coarse = pi_c(z_t, hat_c_t)
    Delta_A_t^fine = pi_f(o_t, z_t, hat_c_t)
    hat_A_t = A_t^coarse + lambda_res * Delta_A_t^fine

Design:
- Coarse branch: takes z_t + flattened certificate embedding. Does NOT see o_t.
  This ensures the coarse action semantics depend on the certificate.
- Fine branch: takes the original actions_hidden_states from LLM (which see o_t)
  plus the certificate embedding. Produces a residual correction.
- lambda_res is a learnable scalar, initialized to 0.1.

The coarse branch replaces the original L1RegressionActionHead for CertVLA.
The fine branch wraps around it to add geometric refinement.
"""

import torch
import torch.nn as nn

# Default LIBERO constants. At runtime these are overridden via constructor args.
# Defined here to avoid importing prismatic (which triggers heavy dependencies).
_DEFAULT_ACTION_DIM = 7
_DEFAULT_NUM_ACTIONS_CHUNK = 8


class CertificateEmbedding(nn.Module):
    """Flatten certificate head outputs into a single vector.

    Takes role_logits and goal_preds dicts, concatenates them into
    a fixed-size vector suitable for action conditioning.
    """

    def __init__(self, cert_raw_dim: int, embed_dim: int):
        """
        Args:
            cert_raw_dim: Total dimension of flattened role_logits + goal_preds.
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cert_raw_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, cert_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cert_flat: Flattened certificate vector. Shape (B, cert_raw_dim).

        Returns:
            Certificate embedding. Shape (B, embed_dim).
        """
        return self.proj(cert_flat)


class CoarseActionBranch(nn.Module):
    """Coarse action branch: pi_c(z_t, c_t) -> A_t^coarse.

    Takes z_t and certificate embedding. Does NOT see observation tokens.
    This ensures the action semantics are certificate-dependent.
    """

    def __init__(
        self,
        llm_dim: int,
        cert_embed_dim: int,
        hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        input_dim = llm_dim + cert_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_actions_chunk * action_dim),
        )

    def forward(self, z_t: torch.Tensor, cert_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: State token hidden state. Shape (B, llm_dim).
            cert_embed: Certificate embedding. Shape (B, cert_embed_dim).

        Returns:
            Coarse actions. Shape (B, num_actions_chunk, action_dim).
        """
        x = torch.cat([z_t, cert_embed], dim=-1)  # (B, llm_dim + cert_embed_dim)
        out = self.net(x)  # (B, num_actions_chunk * action_dim)
        return out.reshape(-1, self.num_actions_chunk, self.action_dim)


class FineActionBranch(nn.Module):
    """Fine residual branch: pi_f(o_t, z_t, c_t) -> Delta_A_t.

    Takes the original actions_hidden_states from LLM (which encode o_t)
    plus z_t and certificate. Produces a residual correction.

    Architecture mirrors the original L1RegressionActionHead's MLPResNet
    but adds cert conditioning.
    """

    def __init__(
        self,
        llm_dim: int,
        cert_embed_dim: int,
        hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # Input: per-chunk action hidden states (reshaped) + z_t + cert_embed
        # The action hidden states are (B, chunk_len, action_dim * llm_dim).
        # We add z_t and cert per chunk step via broadcast.
        per_step_input_dim = action_dim * llm_dim + llm_dim + cert_embed_dim

        self.net = nn.Sequential(
            nn.LayerNorm(per_step_input_dim),
            nn.Linear(per_step_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        actions_hidden_states: torch.Tensor,
        z_t: torch.Tensor,
        cert_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: LLM hidden states at action positions.
                Shape (B, num_actions_chunk * action_dim, llm_dim).
            z_t: State token hidden state. Shape (B, llm_dim).
            cert_embed: Certificate embedding. Shape (B, cert_embed_dim).

        Returns:
            Fine residual actions. Shape (B, num_actions_chunk, action_dim).
        """
        B = actions_hidden_states.shape[0]
        # Reshape to per-chunk-step: (B, chunk_len, action_dim * llm_dim)
        act_h = actions_hidden_states.reshape(B, self.num_actions_chunk, -1)

        # Expand z_t and cert_embed to match chunk steps
        z_expanded = z_t.unsqueeze(1).expand(-1, self.num_actions_chunk, -1)   # (B, chunk, llm_dim)
        c_expanded = cert_embed.unsqueeze(1).expand(-1, self.num_actions_chunk, -1)  # (B, chunk, cert_embed)

        # Concatenate per step
        x = torch.cat([act_h, z_expanded, c_expanded], dim=-1)  # (B, chunk, per_step_input_dim)

        # Process each step (shared weights across steps)
        out = self.net(x)  # (B, chunk, action_dim)
        return out


class CertActionHead(nn.Module):
    """Certificate-conditioned action head: coarse + fine residual.

    hat_A_t = A_t^coarse + lambda_res * Delta_A_t^fine

    This replaces the base L1RegressionActionHead when CertVLA is active.
    """

    def __init__(
        self,
        llm_dim: int = 4096,
        cert_raw_dim: int = None,
        cert_embed_dim: int = 256,
        coarse_hidden_dim: int = 1024,
        fine_hidden_dim: int = 1024,
        action_dim: int = _DEFAULT_ACTION_DIM,
        num_actions_chunk: int = _DEFAULT_NUM_ACTIONS_CHUNK,
        lambda_res_init: float = 0.1,
    ):
        """
        Args:
            llm_dim: LLM hidden dimension.
            cert_raw_dim: Total dimension of flattened cert outputs.
                If None, auto-computed from SLOT_REGISTRY.
            cert_embed_dim: Certificate embedding dimension.
            coarse_hidden_dim: Coarse branch hidden dim.
            fine_hidden_dim: Fine branch hidden dim.
            action_dim: Action space dimension.
            num_actions_chunk: Number of actions per chunk.
            lambda_res_init: Initial value for lambda_res.
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk

        # Auto-compute cert_raw_dim if not provided
        if cert_raw_dim is None:
            cert_raw_dim = self._compute_cert_raw_dim()

        # Certificate embedding
        self.cert_embedding = CertificateEmbedding(cert_raw_dim, cert_embed_dim)

        # Coarse branch (z_t + cert -> actions, no observation)
        self.coarse = CoarseActionBranch(
            llm_dim=llm_dim,
            cert_embed_dim=cert_embed_dim,
            hidden_dim=coarse_hidden_dim,
            action_dim=action_dim,
            num_actions_chunk=num_actions_chunk,
        )

        # Fine residual branch (actions_hidden_states + z_t + cert -> delta actions)
        self.fine = FineActionBranch(
            llm_dim=llm_dim,
            cert_embed_dim=cert_embed_dim,
            hidden_dim=fine_hidden_dim,
            action_dim=action_dim,
            num_actions_chunk=num_actions_chunk,
        )

        # Learnable residual scaling
        self.lambda_res = nn.Parameter(torch.tensor(lambda_res_init))

    @staticmethod
    def _compute_cert_raw_dim() -> int:
        """Compute total flattened dimension of certificate head outputs.

        For each cert slot: 3 role logits + goal_dim.
        """
        from certvla.slots.schema import SlotDomain, SLOT_REGISTRY
        from certvla.slots.role_sets import J_CERT

        total = 0
        for slot_name in J_CERT:
            meta = SLOT_REGISTRY[slot_name]
            # Role logits: 3
            total += 3
            # Goal dim
            if meta.domain == SlotDomain.BINARY:
                total += 1
            elif meta.domain == SlotDomain.CATEGORICAL:
                total += len(meta.categories)
            else:
                total += 1
        return total

    def flatten_certificate(self, role_logits: dict, goal_preds: dict) -> torch.Tensor:
        """Flatten certificate head outputs into a single vector.

        Args:
            role_logits: Dict[SlotName, Tensor(B, 3)]
            goal_preds: Dict[SlotName, Tensor(B, dim)]

        Returns:
            Flattened certificate vector. Shape (B, cert_raw_dim).
        """
        from certvla.slots.role_sets import J_CERT

        parts = []
        for slot_name in sorted(J_CERT, key=lambda s: s.value):
            parts.append(role_logits[slot_name])  # (B, 3)
            parts.append(goal_preds[slot_name])    # (B, dim)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        z_t: torch.Tensor,
        role_logits: dict,
        goal_preds: dict,
        actions_hidden_states: torch.Tensor,
    ) -> tuple:
        """
        Args:
            z_t: State token hidden state. Shape (B, llm_dim).
            role_logits: From certificate head. Dict[SlotName, Tensor(B, 3)].
            goal_preds: From certificate head. Dict[SlotName, Tensor(B, dim)].
            actions_hidden_states: LLM action hidden states.
                Shape (B, num_actions_chunk * action_dim, llm_dim).

        Returns:
            actions: Final combined actions. Shape (B, chunk_len, action_dim).
            actions_coarse: Coarse actions. Shape (B, chunk_len, action_dim).
            actions_fine: Fine residual. Shape (B, chunk_len, action_dim).
        """
        # Flatten and embed certificate
        cert_flat = self.flatten_certificate(role_logits, goal_preds)
        cert_embed = self.cert_embedding(cert_flat)  # (B, cert_embed_dim)

        # Coarse: z_t + cert -> action semantics
        actions_coarse = self.coarse(z_t, cert_embed)  # (B, chunk, action_dim)

        # Fine: actions_hidden_states + z_t + cert -> geometric residual
        actions_fine = self.fine(actions_hidden_states, z_t, cert_embed)  # (B, chunk, action_dim)

        # Combine
        actions = actions_coarse + self.lambda_res * actions_fine

        return actions, actions_coarse, actions_fine
