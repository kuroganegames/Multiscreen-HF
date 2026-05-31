"""Configuration for the Transformers-compatible Multiscreen model."""

from __future__ import annotations

import math
from typing import Any

try:  # Transformers has historically used both spellings in examples.
    from transformers import PreTrainedConfig
except ImportError:  # pragma: no cover - compatibility fallback for old releases.
    from transformers import PretrainedConfig as PreTrainedConfig  # type: ignore


class MultiscreenConfig(PreTrainedConfig):
    """Configuration for Multiscreen causal language models.

    This mirrors the architecture knobs from ``dieOD/multiscreen-pytorch`` while
    exposing the conventional Transformers names where possible.

    Important aliases
    -----------------
    ``hidden_size`` <-> original ``hidden_dim``
    ``num_hidden_layers`` <-> original ``num_layers``
    ``num_attention_heads`` <-> original ``num_heads``
    ``max_position_embeddings`` <-> original ``max_seq_len``

    Reproducibility controls
    ------------------------
    ``mipe_compute_dtype`` and ``softmask_compute_dtype`` can be ``"fp32"``
    for the numerically safer Transformers port behavior, or ``"reference"``
    to use the incoming tensor dtype like the standalone PyTorch reference.
    ``strict_position_ids`` rejects batch-specific or non-contiguous
    ``position_ids`` because the reference cache API is based on a scalar
    ``start_pos``. ``strict_cache_positions`` additionally rejects unsafe
    nonzero no-cache ``start_pos`` and cached calls where ``start_pos != past_len``.
    ``zero_pad_hidden_states`` can additionally zero padded
    query states after each residual layer; it defaults to ``False`` to keep
    original residual behavior.
    """

    model_type = "multiscreen"
    keys_to_ignore_at_inference = ["past_key_values"]
    _alias_to_primary = {
        "hidden_dim": "hidden_size",
        "num_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads",
        "max_seq_len": "max_position_embeddings",
    }

    def __init__(
        self,
        vocab_size: int = 50_257,
        hidden_size: int | None = None,
        hidden_dim: int | None = None,
        num_hidden_layers: int | None = None,
        num_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_heads: int | None = None,
        key_dim: int = 16,
        value_dim: int = 64,
        max_position_embeddings: int | None = None,
        max_seq_len: int | None = None,
        mipe_threshold: float = 256.0,
        gradient_checkpointing: bool = False,
        use_cache: bool = True,
        labels_are_shifted: bool = False,
        mipe_compute_dtype: str = "fp32",
        softmask_compute_dtype: str = "fp32",
        strict_position_ids: bool = True,
        strict_cache_positions: bool = True,
        zero_pad_hidden_states: bool = False,
        initializer_range: float = 0.1,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        # Saved Transformers configs may contain these superclass fields in
        # kwargs. Multiscreen is always a decoder-only model, and passing them
        # through while also setting them below would duplicate keyword args.
        is_decoder = kwargs.pop("is_decoder", True)
        is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        kwargs.pop("model_type", None)
        if is_decoder is not True:
            raise ValueError("MultiscreenConfig requires is_decoder=True")
        if is_encoder_decoder is not False:
            raise ValueError("MultiscreenConfig requires is_encoder_decoder=False")

        hidden_size = self._resolve_alias(
            primary=hidden_size,
            alias=hidden_dim,
            default=256,
            primary_name="hidden_size",
            alias_name="hidden_dim",
        )
        num_hidden_layers = self._resolve_alias(
            primary=num_hidden_layers,
            alias=num_layers,
            default=8,
            primary_name="num_hidden_layers",
            alias_name="num_layers",
        )
        num_attention_heads = self._resolve_alias(
            primary=num_attention_heads,
            alias=num_heads,
            default=8,
            primary_name="num_attention_heads",
            alias_name="num_heads",
        )
        max_position_embeddings = self._resolve_alias(
            primary=max_position_embeddings,
            alias=max_seq_len,
            default=256,
            primary_name="max_position_embeddings",
            alias_name="max_seq_len",
        )

        if not bool(tie_word_embeddings):
            raise ValueError(
                "Multiscreen uses normalized tied input/output embeddings; "
                "tie_word_embeddings must be True."
            )

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.hidden_dim = int(hidden_size)  # original repo alias
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_layers = int(num_hidden_layers)  # original repo alias
        self.num_attention_heads = int(num_attention_heads)
        self.num_heads = int(num_attention_heads)  # original repo alias
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.max_seq_len = int(max_position_embeddings)  # original repo alias
        self.mipe_threshold = float(mipe_threshold)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_cache = bool(use_cache)
        self.labels_are_shifted = bool(labels_are_shifted)
        self.mipe_compute_dtype = str(mipe_compute_dtype)
        self.softmask_compute_dtype = str(softmask_compute_dtype)
        self.strict_position_ids = bool(strict_position_ids)
        self.strict_cache_positions = bool(strict_cache_positions)
        self.zero_pad_hidden_states = bool(zero_pad_hidden_states)
        self.initializer_range = float(initializer_range)

        self._validate()

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            is_encoder_decoder=False,
            use_cache=use_cache,
            **kwargs,
        )

        # Useful when pushing a repo with these Python files to the Hub and
        # loading with trust_remote_code=True.
        if not getattr(self, "auto_map", None):
            self.auto_map = {
                "AutoConfig": "configuration_multiscreen.MultiscreenConfig",
                "AutoModel": "modeling_multiscreen.MultiscreenModel",
                "AutoModelForCausalLM": "modeling_multiscreen.MultiscreenForCausalLM",
            }
        if not getattr(self, "architectures", None):
            self.architectures = ["MultiscreenForCausalLM"]

    @staticmethod
    def _resolve_alias(
        *,
        primary: int | None,
        alias: int | None,
        default: int,
        primary_name: str,
        alias_name: str,
    ) -> int:
        if primary is None and alias is None:
            return default
        if primary is None:
            return int(alias)  # type: ignore[arg-type]
        if alias is None:
            return int(primary)
        if int(primary) != int(alias):
            raise ValueError(
                f"Conflicting values for {primary_name}={primary} and "
                f"{alias_name}={alias}. Use only one or make them equal."
            )
        return int(primary)

    @classmethod
    def from_psi(
        cls,
        psi: int,
        vocab_size: int = 50_257,
        max_seq_len: int = 256,
        **overrides: Any,
    ) -> "MultiscreenConfig":
        """Build a paper-style config from the supraparameter Psi.

        The scaling rule used in the reference repo is ``N_L = N_H = Psi`` and
        ``d_E = Psi²``.
        """

        return cls(
            vocab_size=vocab_size,
            hidden_size=psi * psi,
            num_hidden_layers=psi,
            num_attention_heads=psi,
            max_position_embeddings=max_seq_len,
            **overrides,
        )

    def _validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size/hidden_dim must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers/num_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads/num_heads must be positive")
        if self.key_dim < 2:
            raise ValueError("key_dim must be at least 2 because MiPE rotates the first two coordinates")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings/max_seq_len must be positive")
        if self.mipe_threshold <= 0:
            raise ValueError("mipe_threshold must be positive")
        allowed_compute_dtypes = {"fp32", "reference"}
        if self.mipe_compute_dtype not in allowed_compute_dtypes:
            raise ValueError(
                "mipe_compute_dtype must be either 'fp32' or 'reference', "
                f"got {self.mipe_compute_dtype!r}"
            )
        if self.softmask_compute_dtype not in allowed_compute_dtypes:
            raise ValueError(
                "softmask_compute_dtype must be either 'fp32' or 'reference', "
                f"got {self.softmask_compute_dtype!r}"
            )
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    @classmethod
    def _normalize_alias_updates(cls, updates: dict[str, Any]) -> dict[str, Any]:
        """Map original-repository aliases to canonical Transformers field names."""

        normalized = dict(updates)
        for alias, primary in cls._alias_to_primary.items():
            if alias not in normalized:
                continue
            alias_value = normalized.pop(alias)
            if primary in normalized and int(normalized[primary]) != int(alias_value):
                raise ValueError(
                    f"Conflicting update values for {primary}={normalized[primary]} "
                    f"and {alias}={alias_value}. Use only one or make them equal."
                )
            normalized[primary] = alias_value
        return normalized

    def clone(self, **updates: Any) -> "MultiscreenConfig":
        """Return a config copy with updated fields.

        ``PreTrainedConfig.to_dict()`` contains both Transformers field names and
        original-repository aliases such as ``hidden_size``/``hidden_dim``.
        Reusing that dictionary directly can create alias conflicts when callers
        update only one spelling, so this method rebuilds from canonical fields
        and canonicalizes alias-style updates.
        """

        normalized_updates = self._normalize_alias_updates(updates)

        # ``is_decoder`` and ``is_encoder_decoder`` are forced by this class and
        # are passed explicitly to ``PreTrainedConfig`` in ``__init__``.
        # Silently accepting the matching values makes clone robust to dicts
        # produced by Transformers; conflicting values should fail loudly.
        if "is_decoder" in normalized_updates:
            if normalized_updates.pop("is_decoder") is not True:
                raise ValueError("MultiscreenConfig requires is_decoder=True")
        if "is_encoder_decoder" in normalized_updates:
            if normalized_updates.pop("is_encoder_decoder") is not False:
                raise ValueError("MultiscreenConfig requires is_encoder_decoder=False")

        data: dict[str, Any] = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "mipe_threshold": self.mipe_threshold,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_cache": self.use_cache,
            "labels_are_shifted": self.labels_are_shifted,
            "mipe_compute_dtype": self.mipe_compute_dtype,
            "softmask_compute_dtype": self.softmask_compute_dtype,
            "strict_position_ids": self.strict_position_ids,
            "strict_cache_positions": self.strict_cache_positions,
            "zero_pad_hidden_states": self.zero_pad_hidden_states,
            "initializer_range": self.initializer_range,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "tie_word_embeddings": True,
        }

        # Preserve useful PreTrainedConfig extras without reintroducing explicit
        # constructor arguments, aliases, or forced superclass kwargs.
        skip_keys = set(data) | set(self._alias_to_primary) | {
            "model_type",
            "is_decoder",
            "is_encoder_decoder",
            "transformers_version",
        }
        for key, value in self.to_dict().items():
            if key not in skip_keys:
                data[key] = value

        data.update(normalized_updates)
        return self.__class__(**data)

    @property
    def num_params_estimate(self) -> int:
        """Approximate parameter count, following the reference implementation.

        The estimate assumes tied input/output embeddings and ignores the small
        learned scalar parameters.
        """

        embed = self.vocab_size * self.hidden_size
        per_tile = self.hidden_size * (2 * self.key_dim + 3 * self.value_dim)
        total_tiles = self.num_hidden_layers * self.num_attention_heads
        return int(embed + total_tiles * per_tile)

    @property
    def sqrt_hidden_size(self) -> float:
        return math.sqrt(self.hidden_size)
