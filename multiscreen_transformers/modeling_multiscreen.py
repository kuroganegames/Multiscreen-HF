"""Transformers-compatible Multiscreen model.

The screening block is ported from ``dieOD/multiscreen-pytorch`` and wrapped in
Hugging Face ``PreTrainedModel`` classes.
"""

from __future__ import annotations

import copy
import math
import operator
import weakref
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
try:  # Transformers >=4.50 separates generation helpers from PreTrainedModel.
    from transformers.generation import GenerationMixin
except ImportError:  # pragma: no cover - compatibility with older releases.
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:  # pragma: no cover
        class GenerationMixin:  # type: ignore[no-redef]
            pass
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from .configuration_multiscreen import (
    MIPE_POSITION_MODE_PAPER_ABSOLUTE,
    MIPE_POSITION_MODE_REFERENCE,
    MultiscreenConfig,
)

logger = logging.get_logger(__name__)

# Per-layer screening cache.
# K: (batch, num_heads, cached_length, key_dim), post-MiPE and unit-normalized.
# V: (batch, num_heads, cached_length, value_dim), unit-normalized.
ScreeningCache = tuple[torch.Tensor, torch.Tensor]


def _dtype_safe_eps(x: torch.Tensor, eps: float) -> float:
    """Return an epsilon that remains positive in ``x``'s dtype."""

    eps = float(eps)
    if not torch.is_floating_point(x):
        return eps
    return max(eps, float(torch.finfo(x.dtype).tiny))


def _unit_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=_dtype_safe_eps(x, eps))


def _tanh_norm(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    safe_eps = _dtype_safe_eps(x, eps)
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > safe_eps, torch.tanh(norm) / norm.clamp_min(safe_eps), torch.ones_like(norm))
    return scale * x


def _multiscreen_expected_cache_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the post-normalization cache dtype under active autocast."""

    device_type = x.device.type
    try:
        autocast_enabled = torch.is_autocast_enabled(device_type)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        autocast_enabled = torch.is_autocast_enabled() if device_type == "cuda" else False
    if not autocast_enabled:
        return x.dtype
    # Autocast leaves float64 (and other non-eligible dtypes) unchanged. Cache
    # validation must preserve those dtypes instead of applying its target dtype.
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return x.dtype
    try:
        target_dtype = torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):  # pragma: no cover - older torch fallback.
        if device_type == "cuda":
            target_dtype = torch.get_autocast_gpu_dtype()
        elif device_type == "cpu":
            target_dtype = torch.get_autocast_cpu_dtype()
        else:
            return x.dtype
    # Cache tensors are unit-normalized projections. Autocast policies for
    # normalization differ by backend: CUDA promotes bf16/fp16 normalization to
    # fp32, while CPU bf16 normalization remains bf16. Probe the exact operator
    # with zero elements instead of assuming the GEMM target dtype.
    normalization_probe = torch.empty((0, 1), device=x.device, dtype=target_dtype)
    return F.normalize(normalization_probe, dim=-1).dtype


def convert_original_state_dict_for_causal_lm(
    state_dict: Mapping[str, torch.Tensor],
    *,
    strip_module_prefix: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert original ``dieOD/multiscreen-pytorch`` weights for HF CausalLM.

    The original repository's language model stores parameters under bare keys
    such as ``embed.weight`` and ``layers.0.block.q_proj.weight``.  This
    Transformers port keeps those modules inside ``MultiscreenForCausalLM`` as
    ``self.multiscreen``; state dict keys are prefixed with ``multiscreen.``.
    Loading the original checkpoint into ``MultiscreenForCausalLM`` requires
    that prefix. Already
    prefixed keys are left unchanged, so the helper is safe to call twice.

    Args:
        state_dict: State dict from the original implementation, or an already
            converted state dict.
        strip_module_prefix: Strip a leading ``module.`` prefix often added by
            DataParallel/DDP wrappers before conversion.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        converted_key = key
        if strip_module_prefix and converted_key.startswith("module."):
            converted_key = converted_key[len("module.") :]
        if not converted_key.startswith("multiscreen."):
            converted_key = f"multiscreen.{converted_key}"
        converted[converted_key] = value
    return converted


def convert_original_state_dict_for_model(
    state_dict: Mapping[str, torch.Tensor],
    *,
    strip_module_prefix: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert original or CausalLM-prefixed weights for bare ``MultiscreenModel``.

    Original ``dieOD/multiscreen-pytorch`` keys are already suitable for the bare
    decoder. This helper mainly strips a leading ``module.`` or ``multiscreen.``
    prefix when needed.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        converted_key = key
        if strip_module_prefix and converted_key.startswith("module."):
            converted_key = converted_key[len("module.") :]
        if converted_key.startswith("multiscreen."):
            converted_key = converted_key[len("multiscreen.") :]
        converted[converted_key] = value
    return converted




# ---- Generation cache compatibility helpers ---------------------------------
def _multiscreen_cache_seq_length(past_key_values) -> int:
    """Return cached sequence length for legacy tuple caches or HF Cache objects."""

    if past_key_values is None:
        return 0

    # Transformers Cache / DynamicCache API.  Different versions accept either
    # no argument or a layer index.
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        for args in ((), (0,)):
            try:
                value = get_seq_length(*args)
                if value is None:
                    return 0
                return int(value)
            except TypeError:
                continue

    # Some cache variants expose seen_tokens.
    seen_tokens = getattr(past_key_values, "seen_tokens", None)
    if seen_tokens is not None:
        try:
            return int(seen_tokens)
        except Exception:
            pass

    # Legacy tuple/list: ((K,V), ...), where K is (B,H,T,D).
    try:
        if len(past_key_values) == 0:
            return 0
        return int(past_key_values[0][0].shape[2])
    except Exception as exc:
        raise TypeError(
            "Unsupported past_key_values cache type for Multiscreen generation: "
            f"{type(past_key_values)!r}. Expected legacy tuple/list or a Transformers Cache/DynamicCache."
        ) from exc


def _multiscreen_normalize_past_key_values_for_forward(past_key_values):
    """Convert HF Cache/DynamicCache to the legacy tuple format used internally.

    Empty DynamicCache values are treated as no cache. Transformers 4.x Cache
    objects expose ``to_legacy_cache``; Transformers 5.x DynamicCache exposes
    per-layer ``keys``/``values`` instead. Other cache layouts are rejected.
    """

    if past_key_values is None:
        return None, 0

    past_length = _multiscreen_cache_seq_length(past_key_values)

    to_legacy_cache = getattr(past_key_values, "to_legacy_cache", None)
    if callable(to_legacy_cache):
        legacy = to_legacy_cache()
        if legacy is None or len(legacy) == 0:
            return None, 0
        if all(
            layer_cache is None
            or (
                isinstance(layer_cache, (tuple, list))
                and len(layer_cache) == 2
                and layer_cache[0] is None
                and layer_cache[1] is None
            )
            for layer_cache in legacy
        ):
            return None, 0
        return tuple(legacy), past_length

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        layers = list(layers)
        if not layers:
            return None, 0
        if all(
            getattr(layer, "keys", None) is None
            and getattr(layer, "values", None) is None
            for layer in layers
        ):
            return None, 0
        legacy_layers: list[ScreeningCache] = []
        for layer_idx, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise TypeError(
                    "Unsupported Transformers Cache layer for Multiscreen: "
                    f"layer {layer_idx} does not expose keys and values."
                )
            legacy_layers.append((key, value))
        if not legacy_layers:
            return None, 0
        return tuple(legacy_layers), past_length

    if isinstance(past_key_values, (tuple, list)):
        if len(past_key_values) == 0:
            return None, 0
        return tuple(past_key_values), past_length

    raise TypeError(
        "Unsupported past_key_values cache type for Multiscreen: "
        f"{type(past_key_values)!r}. Expected a legacy tuple/list or DynamicCache."
    )


def _multiscreen_validate_legacy_cache(
    past_key_values,
    *,
    num_layers: int,
    batch_size: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
    reported_length: int,
) -> tuple[Optional[tuple[ScreeningCache, ...]], int]:
    """Validate the complete zero-based prefix cache before screening math."""

    if past_key_values is None:
        return None, 0
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError("normalized past_key_values must be a tuple or list")
    if len(past_key_values) != num_layers:
        raise ValueError(
            f"past_key_values must contain {num_layers} layer caches, got {len(past_key_values)}."
        )

    validated: list[ScreeningCache] = []
    prefix_length: Optional[int] = None
    for layer_idx, layer_cache in enumerate(past_key_values):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
            raise TypeError(f"past_key_values[{layer_idx}] must be a (key, value) pair")
        key, value = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"past_key_values[{layer_idx}] key/value must be tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"past_key_values[{layer_idx}] key/value must both have rank 4, "
                f"got {key.ndim} and {value.ndim}"
            )
        expected_key_prefix = (batch_size, num_heads)
        expected_value_prefix = (batch_size, num_heads)
        if tuple(key.shape[:2]) != expected_key_prefix or key.shape[-1] != key_dim:
            raise ValueError(
                f"past_key_values[{layer_idx}] key must have shape "
                f"({batch_size}, {num_heads}, T, {key_dim}), got {tuple(key.shape)}"
            )
        if tuple(value.shape[:2]) != expected_value_prefix or value.shape[-1] != value_dim:
            raise ValueError(
                f"past_key_values[{layer_idx}] value must have shape "
                f"({batch_size}, {num_heads}, T, {value_dim}), got {tuple(value.shape)}"
            )
        layer_length = int(key.shape[2])
        if int(value.shape[2]) != layer_length:
            raise ValueError(f"past_key_values[{layer_idx}] key/value lengths differ")
        if prefix_length is None:
            prefix_length = layer_length
        elif layer_length != prefix_length:
            raise ValueError("all past_key_values layers must have the same prefix length")
        if key.device != expected_device or value.device != expected_device:
            raise ValueError(
                f"past_key_values[{layer_idx}] must be on {expected_device}, "
                f"got {key.device} and {value.device}"
            )
        if key.dtype != expected_dtype or value.dtype != expected_dtype:
            raise ValueError(
                f"past_key_values[{layer_idx}] must use dtype {expected_dtype}, "
                f"got {key.dtype} and {value.dtype}"
            )
        validated.append((key, value))

    actual_length = 0 if prefix_length is None else prefix_length
    if reported_length != actual_length:
        raise ValueError(
            f"cache reports length {reported_length}, but tensors contain {actual_length} tokens"
        )
    return tuple(validated), actual_length
# -----------------------------------------------------------------------------

class MultiscreenPreTrainedModel(PreTrainedModel):
    """Base class for Multiscreen Transformers models."""

    config_class = MultiscreenConfig
    base_model_prefix = "multiscreen"
    # Transformers 5 expects tied-weight metadata to be a mapping, while older
    # model classes often used a list of regex keys. Multiscreen has no
    # duplicated output-head Parameter to tie or drop from the state dict:
    # logits are computed directly from the normalized input embedding.
    _tied_weights_keys: dict[str, str] = {}
    supports_gradient_checkpointing = True
    _no_split_modules = ["MultiscreenLayer"]
    _skip_keys_device_placement = "past_key_values"
    # Newer Transformers Trainer may pass ``num_items_in_batch`` to models whose
    # forward signature has **kwargs. Multiscreen computes a standard mean CE
    # loss internally and does not consume that normalization hint.
    accepts_loss_kwargs = False

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict[str, str]:
        """Return no storage-level tied-parameter mapping for Multiscreen.

        The input/output embedding relationship is implemented by construction
        in ``_compute_logits`` / ``_NormalizedTiedLMHead`` instead of by
        assigning a second registered output-head Parameter to the input
        embedding Parameter. Returning an empty mapping keeps Transformers 5
        tied-weight bookkeeping on the mapping code path and avoids legacy
        list-vs-dict crashes.
        """

        return {}

    def _init_weights(self, module: nn.Module) -> None:  # pragma: no cover - post_init hook.
        """No-op because modules initialize with the original Multiscreen rules.

        The reference implementation uses per-projection initializers rather than
        a single global initializer. Those are applied in each module's ``__init__``.
        """

        return None

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Enable the supported Transformers checkpointing path non-reentrantly.

        Transformers 4.57 defaults to reentrant checkpointing while newer
        releases default to non-reentrant checkpointing. Multiscreen's validated
        training path is non-reentrant, so make that cross-version default
        explicit while still honoring caller-supplied checkpoint kwargs.
        """

        checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})
        checkpointing_kwargs.setdefault("use_reentrant", False)
        super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpointing_kwargs
        )


class MultiscreenModel(MultiscreenPreTrainedModel):
    """Bare Multiscreen decoder model.

    This returns hidden states, not vocabulary logits. Use
    :class:`MultiscreenForCausalLM` for the original language-model behavior.
    """

    def __init__(self, config: MultiscreenConfig) -> None:
        super().__init__(config)
        self.config = config
        # Runtime checkpointing is installed by PreTrainedModel.post_init() for
        # legacy config opt-in, or by gradient_checkpointing_enable(). Keeping
        # this false until then guarantees that a true flag has an installed
        # `_gradient_checkpointing_func` alongside it.
        self.gradient_checkpointing = False
        self.zero_pad_hidden_states = bool(config.zero_pad_hidden_states)

        d_e = config.hidden_size
        self.embed = nn.Embedding(config.vocab_size, d_e)
        self.s_E = nn.Parameter(torch.tensor(0.0))
        self.s_F = nn.Parameter(torch.tensor(math.log(math.sqrt(d_e))))
        self.layers = nn.ModuleList(
            [MultiscreenLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Original embedding initialization: N(0, 0.1 / sqrt(d_E)).
        nn.init.normal_(self.embed.weight, mean=0.0, std=config.initializer_range / math.sqrt(d_e))
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value
        self.config.vocab_size = value.num_embeddings

    def get_output_embeddings(self) -> nn.Embedding:
        # Output is tied by construction via normalized input embedding.
        return self.embed

    def set_output_embeddings(self, value: nn.Embedding) -> None:
        self.set_input_embeddings(value)

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        # We do not create a separate lm_head; logits use self.embed.weight.
        # Recent Transformers releases call tie_weights with keyword arguments
        # such as recompute_mapping=... or missing_keys=...; accept and ignore
        # them because Multiscreen ties weights by construction.
        return None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Sequence[ScreeningCache] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        start_pos: int | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[torch.Tensor, ...]:
        """Run the Multiscreen decoder.

        Args follow Transformers conventions. ``start_pos`` is kept as an
        explicit compatibility escape hatch for the original cache API.
        """

        kv_caches = kwargs.pop("kv_caches", None)
        if kv_caches is not None:
            if past_key_values is not None:
                raise ValueError("Pass only one of `past_key_values` or original-api `kv_caches`, not both.")
            past_key_values = kv_caches

        # Compatibility with Trainer/TRL/Transformers call paths.
        use_return_dict = kwargs.pop("use_return_dict", None)
        if return_dict is None and use_return_dict is not None:
            return_dict = bool(use_return_dict)
        kwargs.pop("num_items_in_batch", None)

        if kwargs:
            # Keep forward permissive, but surface likely typo/debug information.
            # ``warning_once`` caches calls, so every argument must be hashable.
            unused_kwargs = ", ".join(sorted(str(key) for key in kwargs.keys()))
            logger.warning_once("Unused MultiscreenModel.forward kwargs: %s", unused_kwargs)

        if output_attentions:
            logger.warning_once(
                "Multiscreen has no softmax attention weights; `output_attentions=True` returns None."
            )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        requested_cache = self.config.use_cache if use_cache is None else bool(use_cache)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Pass either input_ids or inputs_embeds, not both.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must pass input_ids or inputs_embeds.")
        if inputs_embeds is not None:
            raise ValueError(
                "Multiscreen does not accept `inputs_embeds` through the public Transformers API. "
                "The reference architecture normalizes token embedding weights before lookup, so raw "
                "embeddings from `get_input_embeddings()` are not equivalent to `input_ids`. "
                "Pass `input_ids` instead."
            )

        if input_ids is None:
            raise ValueError("input_ids unexpectedly None")
        input_shape = input_ids.shape
        batch_size, seq_len = input_shape
        W_norm = _unit_normalize(self.embed.weight)
        hidden_states = F.embedding(input_ids, W_norm) * self.s_E.exp()

        past_key_values, reported_past_length = (
            _multiscreen_normalize_past_key_values_for_forward(past_key_values)
        )
        past_key_values, past_length = _multiscreen_validate_legacy_cache(
            past_key_values,
            num_layers=len(self.layers),
            batch_size=batch_size,
            num_heads=self.config.num_attention_heads,
            key_dim=self.config.key_dim,
            value_dim=self.config.value_dim,
            expected_device=hidden_states.device,
            expected_dtype=_multiscreen_expected_cache_dtype(hidden_states),
            reported_length=reported_past_length,
        )

        explicit_start_pos = None
        if start_pos is not None:
            explicit_start_pos = self._coerce_nonnegative_position(start_pos, "start_pos")
        position_start_pos = None
        if position_ids is not None:
            position_start_pos = self._start_pos_from_position_ids(
                position_ids=position_ids,
                batch_size=batch_size,
                seq_len=seq_len,
                strict=bool(self.config.strict_position_ids),
            )
        if explicit_start_pos is not None and position_start_pos is not None:
            if explicit_start_pos != position_start_pos:
                raise ValueError(
                    f"start_pos={explicit_start_pos} conflicts with position_ids start "
                    f"{position_start_pos}"
                )
        if start_pos is None:
            start_pos = position_start_pos if position_start_pos is not None else past_length
        else:
            start_pos = explicit_start_pos

        if bool(getattr(self.config, "strict_cache_positions", True)):
            if past_length > 0 and start_pos != past_length:
                raise ValueError(
                    "Multiscreen cached decoding requires a contiguous prefix cache starting at position 0; "
                    f"got start_pos={start_pos} but past_length={past_length}."
                )
            if past_length == 0 and start_pos != 0:
                raise ValueError(
                    "Multiscreen full-context/no-cache calls require start_pos=0. "
                    "Offset position_ids without a prefix cache are not supported because MiPE and the "
                    "distance Softmask would use different key-position origins."
                )

        use_cache = requested_cache and (not self.training)
        if requested_cache and self.training:
            logger.warning_once("Multiscreen disables cache materialization while model.training is True.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("use_cache=True is incompatible with gradient checkpointing; disabling cache.")
            use_cache = False

        total_length = past_length + seq_len
        key_attention_mask, query_attention_mask = self._prepare_attention_masks(
            attention_mask=attention_mask,
            batch_size=batch_size,
            past_length=past_length,
            seq_len=seq_len,
            total_length=total_length,
            device=hidden_states.device,
        )
        query_mask_3d = (
            query_attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            if query_attention_mask is not None
            else None
        )
        if self.zero_pad_hidden_states and query_mask_3d is not None:
            hidden_states = hidden_states * query_mask_3d

        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        new_key_values: list[ScreeningCache] = []

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

            past_layer = past_key_values[layer_idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def custom_forward(
                    x: torch.Tensor,
                    layer_ref: MultiscreenLayer = layer,
                    start_pos_ref: int = start_pos,
                    key_attention_mask_ref: torch.Tensor | None = key_attention_mask,
                    query_attention_mask_ref: torch.Tensor | None = query_attention_mask,
                ) -> torch.Tensor:
                    y, _ = layer_ref(
                        x,
                        start_pos=start_pos_ref,
                        past_kv=None,
                        use_cache=False,
                        key_attention_mask=key_attention_mask_ref,
                        query_attention_mask=query_attention_mask_ref,
                    )
                    return y

                hidden_states = self._gradient_checkpointing_func(custom_forward, hidden_states)
                new_kv = None
            else:
                hidden_states, new_kv = layer(
                    hidden_states,
                    start_pos=start_pos,
                    past_kv=past_layer,
                    use_cache=use_cache,
                    key_attention_mask=key_attention_mask,
                    query_attention_mask=query_attention_mask,
                )

            if self.zero_pad_hidden_states and query_mask_3d is not None:
                hidden_states = hidden_states * query_mask_3d

            if use_cache:
                if new_kv is None:
                    raise RuntimeError("Layer did not return a cache while use_cache=True")
                new_key_values.append(new_kv)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        past = tuple(new_key_values) if use_cache else None

        if not return_dict:
            outputs: tuple[Any, ...] = (hidden_states,)
            if past is not None:
                outputs += (past,)
            if all_hidden_states is not None:
                outputs += (all_hidden_states,)
            return outputs  # type: ignore[return-value]

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    @staticmethod
    def _coerce_nonnegative_position(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        try:
            position = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if position < 0:
            raise ValueError(f"{name} must be nonnegative")
        return position

    @staticmethod
    def _start_pos_from_position_ids(
        *,
        position_ids: torch.LongTensor,
        batch_size: Optional[int] = None,
        seq_len: int,
        strict: bool,
    ) -> int:
        """Extract the scalar reference-style ``start_pos`` from position IDs.

        Multiscreen's reference implementation uses one scalar ``start_pos`` for
        every batch item. Arbitrary per-token or per-batch ``position_ids`` would
        misalign MiPE and the distance softmask, so strict mode fails loudly.
        """

        if not isinstance(position_ids, torch.Tensor):
            raise TypeError("position_ids must be a tensor")
        if position_ids.dim() != 2:
            raise ValueError("position_ids must have shape (batch, sequence_length)")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if position_ids.dtype not in integer_dtypes:
            raise TypeError("position_ids must use an integer dtype")
        if batch_size is not None and int(position_ids.shape[0]) != batch_size:
            raise ValueError(
                f"position_ids batch {position_ids.shape[0]} does not match input batch {batch_size}"
            )
        if int(position_ids.shape[1]) != seq_len:
            raise ValueError(
                f"position_ids length {position_ids.shape[1]} does not match input sequence length {seq_len}"
            )
        if seq_len == 0:
            return 0

        start_pos = int(position_ids[0, 0].item())
        if start_pos < 0:
            raise ValueError("position_ids must be nonnegative")
        if seq_len - 1 > torch.iinfo(torch.int64).max - start_pos:
            raise ValueError("position_ids contiguous range exceeds int64")
        positions_i64 = position_ids.to(dtype=torch.int64)
        expected = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=position_ids.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(position_ids.shape[0], -1)

        if not torch.equal(positions_i64, expected):
            message = (
                "Multiscreen only supports batch-shared contiguous position_ids, "
                "because the reference cache API is based on a scalar start_pos. "
                "Pass start_pos explicitly for reference-style decoding, or disable "
                "config.strict_position_ids only if you intentionally want to use "
                "position_ids[0, 0] and ignore the rest."
            )
            if strict:
                raise ValueError(message)
            logger.warning_once(message)
        return start_pos

    @staticmethod
    def _start_pos_from_cache_position(
        *, cache_position: torch.LongTensor, seq_len: int
    ) -> int:
        if not isinstance(cache_position, torch.Tensor):
            raise TypeError("cache_position must be a tensor")
        if cache_position.dim() != 1:
            raise ValueError("cache_position must have shape (sequence_length,)")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if cache_position.dtype not in integer_dtypes:
            raise TypeError("cache_position must use an integer dtype")
        if int(cache_position.numel()) != seq_len:
            raise ValueError(
                f"cache_position length {cache_position.numel()} does not match sequence length {seq_len}"
            )
        if seq_len == 0:
            raise ValueError("cache_position must not be empty")
        start_pos = int(cache_position[0].item())
        if start_pos < 0:
            raise ValueError("cache_position must be nonnegative")
        if seq_len - 1 > torch.iinfo(torch.int64).max - start_pos:
            raise ValueError("cache_position contiguous range exceeds int64")
        expected = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=cache_position.device,
            dtype=torch.int64,
        )
        if not torch.equal(cache_position.to(dtype=torch.int64), expected):
            raise ValueError("cache_position must be a contiguous increasing range")
        return start_pos

    @staticmethod
    def _prepare_attention_masks(
        *,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        past_length: int,
        seq_len: int,
        total_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if attention_mask is None:
            return None, None

        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape (batch, sequence_length)")
        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch size {attention_mask.shape[0]} does not match input batch {batch_size}"
            )

        mask = attention_mask.to(device=device)
        mask_len = int(mask.shape[1])

        if past_length > 0 and mask_len != total_length:
            logger.warning_once(
                "Cached Multiscreen decoding received an attention_mask whose length (%s) "
                "does not cover the full cache length (%s). Omitted past cache positions are "
                "treated as valid. Pass a full-length attention_mask when cached prefixes "
                "contain padding.",
                mask_len,
                total_length,
            )

        if mask_len == total_length:
            key_mask = mask
            query_mask = mask[:, -seq_len:]
        elif mask_len == seq_len:
            if past_length > 0:
                prefix = torch.ones(batch_size, past_length, device=device, dtype=mask.dtype)
                key_mask = torch.cat([prefix, mask], dim=1)
            else:
                key_mask = mask
            query_mask = mask
        elif mask_len > total_length:
            key_mask = mask[:, -total_length:]
            query_mask = key_mask[:, -seq_len:]
        elif mask_len < total_length:
            # If only a shorter mask is supplied during cached decoding, assume
            # the missing older cache positions are valid.
            prefix = torch.ones(batch_size, total_length - mask_len, device=device, dtype=mask.dtype)
            key_mask = torch.cat([prefix, mask], dim=1)
            query_mask = key_mask[:, -seq_len:]
        else:  # pragma: no cover - unreachable, kept for clarity.
            key_mask = mask
            query_mask = mask[:, -seq_len:]

        return key_mask, query_mask




class _NormalizedTiedLMHead(nn.Module):
    """Parameter-free lm_head proxy for trainers that expect ``model.lm_head``.

    Multiscreen computes logits with the unit-normalized input embedding matrix
    and the learned scalar ``s_F`` instead of a standalone Linear layer.  Some
    Hugging Face/TRL training paths look for ``model.lm_head.weight`` to avoid
    materializing full logits.  This proxy exposes the mathematically equivalent
    dynamic weight while keeping the true parameters tied to ``embed.weight`` and
    ``s_F``.
    """

    def __init__(self, owner: "MultiscreenForCausalLM") -> None:
        super().__init__()
        self._owner_ref = weakref.ref(owner)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_NormalizedTiedLMHead":
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("Cannot copy a Multiscreen lm_head after its owner was collected")
        copied_owner = memo.get(id(owner))
        if copied_owner is None:
            raise RuntimeError("Multiscreen lm_head must be copied as part of its owning model")

        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.__dict__ = copy.deepcopy(self.__dict__, memo)
        copied._owner_ref = weakref.ref(copied_owner)
        return copied

    @property
    def weight(self) -> torch.Tensor:
        owner = self._owner_ref()
        if owner is None:  # pragma: no cover - defensive only.
            raise RuntimeError("Multiscreen lm_head owner has been garbage-collected")
        W_norm = _unit_normalize(owner.multiscreen.embed.weight)
        return W_norm * owner.multiscreen.s_F.exp()

    @property
    def bias(self) -> None:
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class MultiscreenForCausalLM(MultiscreenPreTrainedModel, GenerationMixin):
    """Multiscreen decoder with normalized tied LM head."""

    # ``self.lm_head`` is a parameter-free proxy whose dynamic ``weight``
    # property is computed from ``multiscreen.embed.weight`` and ``s_F``. There
    # is no registered ``lm_head.weight`` Parameter, so report no explicit tied
    # parameter pair to Transformers.
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: MultiscreenConfig) -> None:
        if not bool(getattr(config, "tie_word_embeddings", True)):
            raise ValueError(
                "Multiscreen uses normalized tied input/output embeddings; "
                "tie_word_embeddings must be True."
            )
        super().__init__(config)
        self.multiscreen = MultiscreenModel(config)
        # Compatibility shim for TRL/SFTTrainer paths that expect a ``lm_head``
        # attribute. It has no parameters; it dynamically reuses the normalized
        # tied input embeddings exactly like ``_compute_logits``.
        self.lm_head = _NormalizedTiedLMHead(self)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.multiscreen.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.multiscreen.set_input_embeddings(value)
        self.config.vocab_size = value.num_embeddings
        self.vocab_size = value.num_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        if value is self.lm_head:
            return
        raise ValueError(
            "Multiscreen uses a parameter-free normalized tied lm_head; "
            "replacing the output embeddings is unsupported."
        )

    def _resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        # The output head is a parameter-free dynamic view of the input
        # embedding. Transformers 5.x otherwise tries to resize it as an
        # nn.Linear, so resize only the true input parameter.
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings,
            new_num_tokens,
            pad_to_multiple_of,
            mean_resizing,
        )
        if hasattr(old_embeddings, "_hf_hook"):
            from accelerate.hooks import add_hook_to_module

            add_hook_to_module(new_embeddings, old_embeddings._hf_hook)
        new_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        # Output logits directly reuse the normalized input embedding matrix.
        # Recent Transformers releases call tie_weights with keyword arguments
        # such as recompute_mapping=... or missing_keys=...; accept and ignore
        # them because Multiscreen ties weights by construction.
        return None

    @staticmethod
    def convert_original_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert original ``multiscreen-pytorch`` checkpoint keys for this class."""

        return convert_original_state_dict_for_causal_lm(state_dict)

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @staticmethod
    def _coerce_optional_bool(value: Any, name: str) -> bool | None:
        """Convert Python or collated tensor booleans to a scalar bool.

        ``PackedTextDataset`` can emit a scalar ``labels_are_shifted`` flag so a
        standard Transformers data collator/Trainer forwards it to the model.
        After collation this arrives as a batch tensor; mixed True/False values
        in one batch are rejected because a single loss path must be chosen.
        """

        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            bool_values = value.detach().to(dtype=torch.bool).flatten()
            has_true = bool(bool_values.any().item())
            has_false = bool((~bool_values).any().item())
            if has_true and has_false:
                raise ValueError(f"{name} must be the same for every item in a batch.")
            return has_true
        return bool(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Sequence[ScreeningCache] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        start_pos: int | None = None,
        labels_are_shifted: bool | None = None,
        legacy_shifted_labels: bool | None = None,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        # Compatibility with Trainer/TRL/Transformers call paths.
        # ``use_return_dict`` is a deprecated alias that may still be forwarded,
        # and ``num_items_in_batch`` can be injected by recent Trainer versions
        # when a model forward has **kwargs. Multiscreen does not consume it.
        use_return_dict = kwargs.pop("use_return_dict", None)
        if return_dict is None and use_return_dict is not None:
            return_dict = bool(use_return_dict)
        kwargs.pop("num_items_in_batch", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        labels_are_shifted_value = self._coerce_optional_bool(labels_are_shifted, "labels_are_shifted")
        legacy_shifted_labels_value = self._coerce_optional_bool(
            legacy_shifted_labels, "legacy_shifted_labels"
        )
        if labels_are_shifted_value is not None and legacy_shifted_labels_value is not None:
            if labels_are_shifted_value != legacy_shifted_labels_value:
                raise ValueError("labels_are_shifted and legacy_shifted_labels disagree.")
        if labels_are_shifted_value is None:
            labels_are_shifted = (
                legacy_shifted_labels_value
                if legacy_shifted_labels_value is not None
                else bool(getattr(self.config, "labels_are_shifted", False))
            )
        else:
            labels_are_shifted = labels_are_shifted_value

        kv_caches = kwargs.pop("kv_caches", None)
        if kv_caches is not None:
            if past_key_values is not None:
                raise ValueError("Pass only one of `past_key_values` or original-api `kv_caches`, not both.")
            past_key_values = kv_caches

        if kwargs:
            # Keep forward permissive for HF/TRL extras, but do not pass them to
            # the bare decoder where they would only generate duplicate warnings.
            unused_kwargs = ", ".join(sorted(str(key) for key in kwargs.keys()))
            logger.warning_once("Unused MultiscreenForCausalLM.forward kwargs: %s", unused_kwargs)

        model_outputs = self.multiscreen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            start_pos=start_pos,
        )
        hidden_states = model_outputs.last_hidden_state

        if labels is None and logits_to_keep and logits_to_keep > 0:
            logits_hidden_states = hidden_states[:, -logits_to_keep:, :]
        else:
            logits_hidden_states = hidden_states

        logits = self._compute_logits(logits_hidden_states)
        loss = None

        if labels is not None:
            if logits.shape[1] != labels.shape[1]:
                # This can only happen if a caller forced logits_to_keep with labels.
                logits = self._compute_logits(hidden_states)

            loss_labels = labels.to(device=logits.device).clone()
            loss_attention_mask = None
            if attention_mask is not None:
                loss_attention_mask = self._slice_loss_attention_mask(
                    attention_mask=attention_mask,
                    target_length=loss_labels.shape[1],
                    device=loss_labels.device,
                )
                loss_labels = loss_labels.masked_fill(loss_attention_mask == 0, -100)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if labels_are_shifted:
                loss = loss_fct(
                    logits.reshape(-1, self.config.vocab_size),
                    loss_labels.reshape(-1),
                )
            else:
                if logits.shape[1] < 2:
                    loss = logits.new_zeros(())
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = loss_labels[..., 1:].contiguous()
                    if loss_attention_mask is not None:
                        # Ignore predictions where either the query token or the
                        # target token is padding. This avoids a left-padding edge
                        # case where the last pad token predicts the first real token.
                        valid_shift = (loss_attention_mask[..., :-1] != 0) & (
                            loss_attention_mask[..., 1:] != 0
                        )
                        shift_labels = shift_labels.masked_fill(~valid_shift, -100)
                    loss = loss_fct(
                        shift_logits.reshape(-1, self.config.vocab_size),
                        shift_labels.reshape(-1),
                    )

        if not return_dict:
            output: tuple[Any, ...] = (logits,)
            if model_outputs.past_key_values is not None:
                output += (model_outputs.past_key_values,)
            if model_outputs.hidden_states is not None:
                output += (model_outputs.hidden_states,)
            return ((loss,) + output) if loss is not None else output  # type: ignore[return-value]

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=None,
        )

    @staticmethod
    def _slice_loss_attention_mask(
        *,
        attention_mask: torch.Tensor,
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape (batch, sequence_length)")
        mask = attention_mask.to(device=device)
        mask_length = int(mask.shape[1])
        if mask_length == target_length:
            return mask
        if mask_length > target_length:
            return mask[:, -target_length:]
        prefix = torch.ones(
            mask.shape[0],
            target_length - mask_length,
            device=device,
            dtype=mask.dtype,
        )
        return torch.cat([prefix, mask], dim=1)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Sequence[ScreeningCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        start_pos: int | None = None,
        use_cache: bool | None = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for ``GenerationMixin.generate``.

        Multiscreen keeps a simple tuple cache. When a cache is present, the
        method slices ``input_ids`` to the new suffix and sets a scalar
        ``start_pos`` equal to the cached sequence length.
        """

        kv_caches = kwargs.pop("kv_caches", None)
        if kv_caches is not None:
            if past_key_values is not None:
                raise ValueError("Pass only one of `past_key_values` or original-api `kv_caches`, not both.")
            past_key_values = kv_caches

        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length)")
        batch_size, original_input_length = input_ids.shape
        past_key_values, reported_past_length = (
            _multiscreen_normalize_past_key_values_for_forward(past_key_values)
        )
        past_key_values, past_length = _multiscreen_validate_legacy_cache(
            past_key_values,
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            num_heads=self.config.num_attention_heads,
            key_dim=self.config.key_dim,
            value_dim=self.config.value_dim,
            expected_device=input_ids.device,
            expected_dtype=_multiscreen_expected_cache_dtype(self.multiscreen.embed.weight),
            reported_length=reported_past_length,
        )

        if past_length > 0:
            input_contains_prefix = original_input_length > past_length
            if cache_position is not None:
                suffix_length = int(cache_position.numel())
                if suffix_length == original_input_length:
                    input_contains_prefix = False
                elif not (
                    input_contains_prefix
                    and suffix_length == original_input_length - past_length
                ):
                    raise ValueError(
                        f"cache_position length {suffix_length} matches neither the entire "
                        f"already-sliced input length {original_input_length} nor the inferred "
                        f"full-input suffix for cached length {past_length}"
                    )
                cache_start = MultiscreenModel._start_pos_from_cache_position(
                    cache_position=cache_position,
                    seq_len=suffix_length,
                )
                if cache_start != past_length:
                    raise ValueError(
                        f"cache_position starts at {cache_start}, expected cached length {past_length}"
                    )
                if input_contains_prefix:
                    input_ids = input_ids[:, past_length:]
            elif input_contains_prefix:
                input_ids = input_ids[:, past_length:]
                suffix_length = int(input_ids.shape[1])
            else:
                # Treat already-sliced multi-token input as the new suffix.
                suffix_length = original_input_length

            if suffix_length <= 0:
                raise ValueError("cached generation requires at least one new token")
            if start_pos is not None:
                explicit_start = MultiscreenModel._coerce_nonnegative_position(
                    start_pos, "start_pos"
                )
                if explicit_start != past_length:
                    raise ValueError(
                        f"start_pos={explicit_start} conflicts with cached length {past_length}"
                    )
            if position_ids is not None:
                position_ids_for_suffix = position_ids
                if (
                    position_ids.dim() == 2
                    and int(position_ids.shape[1]) == original_input_length
                    and suffix_length < original_input_length
                ):
                    position_ids_for_suffix = position_ids[:, -suffix_length:]
                position_start = MultiscreenModel._start_pos_from_position_ids(
                    position_ids=position_ids_for_suffix,
                    batch_size=batch_size,
                    seq_len=suffix_length,
                    strict=bool(self.config.strict_position_ids),
                )
                if position_start != past_length:
                    raise ValueError(
                        f"position_ids start at {position_start}, expected cached length {past_length}"
                    )
            start_pos = past_length
        else:
            seq_len = original_input_length
            cache_start = None
            if cache_position is not None:
                cache_start = MultiscreenModel._start_pos_from_cache_position(
                    cache_position=cache_position,
                    seq_len=seq_len,
                )
            position_start = None
            if position_ids is not None:
                position_start = MultiscreenModel._start_pos_from_position_ids(
                    position_ids=position_ids,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    strict=bool(self.config.strict_position_ids),
                )
            explicit_start = None
            if start_pos is not None:
                explicit_start = MultiscreenModel._coerce_nonnegative_position(
                    start_pos, "start_pos"
                )
            supplied_starts = [
                value for value in (cache_start, position_start, explicit_start) if value is not None
            ]
            if supplied_starts and any(value != supplied_starts[0] for value in supplied_starts[1:]):
                raise ValueError("cache_position, position_ids, and start_pos disagree")
            start_pos = supplied_starts[0] if supplied_starts else 0
            if bool(getattr(self.config, "strict_cache_positions", True)) and start_pos != 0:
                raise ValueError("generation without a prefix cache requires start_pos=0")

        # The model consumes scalar `start_pos`; forwarding arbitrary
        # `position_ids` would give the false impression that batch-specific
        # offsets are honored.
        position_ids = None

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "start_pos": start_pos,
        }
        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: Sequence[ScreeningCache], beam_idx: torch.LongTensor
    ) -> tuple[ScreeningCache, ...]:
        """Beam-search cache reordering."""

        reordered: list[ScreeningCache] = []
        for key_cache, value_cache in past_key_values:
            beam_idx_device = beam_idx.to(key_cache.device)
            reordered.append(
                (
                    key_cache.index_select(0, beam_idx_device),
                    value_cache.index_select(0, beam_idx_device.to(value_cache.device)),
                )
            )
        return tuple(reordered)


class MultiscreenLayer(nn.Module):
    """Single residual Multiscreen layer."""

    def __init__(self, config: MultiscreenConfig, layer_idx: int) -> None:
        super().__init__()
        self.block = GatedScreeningBlock(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kv: ScreeningCache | None = None,
        use_cache: bool = False,
        key_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ScreeningCache | None]:
        block_out, new_kv = self.block(
            x,
            start_pos=start_pos,
            past_kv=past_kv,
            use_cache=use_cache,
            key_attention_mask=key_attention_mask,
            query_attention_mask=query_attention_mask,
        )
        if query_attention_mask is not None:
            block_out = block_out * query_attention_mask.to(dtype=block_out.dtype).unsqueeze(-1)
        return x + block_out, new_kv


class GatedScreeningBlock(nn.Module):
    """Parallel gated screening tiles for one Multiscreen layer.

    Each tile performs Q/K/V/G projection, unit normalization, MiPE, independent
    screening, TanhNorm, bounded gating, per-head scaling, and output projection.
    """

    def __init__(self, config: MultiscreenConfig, layer_idx: int) -> None:
        super().__init__()
        d_e = config.hidden_size
        d_k = config.key_dim
        d_v = config.value_dim
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers

        self.layer_idx = layer_idx
        self.NH = num_heads
        self.dK = d_k
        self.dV = d_v
        self.wth = float(config.mipe_threshold)
        self.max_seq_len = int(config.max_position_embeddings)  # compatibility alias
        self.mipe_position_mode = str(config.mipe_position_mode)
        self.mipe_reference_wrap_boundary = int(config.mipe_reference_wrap_boundary)
        self.mipe_compute_dtype = str(config.mipe_compute_dtype)
        self.softmask_compute_dtype = str(config.softmask_compute_dtype)

        self.q_proj = nn.Linear(d_e, num_heads * d_k, bias=False)
        self.k_proj = nn.Linear(d_e, num_heads * d_k, bias=False)
        self.v_proj = nn.Linear(d_e, num_heads * d_v, bias=False)
        self.g_proj = nn.Linear(d_e, num_heads * d_v, bias=False)
        self.o_proj = nn.Linear(num_heads * d_v, d_e, bias=False)

        # Per-head learned parameters from the reference implementation.
        self.sw = nn.Parameter(torch.linspace(0, math.log(self.wth), num_heads))
        self.sr = nn.Parameter(torch.zeros(num_heads))
        self.sO = nn.Parameter(torch.full((num_heads,), math.log(1.0 / math.sqrt(num_heads * num_layers))))

        init = config.initializer_range
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=init / math.sqrt(d_k))
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=init / math.sqrt(d_k))
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=init / math.sqrt(d_v))
        nn.init.normal_(self.g_proj.weight, mean=0.0, std=init)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=init / math.sqrt(d_e))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kv: ScreeningCache | None = None,
        use_cache: bool = False,
        key_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ScreeningCache | None]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.NH, self.dK)
        k_new = self.k_proj(x).view(batch_size, seq_len, self.NH, self.dK)
        v_new = self.v_proj(x).view(batch_size, seq_len, self.NH, self.dV)
        g = self.g_proj(x).view(batch_size, seq_len, self.NH, self.dV)

        u, new_kv = self._screening(
            q=q,
            k_new=k_new,
            v_new=v_new,
            start_pos=start_pos,
            past_kv=past_kv,
            use_cache=use_cache,
            key_attention_mask=key_attention_mask,
        )

        g_hat = torch.tanh(F.silu(g))
        h = u * g_hat
        if query_attention_mask is not None:
            h = h * query_attention_mask.to(dtype=h.dtype).view(batch_size, seq_len, 1, 1)
        h = h * self.sO.exp().view(1, 1, self.NH, 1)
        h = h.reshape(batch_size, seq_len, self.NH * self.dV)
        return self.o_proj(h), new_kv

    def _screening(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int = 0,
        past_kv: ScreeningCache | None = None,
        use_cache: bool = False,
        key_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ScreeningCache | None]:
        """Screening unit with optional per-layer KV cache."""

        q = _unit_normalize(q)
        k_new = _unit_normalize(k_new)
        v_new = _unit_normalize(v_new)

        w = self.sw.exp() + 1.0
        r = self.sr.exp() + 1.0

        q, k_new = self._apply_mipe(q, k_new, w, start_pos=start_pos)

        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            full_k = torch.cat([past_k, k_new], dim=2)
            full_v = torch.cat([past_v, v_new], dim=2)
        else:
            full_k = k_new
            full_v = v_new

        seq_len = q.shape[2]
        total_length = full_k.shape[2]

        sim = torch.matmul(q, full_k.transpose(-2, -1))
        mask = self._softmask(
            T_new=seq_len,
            T_total=total_length,
            start_pos=start_pos,
            w=w,
            device=sim.device,
            dtype=sim.dtype,
            key_attention_mask=key_attention_mask,
        )

        rho_d = torch.clamp(
            1.0 - r.view(1, -1, 1, 1).to(dtype=sim.dtype) * (1.0 - sim),
            min=0.0,
        ).square_().mul_(mask)

        h = torch.matmul(rho_d, full_v)
        u = _tanh_norm(h, eps=1e-8)

        new_kv = (full_k, full_v) if use_cache else None
        return u.transpose(1, 2), new_kv

    @staticmethod
    def _select_compute_dtype(input_dtype: torch.dtype, mode: str) -> torch.dtype:
        if mode == "fp32":
            # Stable auxiliary math is an fp32 floor, not a float64 downcast.
            return (
                torch.float32 if input_dtype in {torch.float16, torch.bfloat16} else input_dtype
            )
        if mode == "reference":
            return input_dtype
        raise ValueError(f"Unknown Multiscreen compute dtype mode: {mode!r}")

    def _apply_mipe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimal positional encoding on the first two Q/K coordinates."""

        seq_len = q.shape[1]
        compute_dtype = self._select_compute_dtype(q.dtype, self.mipe_compute_dtype)
        w_float = w.to(device=q.device, dtype=compute_dtype)
        phi = torch.where(
            w_float < self.wth,
            0.5 * (torch.cos(math.pi * w_float / self.wth) + 1.0),
            torch.zeros_like(w_float),
        )

        positions = torch.arange(start_pos, start_pos + seq_len, device=q.device, dtype=compute_dtype)
        pos_2d = positions.unsqueeze(1)
        if self.mipe_position_mode == MIPE_POSITION_MODE_PAPER_ABSOLUTE:
            effective_pos = pos_2d.expand(-1, w_float.numel())
        elif self.mipe_position_mode == MIPE_POSITION_MODE_REFERENCE:
            w_2d = w_float.unsqueeze(0)
            effective_pos = torch.where(
                pos_2d >= self.mipe_reference_wrap_boundary,
                torch.remainder(pos_2d, w_2d),
                pos_2d,
            )
        else:  # Config validation should make this unreachable.
            raise RuntimeError(f"Unknown MiPE position mode: {self.mipe_position_mode!r}")
        angles = effective_pos * (math.pi * phi / w_float).unsqueeze(0)

        cos_a = torch.cos(angles).to(dtype=q.dtype)
        sin_a = torch.sin(angles).to(dtype=q.dtype)

        q0, q1 = q[..., 0], q[..., 1]
        k0, k1 = k[..., 0], k[..., 1]

        q_rot = torch.empty_like(q)
        q_rot[..., 0] = q0 * cos_a - q1 * sin_a
        q_rot[..., 1] = q0 * sin_a + q1 * cos_a
        q_rot[..., 2:] = q[..., 2:]

        k_rot = torch.empty_like(k)
        k_rot[..., 0] = k0 * cos_a - k1 * sin_a
        k_rot[..., 1] = k0 * sin_a + k1 * cos_a
        k_rot[..., 2:] = k[..., 2:]
        return q_rot, k_rot

    def _softmask(
        self,
        T_new: int,
        T_total: int,
        start_pos: int,
        w: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        key_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal distance-aware softmask for new queries over all keys."""

        compute_dtype = self._select_compute_dtype(dtype, self.softmask_compute_dtype)
        q_pos = torch.arange(start_pos, start_pos + T_new, device=device, dtype=compute_dtype)
        k_pos = torch.arange(T_total, device=device, dtype=compute_dtype)
        rel = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)

        w_exp = w.to(device=device, dtype=compute_dtype).view(-1, 1, 1)
        valid = (rel <= 0) & (rel > -w_exp)
        mask = (0.5 * (torch.cos(math.pi * rel / w_exp) + 1.0)) * valid
        mask = mask.unsqueeze(0).to(dtype=dtype)

        if key_attention_mask is not None:
            if key_attention_mask.shape[1] != T_total:
                raise ValueError(
                    f"key_attention_mask length {key_attention_mask.shape[1]} must equal total key length {T_total}"
                )
            mask = mask * key_attention_mask.to(device=device, dtype=dtype).view(-1, 1, 1, T_total)

        return mask
