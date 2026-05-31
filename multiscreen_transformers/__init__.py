"""Transformers-compatible Multiscreen implementation.

This package ports the core architecture from ``dieOD/multiscreen-pytorch`` to
Hugging Face Transformers-style ``PreTrainedConfig`` / ``PreTrainedModel``
classes.
"""

from .configuration_multiscreen import MultiscreenConfig
from .compile_utils import find_msvc_cl, load_vcvars_env, setup_compile_env
from .data import PackedTextDataset
from .modeling_multiscreen import (
    GatedScreeningBlock,
    MultiscreenForCausalLM,
    MultiscreenLayer,
    MultiscreenModel,
    MultiscreenPreTrainedModel,
    ScreeningCache,
    convert_original_state_dict_for_causal_lm,
    convert_original_state_dict_for_model,
)

__version__ = "0.1.2"

__all__ = [
    "MultiscreenConfig",
    "MultiscreenPreTrainedModel",
    "MultiscreenModel",
    "MultiscreenForCausalLM",
    "MultiscreenLayer",
    "GatedScreeningBlock",
    "ScreeningCache",
    "convert_original_state_dict_for_causal_lm",
    "convert_original_state_dict_for_model",
    "PackedTextDataset",
    "find_msvc_cl",
    "load_vcvars_env",
    "setup_compile_env",
    "register_multiscreen_auto_classes",
]


def register_multiscreen_auto_classes() -> None:
    """Register Multiscreen with Transformers auto classes in this process.

    Use this when loading local checkpoints without ``trust_remote_code`` and
    without installing the model into a Transformers source tree::

        from multiscreen_transformers import register_multiscreen_auto_classes
        register_multiscreen_auto_classes()

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("./checkpoint")
    """

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register(MultiscreenConfig.model_type, MultiscreenConfig)
    AutoModel.register(MultiscreenConfig, MultiscreenModel)
    AutoModelForCausalLM.register(MultiscreenConfig, MultiscreenForCausalLM)
