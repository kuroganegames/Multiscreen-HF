"""Generate the deterministic P0.5-C1 paper-scale architecture manifest.

The manifest derives parameter counts from named tensor shapes and then checks
those shapes against allocation-safe meta-device model construction.  It never
uses ``MultiscreenConfig.num_params_estimate`` because that compatibility
helper intentionally omits learned scalar parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM


PAPER_ARXIV_ID = "2604.01178v3"
PAPER_URL = "https://arxiv.org/abs/2604.01178v3"
PAPER_TABLES = [1, 2, 4]
GENERATOR_RELATIVE_PATH = "scripts/generate_paper_scale_manifest.py"

VOCAB_SIZE = 50_257
KEY_DIM = 16
VALUE_DIM = 64
MIPE_THRESHOLD = 256.0
CONSTRUCTION_MAX_POSITION_EMBEDDINGS = 256
INITIALIZER_RANGE = 0.1

PAPER_COUNTS = {
    8: (4_134_146, 917_698),
    16: (27_546_626, 14_680_834),
    32: (286_347_266, 234_884_098),
    48: (1_304_884_226, 1_189_092_098),
    64: (3_963_961_346, 3_758_108_674),
}


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def expected_state_shapes(psi: int) -> dict[str, tuple[int, ...]]:
    """Return paper-derived HF state shapes without inspecting a model."""

    num_layers = psi
    num_heads = psi
    hidden_size = psi * psi
    shapes: dict[str, tuple[int, ...]] = {
        "multiscreen.embed.weight": (VOCAB_SIZE, hidden_size),
        "multiscreen.s_E": (),
        "multiscreen.s_F": (),
    }

    for layer_idx in range(num_layers):
        prefix = f"multiscreen.layers.{layer_idx}.block"
        shapes.update(
            {
                f"{prefix}.g_proj.weight": (num_heads * VALUE_DIM, hidden_size),
                f"{prefix}.k_proj.weight": (num_heads * KEY_DIM, hidden_size),
                f"{prefix}.o_proj.weight": (hidden_size, num_heads * VALUE_DIM),
                f"{prefix}.q_proj.weight": (num_heads * KEY_DIM, hidden_size),
                f"{prefix}.sO": (num_heads,),
                f"{prefix}.sr": (num_heads,),
                f"{prefix}.sw": (num_heads,),
                f"{prefix}.v_proj.weight": (num_heads * VALUE_DIM, hidden_size),
            }
        )
    return shapes


def independent_parameter_accounting(psi: int) -> dict[str, int]:
    """Derive counts from the paper's named matrix and scalar shapes."""

    num_layers = psi
    num_heads = psi
    hidden_size = psi * psi
    embedding = VOCAB_SIZE * hidden_size
    projection_per_tile = hidden_size * (2 * KEY_DIM + 3 * VALUE_DIM)
    scalar_per_tile = 3
    tile_count = num_layers * num_heads
    global_scalars = 2
    non_embedding = tile_count * (projection_per_tile + scalar_per_tile) + global_scalars
    return {
        "embedding_parameters": embedding,
        "global_scalar_parameters": global_scalars,
        "non_embedding_parameters": non_embedding,
        "projection_parameters_per_tile": projection_per_tile,
        "scalar_parameters_per_tile": scalar_per_tile,
        "tile_count": tile_count,
        "total_parameters": embedding + non_embedding,
    }


def _paper_config(psi: int) -> MultiscreenConfig:
    return MultiscreenConfig.from_psi(
        psi,
        vocab_size=VOCAB_SIZE,
        max_seq_len=CONSTRUCTION_MAX_POSITION_EMBEDDINGS,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        mipe_threshold=MIPE_THRESHOLD,
        initializer_range=INITIALIZER_RANGE,
    )


def _build_scale_record(psi: int) -> dict[str, Any]:
    expected_shapes = expected_state_shapes(psi)
    accounting = independent_parameter_accounting(psi)
    paper_total, paper_non_embedding = PAPER_COUNTS[psi]
    if accounting["total_parameters"] != paper_total:
        raise AssertionError(
            f"Psi={psi} independently derived total {accounting['total_parameters']} "
            f"does not match paper total {paper_total}"
        )
    if accounting["non_embedding_parameters"] != paper_non_embedding:
        raise AssertionError(
            f"Psi={psi} independently derived non-embedding count "
            f"{accounting['non_embedding_parameters']} does not match paper count "
            f"{paper_non_embedding}"
        )

    config = _paper_config(psi)
    expected_config = {
        "hidden_size": psi * psi,
        "initializer_range": INITIALIZER_RANGE,
        "key_dim": KEY_DIM,
        "max_position_embeddings": CONSTRUCTION_MAX_POSITION_EMBEDDINGS,
        "mipe_threshold": MIPE_THRESHOLD,
        "num_attention_heads": psi,
        "num_hidden_layers": psi,
        "tie_word_embeddings": True,
        "value_dim": VALUE_DIM,
        "vocab_size": VOCAB_SIZE,
    }
    actual_config = {name: getattr(config, name) for name in expected_config}
    if actual_config != expected_config:
        raise AssertionError(
            f"Psi={psi} config mismatch: expected {expected_config!r}, got {actual_config!r}"
        )

    with torch.device("meta"):
        model = MultiscreenForCausalLM(config)

    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    state_dict = model.state_dict()
    if not all(parameter.device.type == "meta" for parameter in named_parameters.values()):
        raise AssertionError(f"Psi={psi} allocated a non-meta parameter")
    if not all(buffer.device.type == "meta" for buffer in named_buffers.values()):
        raise AssertionError(f"Psi={psi} allocated a non-meta buffer")
    if not all(tensor.device.type == "meta" for tensor in state_dict.values()):
        raise AssertionError(f"Psi={psi} produced a non-meta state tensor")

    actual_shapes = {key: tuple(tensor.shape) for key, tensor in state_dict.items()}
    if actual_shapes != expected_shapes:
        missing = sorted(set(expected_shapes) - set(actual_shapes))
        unexpected = sorted(set(actual_shapes) - set(expected_shapes))
        mismatched = sorted(
            key
            for key in set(expected_shapes) & set(actual_shapes)
            if expected_shapes[key] != actual_shapes[key]
        )
        raise AssertionError(
            f"Psi={psi} state-shape mismatch: missing={missing}, "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )

    parameter_count = sum(parameter.numel() for parameter in named_parameters.values())
    state_count = sum(tensor.numel() for tensor in state_dict.values())
    shape_count = sum(_numel(shape) for shape in expected_shapes.values())
    expected_total = accounting["total_parameters"]
    if parameter_count != expected_total or state_count != expected_total or shape_count != expected_total:
        raise AssertionError(
            f"Psi={psi} count mismatch: parameters={parameter_count}, state={state_count}, "
            f"named_shapes={shape_count}, expected={expected_total}"
        )
    if any(name.startswith("lm_head.") for name in named_parameters):
        raise AssertionError(f"Psi={psi} has a separate trainable lm_head parameter")

    entries = [
        {
            "key": key,
            "numel": _numel(expected_shapes[key]),
            "shape": list(expected_shapes[key]),
        }
        for key in sorted(expected_shapes)
    ]
    return {
        "allocation": {
            "all_buffers_meta": True,
            "all_parameters_meta": True,
            "all_state_tensors_meta": True,
            "device": "meta",
            "real_weight_elements_allocated": 0,
        },
        "config": expected_config,
        "parameter_accounting": {
            **accounting,
            "paper_non_embedding_parameters": paper_non_embedding,
            "paper_total_parameters": paper_total,
        },
        "psi": psi,
        "state_dict": {
            "entries": entries,
            "key_count": len(entries),
            "parameter_elements": state_count,
        },
    }


def build_manifest() -> dict[str, Any]:
    """Build and validate the complete deterministic C1 manifest."""

    return {
        "artifact": "P0.5-C1 paper architecture and state-shape manifest",
        "contract": {
            "initializer_range": INITIALIZER_RANGE,
            "key_dim": KEY_DIM,
            "mipe_threshold": MIPE_THRESHOLD,
            "parameter_formula": {
                "non_embedding": "N_L*N_H*(d_E*(2*d_K+3*d_V)+3)+2",
                "total": "vocab_size*d_E+N_L*N_H*(d_E*(2*d_K+3*d_V)+3)+2",
            },
            "scaling": "N_L=N_H=Psi; d_E=Psi^2",
            "value_dim": VALUE_DIM,
            "vocab_size": VOCAB_SIZE,
        },
        "construction": {
            "max_position_embeddings": CONSTRUCTION_MAX_POSITION_EMBEDDINGS,
            "position_semantics_contract": False,
        },
        "generator": GENERATOR_RELATIVE_PATH,
        "invariants": {
            "projection_biases": False,
            "separate_trainable_lm_head": False,
            "tie_word_embeddings": True,
        },
        "scales": [_build_scale_record(psi) for psi in PAPER_COUNTS],
        "schema_version": 1,
        "source": {
            "arxiv_id": PAPER_ARXIV_ID,
            "paper_tables": PAPER_TABLES,
            "paper_url": PAPER_URL,
        },
    }


def render_manifest(manifest: dict[str, Any] | None = None) -> bytes:
    if manifest is None:
        manifest = build_manifest()
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="write the canonical manifest to this path")
    destination.add_argument(
        "--check",
        type=Path,
        help="fail unless this file exactly matches the regenerated canonical bytes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered = render_manifest()
    if args.check is not None:
        try:
            existing = args.check.read_bytes()
        except FileNotFoundError:
            print(f"manifest does not exist: {args.check}", file=sys.stderr)
            return 2
        if existing != rendered:
            print(
                f"manifest drift: regenerate {args.check} with "
                f"{GENERATOR_RELATIVE_PATH} --output {args.check}",
                file=sys.stderr,
            )
            return 1
        print(f"paper-scale manifest matches: {args.check}")
        return 0
    if args.output is not None:
        args.output.write_bytes(rendered)
        print(f"wrote paper-scale manifest: {args.output}")
        return 0
    sys.stdout.buffer.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
