# Third-Party Notices

## dieOD/multiscreen-pytorch

This repository vendors `dieOD/multiscreen-pytorch` under:

```text
third_party/multiscreen-pytorch/
```

It is used only as a reference implementation for P0-2 three-way comparison.

Original project metadata from its `pyproject.toml`:

```text
Homepage: https://github.com/dieOD/multiscreen-pytorch
License: Apache-2.0
Author: diodieide
```

The vendored copy retains its original `LICENSE` file. Any use or redistribution of that component must follow its Apache-2.0 terms.

## TinyStories tokenizer

The small tokenizer under `tokenizers/tinystories_spm768/` was produced with `scripts/train_tokenizer_spm.py` for smoke-test reproducibility. The training data source used in validation was `roneneldan/TinyStories`; check the dataset's own license/terms before redistributing derived artifacts in a public repository if required by your use case.
