# Validation Environment Template

Use this file to record the exact environment for any future validation gate.
Do not edit historical validation records after the fact; copy this template into
`docs/validation_results/` or into the relevant experiment output directory.

## Hardware

```text
Host:
OS:
CPU:
RAM:
GPU 0:
GPU 1:
Power limits:
PCIe topology:
Driver:
CUDA runtime:
```

Helpful commands:

```bash
uname -a
lscpu | head -40
free -h
nvidia-smi
nvidia-smi topo -m
```

## Python environment

```bash
python - <<'PY'
import sys, torch
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

python - <<'PY'
try:
    import transformers; print('transformers:', transformers.__version__)
except Exception as e: print('transformers: unavailable', e)
try:
    import datasets; print('datasets:', datasets.__version__)
except Exception as e: print('datasets: unavailable', e)
try:
    import tokenizers; print('tokenizers:', tokenizers.__version__)
except Exception as e: print('tokenizers: unavailable', e)
PY

pip freeze > validation_pip_freeze.txt
conda env export > validation_conda_env.yml  # if using conda
```

## Validation run

```text
Validation gate:
Commit SHA:
Tag:
Command:
Dataset/cache path:
Tokenizer path:
Device:
Dtype:
Result:
```

## Notes

```text
Known deviations from documented command:
Warnings observed:
Failures and fixes:
```
