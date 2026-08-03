# Apply validation log supplement

> Historical pre-P0-4 import instructions. Do not apply this bundle to the
> current tree; use the canonical [validation log index](../docs/validation_results/VALIDATION_LOG_INDEX.md).

Copy this supplement into the repository root:

```bash
unzip multiscreen_log_supplement.zip
cp -r multiscreen_log_supplement/. /path/to/Multiscreen-HF/
```

Then add links from `README.md` or `docs/VALIDATION_STATUS.md` if desired:

```markdown
For compact validation run summaries, see [docs/validation_results/VALIDATION_LOG_INDEX.md](docs/validation_results/VALIDATION_LOG_INDEX.md).
For future logging rules, see [docs/LOGGING_POLICY.md](docs/LOGGING_POLICY.md).
```

Recommended commit message:

```text
Add compact P0 validation logs and logging policy
```
