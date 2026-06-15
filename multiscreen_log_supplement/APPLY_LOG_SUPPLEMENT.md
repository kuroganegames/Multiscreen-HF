# Apply validation log supplement

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
