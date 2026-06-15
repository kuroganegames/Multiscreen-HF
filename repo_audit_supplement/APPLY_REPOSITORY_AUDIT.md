# Apply repository audit document

Copy `docs/REPOSITORY_AUDIT.md` into the root of `kuroganegames/Multiscreen-HF`.

```bash
unzip repository_audit_supplement.zip
cp -r repository_audit_supplement/. /path/to/Multiscreen-HF/
cd /path/to/Multiscreen-HF
```

Then add links from README and HANDOFF if they are not already present.

Suggested README line:

```markdown
For repository hygiene and release-readiness checks, see [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md).
```

Suggested HANDOFF line:

```markdown
For final repository hygiene checks, see [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md).
```

Commit:

```bash
git add docs/REPOSITORY_AUDIT.md README.md docs/HANDOFF.md
git commit -m "Add repository audit documentation"
```
