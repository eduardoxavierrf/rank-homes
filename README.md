# Rank Homes

Install with `uv sync --locked`. Preparation, training and prediction scripts
remain in `scripts/` and run with `uv run python scripts/<script>.py`.

The search command surface is available through `uv run rank-homes --help`
or `uv run python -m rank_homes --help`:

```text
rank-homes references build --images data/references --output artifacts/references
rank-homes compare --references artifacts/references --references-images data/references --listings data/candidates --threshold 0.70
```

Inputs use `<root>/<apartment>/kitchen/<photo>` or
`<root>/<apartment>/bathroom/<photo>`. Original reference images remain required
for comparison and evidence paths.

This is the minimal application integration slice. Real inference is not yet
integrated: production search commands fail clearly without loading weights.
Tests supply a deterministic extractor through `main(..., extractor_factory=...)`;
there is no synthetic command-line mode. Broader catalog handling, geometry,
reference integrity and real offline inference are subsequent tickets.

Run checks with `uv run pyright` and `uv run python -m unittest discover -s tests`.
