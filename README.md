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

Catalog discovery warns about photos outside the layout, keeps all eligible
reference files in its inventory, and extracts each byte-identical group once per
apartment and room. Unreadable references stop the build; unreadable candidates
warn and apartments without usable photos are skipped. Geometry uses oriented RGB
images, a shared detector cutoff (default 0.5), and clamped integer rectangles
with exclusive maxima. Evidence includes source hashes and oriented dimensions.

This is an application integration slice. Real inference is not yet
integrated: production search commands fail clearly without loading weights.
Tests supply a deterministic extractor through `main(..., extractor_factory=...)`;
there is no synthetic command-line mode. Reference integrity hardening and real offline inference are subsequent tickets.

Run checks with `uv run pyright` and `uv run python -m unittest discover -s tests`.
