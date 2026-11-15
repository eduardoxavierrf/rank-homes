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
there is no synthetic command-line mode. Real offline inference is a subsequent ticket.

Reference builds hold an OS-managed writer lock for the complete command. Each
build writes a new generation under `generations/`, reloads and validates its
manifest, float32 NumPy matrix and original-size RGB PNG crops, then atomically
replaces `CURRENT.json`. Previous generations remain available; interrupted or
failed builds preserve the current pointer. Retry the complete build after a
failure. The persistent `.build.lock` file is normal and must not be removed;
the operating system releases the lock when the writer exits.

Comparison validates one captured generation before candidate inference. Every
eligible original reference file, including duplicates and photos without
detections, must retain its relative path and bytes. Moving only the reference
root is compatible. Inventory changes, corruption, unknown formats or extraction
configuration changes require an explicit rebuild. Changing `--threshold` does
not require a rebuild.

Extractor adapters must supply effective configuration: `extractor_version`,
`detector_weights`, `detector_weights_sha256`, `classes`, `detector_cutoff`,
`geometry`, `embedding_weights`, `embedding_weights_sha256`, `embedding_revision`,
`processor`, `pooling`, `dimension`, `normalization` (`L2`) and `dtype` (`float32`).
Weight hashes identify actual artifact bytes; the processor records the effective
resize, interpolation, rescale and normalization settings. The canonical JSON
SHA-256 fingerprint includes these settings and exact installed RF-DETR, Torch,
Torchvision, Transformers, Pillow and NumPy versions, without importing models.
Missing distributions are recorded explicitly for model-free adapters. Listing
threshold, presentation, device and hardware are excluded from extraction identity.

Run checks with `uv run pyright` and `uv run python -m unittest discover -s tests`.
