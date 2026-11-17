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

Ranking averages only observed, comparable object classes within each reference
room, then averages rooms equally. A room with no comparable classes scores zero.
Observed zero-similarity matches count in the average; missing classes remain in
the diagnostics but do not count. This can favor sparse evidence. Historical
saved results used missing-class zero penalties and do not reflect this formula.
The default threshold remains 0.70 and needs calibration for the new scores.
This ranking change does not require rebuilding reference embeddings.

Catalog discovery warns about photos outside the layout, keeps all eligible
reference files in its inventory, and extracts each byte-identical group once per
apartment and room. Unreadable references stop the build; unreadable candidates
warn and apartments without usable photos are skipped. Geometry uses oriented RGB
images, a shared detector cutoff (default 0.5), and clamped integer rectangles
with exclusive maxima. Confidence equal to the cutoff is included. Evidence
includes source hashes and oriented dimensions.

Search runs entirely from staged local artifacts. Defaults are
`artifacts/runs/rooms-expanded-small/checkpoint_best_total.pth` and
`artifacts/weights/dinov3-vits16-lvd`, relative to the working directory.
Both commands accept `--detector <checkpoint>`, `--embedding-model <directory>`,
`--detector-confidence 0.5` and `--device cpu|cuda`. Without a device override,
CUDA is selected when available, otherwise CPU. Each command loads each model
once, uses float32, evaluation/inference mode and batch size one, and stops on
model failure without changing device or precision. Import and help load no models.

The detector must be the canonical 512-pixel RF-DETR Small with the full ordered
23-class mapping, including distinct shower, shower stall and shower door labels.
Legacy Nano checkpoints and incomplete/incompatible weights are rejected. DINOv3
must be the approved pinned ViT-S/16 LVD snapshot with its validated artifact
manifest; see [staging instructions](evidence/offline-dinov3/README.md).
Search never acquires weights or imports preparation/training scripts. It keeps
candidate detections, crops and vectors in memory only. Device and extraction
fingerprint are printed to stderr; the reference manifest records effective
model/processor settings, artifact hashes and installed dependency versions.

The [real offline search evidence](evidence/offline-search/README.md) covers both
rooms, command reuse and rejection of stale references. Detection quality remains
a limitation; successful execution does not establish retrieval accuracy.

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
The real search smoke is opt-in: set `RANK_HOMES_REAL_SMOKE=1` and run
`uv run python -m unittest discover -s tests -p test_offline_search_smoke.py`.
It requires the default local artifacts and real `data/references` and
`data/candidates` catalogs, blocks socket connections, and uses a temporary library.
