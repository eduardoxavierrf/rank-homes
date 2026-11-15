"""English command handlers; inference is injected only at composition."""
import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from .vision import VisualExtractor, create_extractor
from .workflows import build_references, compare_listings


def main(argv: Sequence[str] | None = None, *, extractor_factory: Callable[[], VisualExtractor] = create_extractor) -> int:
    parser = argparse.ArgumentParser(prog="rank-homes", description="Compare local rental photos against reference objects.")
    commands = parser.add_subparsers(dest="command", required=True)
    references = commands.add_parser("references", help="Manage reference libraries")
    builds = references.add_subparsers(dest="reference_command", required=True)
    build = builds.add_parser("build", help="Build a reference library")
    build.add_argument("--images", type=Path, required=True, help="Original reference image root")
    build.add_argument("--output", type=Path, required=True, help="Reference library destination")
    compare = commands.add_parser("compare", help="Rank listing photos against references")
    compare.add_argument("--references", type=Path, required=True, help="Reference library")
    compare.add_argument("--references-images", type=Path, required=True, help="Original reference image root")
    compare.add_argument("--listings", type=Path, required=True, help="Listing image root")
    compare.add_argument("--threshold", type=float, default=0.70, help="Strict score cutoff (default: 0.70)")
    args = parser.parse_args(argv)
    if args.command == "compare" and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    try:
        extractor = extractor_factory()
        if args.command == "references":
            generation = build_references(args.images, args.output, extractor)
            print(f"Published reference library: {args.output} (generation {generation})")
            return 0
        results = compare_listings(args.references, args.references_images, args.listings, extractor)
        selected = [result for result in results if result.score > args.threshold]
        for result in selected:
            print(f"Listing: {result.apartment} ({args.listings / result.apartment})")
            print(f"Score: {result.score:.4f}; Threshold: {args.threshold:.4f}")
            print(f"Comparison value: {result.score!r} > {args.threshold!r}")
            for room, score in result.room_scores:
                print(f"Room {room}: {score:.4f}")
                matches = [m for m in result.matches if m.room == room]
                for match in matches:
                    print(f"  Class {match.object_class}: {match.score:.4f}")
                    if match.candidate is not None and match.reference is not None:
                        print(f"  Candidate: {args.listings / match.candidate.photo.relative_path}; box {match.candidate.box}")
                        print(f"  Reference: {args.references_images / match.reference.photo.relative_path}; box {match.reference.box}")
                        for label, region in (("Candidate", match.candidate), ("Reference", match.reference)):
                            print(f"  {label} source: SHA-256 {region.photo.sha256}; oriented size {region.photo.width}x{region.photo.height}")
                    else:
                        print("  Warning: missing comparable evidence")
                pairs = [m for m in matches if m.candidate is not None]
                if pairs:
                    winner = max(pairs, key=lambda m: m.score)
                    assert winner.candidate is not None
                    print(f"Matching photo: {args.listings / winner.candidate.photo.relative_path} (one object match)")
        skipped = sum(path.is_dir() for path in args.listings.iterdir()) - len(results)
        print(f"Evaluated: {len(results)}; Selected: {len(selected)}; Below threshold: {len(results) - len(selected)}; Skipped: {skipped}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error during {args.command}: {error}", file=sys.stderr)
        return 1
